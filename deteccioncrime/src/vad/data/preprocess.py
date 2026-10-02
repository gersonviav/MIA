"""
Preprocesamiento data/raw → data/processed
==========================================
Parte del manifest generado por `vad ingest` (garantiza que se procesa la versión registrada).

1. Agrupa frames por video y aplica filtros de calidad:
     - < min_frames frames            → excluido ("corto")
     - brillo medio < min_brightness  → excluido ("oscuro")
2. Remapea las categorías UCF-Crime a 3 macro-clases.
3. Train (split oficial UCF): aplica el tope por clase y separa validación estratificada.
   Test  (split oficial UCF): se conserva completo y NO se usa para seleccionar modelos.
4. Verifica que no haya fuga (videos repetidos entre splits).
5. Escribe en data/processed/ucf_crime_3class/:
     splits/{train,val,test}.jsonl   metadata.json   excluded_videos.csv
   y data/manifests/processed_dataset_version.json (hash + linaje con el dataset crudo).
"""

from __future__ import annotations

import csv
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from vad import __version__
from vad.config import snapshot
from vad.data.common import (
    MANIFEST_NAME, PROCESSED_VERSION_NAME, RAW_VERSION_NAME, group_manifest_by_video,
    read_json, read_manifest, sha256_file, sha256_lines, write_json, write_jsonl,
)

log = logging.getLogger("vad.preprocess")

SPLIT_FILES = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}


def _brightness(raw_dir: Path, rel_paths: list[str], n: int) -> float:
    idx = np.linspace(0, len(rel_paths) - 1, min(n, len(rel_paths)), dtype=int)
    vals = []
    for i in idx:
        img = cv2.imread(str(raw_dir / rel_paths[i]))
        if img is not None:
            vals.append(float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))))
    return float(np.mean(vals)) if vals else -1.0


def _norm_split(name: str) -> str | None:
    n = name.lower()
    if n.startswith("train"):
        return "Train"
    if n.startswith("test"):
        return "Test"
    return None


def run_preprocess(cfg: dict) -> dict:
    paths, pp = cfg["paths"], cfg["preprocess"]
    raw_dir = Path(paths["raw_dir"])
    manifests_dir = Path(paths["manifests_dir"])
    out_dir = Path(paths["processed_dir"])
    rng = np.random.default_rng(pp["seed"])

    log.info("=" * 70)
    log.info("PREPROCESAMIENTO → 3 clases")
    log.info("=" * 70)

    manifest_path = manifests_dir / MANIFEST_NAME
    version_path = manifests_dir / RAW_VERSION_NAME
    if not manifest_path.exists() or not version_path.exists():
        log.error("No hay manifest del dataset crudo. Ejecuta primero: vad ingest")
        sys.exit(1)
    raw_version = read_json(version_path)
    log.info("Dataset crudo: %s… (%s frames, ingestado %s)",
             raw_version["dataset_sha256"][:16], f"{raw_version['totals']['frames']:,}",
             raw_version["ingested_at"])

    rows = read_manifest(manifest_path)
    missing = [r["rel_path"] for r in rows[:: max(len(rows) // 200, 1)]
               if not (raw_dir / r["rel_path"]).exists()]
    if missing:
        log.error("%d archivos del manifest no existen en disco (ej. %s). "
                  "¿Cambió data/raw? Vuelve a correr `vad ingest`.", len(missing), missing[0])
        sys.exit(1)

    groups = group_manifest_by_video(rows)
    cat_to_class = {cat: cls for cls, cats in pp["class_mapping"].items() for cat in cats}
    c2i = pp["class_to_idx"]

    # 1-2. Filtros + remapeo
    log.info("[1/4] Filtrando videos (min_frames=%d, min_brightness=%d)...",
             pp["min_frames"], pp["min_brightness"])
    accepted = {"Train": {c: [] for c in c2i}, "Test": {c: [] for c in c2i}}
    excluded, stats = [], Counter()
    keys = sorted(k for k in groups if k[1] in cat_to_class)
    ignored_cats = sorted({k[1] for k in groups if k[1] not in cat_to_class})
    if ignored_cats:
        log.info("Categorías fuera del mapeo (se ignoran): %s", ignored_cats)

    for n, (split, cat, vid) in enumerate(keys, 1):
        split_norm = _norm_split(split)
        if split_norm is None:
            continue
        frames = groups[(split, cat, vid)]
        rel = [p for _, p in frames]
        cls = cat_to_class[cat]
        base = {"video_id": vid, "ucf_category": cat, "class_name": cls, "source_split": split_norm}

        if len(rel) < pp["min_frames"]:
            excluded.append({**base, "reason": "corto", "n_frames": len(rel), "brightness": ""})
            stats[(split_norm, cls, "corto")] += 1
            continue
        b = _brightness(raw_dir, rel, pp["brightness_sample_frames"])
        if b < pp["min_brightness"]:
            excluded.append({**base, "reason": "oscuro" if b >= 0 else "ilegible",
                             "n_frames": len(rel), "brightness": round(b, 1)})
            stats[(split_norm, cls, "oscuro")] += 1
            continue
        accepted[split_norm][cls].append({
            **base, "label": c2i[cls], "n_frames": len(rel), "brightness": round(b, 1),
            "frame_indices": [i for i, _ in frames], "frames": rel,
        })
        stats[(split_norm, cls, "ok")] += 1
        if n % 200 == 0:
            log.info("  %d/%d videos evaluados", n, len(keys))

    for split_norm in ("Train", "Test"):
        for cls in c2i:
            log.info("  %-5s %-13s ok=%4d  cortos=%3d  oscuros=%3d", split_norm, cls,
                     stats[(split_norm, cls, "ok")], stats[(split_norm, cls, "corto")],
                     stats[(split_norm, cls, "oscuro")])

    # 3. Tope por clase (solo Train) + split train/val estratificado
    log.info("[2/4] Tope por clase en Train y separación train/val (val_ratio=%.2f)...", pp["val_ratio"])
    splits = {"train": [], "val": [], "test": []}
    for cls in sorted(c2i, key=c2i.get):
        pool = accepted["Train"][cls]
        cap = pp["max_per_class_train"].get(cls)
        if cap and len(pool) > cap:
            idx = np.sort(rng.choice(len(pool), cap, replace=False))
            pool = [pool[i] for i in idx]
            log.info("  %-13s limitado a %d (había %d)", cls, cap, len(accepted["Train"][cls]))
        perm = rng.permutation(len(pool))
        n_val = int(round(len(pool) * pp["val_ratio"]))
        splits["val"] += [{**pool[i], "split": "val"} for i in sorted(perm[:n_val])]
        splits["train"] += [{**pool[i], "split": "train"} for i in sorted(perm[n_val:])]
        splits["test"] += [{**v, "split": "test"} for v in accepted["Test"][cls]]

    if not splits["test"]:
        log.warning("Test vacío: no se encontró el split Test en data/raw.")

    # 4. Verificación de fuga
    log.info("[3/4] Verificando ausencia de fuga entre splits...")
    ids = {k: {v["video_id"] for v in vs} for k, vs in splits.items()}
    leaks = {f"{a}∩{b}": sorted(ids[a] & ids[b])
             for a, b in [("train", "val"), ("train", "test"), ("val", "test")] if ids[a] & ids[b]}
    if leaks:
        log.error("FUGA DETECTADA entre splits: %s", {k: len(v) for k, v in leaks.items()})
        sys.exit(1)
    log.info("  OK — sin videos compartidos entre train/val/test")

    # 5. Escritura
    log.info("[4/4] Escribiendo salidas en %s...", out_dir)
    split_dir = out_dir / "splits"
    counts = {}
    for k, fname in SPLIT_FILES.items():
        write_jsonl(splits[k], split_dir / fname)
        counts[k] = dict(sorted(Counter(v["class_name"] for v in splits[k]).items()))
        log.info("  %-5s %4d videos  %s", k, len(splits[k]), counts[k])

    with open(out_dir / "excluded_videos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "ucf_category", "class_name",
                                          "source_split", "reason", "n_frames", "brightness"])
        w.writeheader()
        w.writerows(excluded)

    n_train = len(splits["train"])
    class_weights = {cls: n_train / (len(c2i) * max(counts["train"].get(cls, 0), 1)) for cls in c2i}

    split_hashes = {k: sha256_file(split_dir / f) for k, f in SPLIT_FILES.items()}
    processed_sha = sha256_lines(f"{k}\t{h}" for k, h in sorted(split_hashes.items()))

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_dataset_sha256": raw_version["dataset_sha256"],
        "processed_dataset_sha256": processed_sha,
        "raw_dir": raw_dir.as_posix(),
        "class_to_idx": c2i,
        "class_mapping": pp["class_mapping"],
        "counts": counts,
        "n_videos": {k: len(v) for k, v in splits.items()},
        "n_excluded": len(excluded),
        "class_weights_train": class_weights,
        "preprocess_config": snapshot(cfg)["preprocess"],
        "tool_version": __version__,
    }
    write_json(metadata, out_dir / "metadata.json")
    write_json({
        "processed_dataset_sha256": processed_sha,
        "split_files_sha256": split_hashes,
        "raw_dataset_sha256": raw_version["dataset_sha256"],
        "created_at": metadata["created_at"],
        "n_videos": metadata["n_videos"],
        "counts": counts,
        "preprocess_config": metadata["preprocess_config"],
    }, manifests_dir / PROCESSED_VERSION_NAME)

    log.info("-" * 70)
    log.info("Excluidos: %d (ver excluded_videos.csv)", len(excluded))
    log.info("SHA-256 dataset procesado: %s", processed_sha)
    log.info("Siguiente paso: vad train")
    return metadata
