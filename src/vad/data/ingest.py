"""
Ingesta del dataset crudo (data/raw)
====================================
1. Valida la estructura  data/raw/ucf_crime/{Train,Test}/<Categoria>/<video>_<frame>.jpg|png
2. Inventaria todos los frames (split, categoría, video, nº de frame, tamaño).
3. Calcula SHA-256 por archivo y un hash global del dataset (versión).
4. Mide la resolución de una muestra de imágenes.
5. Escribe:
     data/manifests/raw_manifest.csv.gz        (inventario completo; no se versiona en git)
     data/manifests/raw_dataset_version.json   (resumen + hashes; SÍ se versiona en git)
   y actualiza el bloque AUTOGEN de docs/DATASET.md.
6. Si ya existía una versión previa, informa si el dataset cambió.
"""

from __future__ import annotations

import logging
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2

from vad import __version__
from vad.data.common import (
    MANIFEST_NAME, RAW_VERSION_NAME, UCF_CATEGORIES, human_bytes, parse_frame_name,
    read_json, sha256_file, sha256_lines, write_json, write_manifest,
)

log = logging.getLogger("vad.ingest")

AUTOGEN_START = "<!-- AUTOGEN:START -->"
AUTOGEN_END = "<!-- AUTOGEN:END -->"


def _scan(raw_dir: Path, img_exts: set[str]) -> tuple[list[dict], dict]:
    rows, issues = [], {"unparsed_names": 0, "ignored_files": 0, "unknown_categories": set()}
    splits = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    for split_dir in splits:
        for cat_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if cat_dir.name not in UCF_CATEGORIES:
                issues["unknown_categories"].add(f"{split_dir.name}/{cat_dir.name}")
            n_cat = 0
            for f in cat_dir.iterdir():
                if not f.is_file():
                    continue
                if f.suffix.lower() not in img_exts:
                    issues["ignored_files"] += 1
                    continue
                video_id, frame_idx = parse_frame_name(f.stem)
                if frame_idx is None:
                    issues["unparsed_names"] += 1
                rows.append({
                    "split": split_dir.name,
                    "category": cat_dir.name,
                    "video_id": video_id,
                    "frame_idx": "" if frame_idx is None else frame_idx,
                    "rel_path": f.relative_to(raw_dir).as_posix(),
                    "bytes": f.stat().st_size,
                    "sha256": "",
                })
                n_cat += 1
            log.info("  %-6s / %-14s %8d frames", split_dir.name, cat_dir.name, n_cat)
    rows.sort(key=lambda r: r["rel_path"])
    issues["unknown_categories"] = sorted(issues["unknown_categories"])
    return rows, issues


def _hash_all(rows: list[dict], raw_dir: Path, workers: int) -> None:
    total = len(rows)
    step = max(total // 10, 1)

    def work(i: int) -> None:
        rows[i]["sha256"] = sha256_file(raw_dir / rows[i]["rel_path"])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, _ in enumerate(ex.map(work, range(total)), 1):
            if n % step == 0 or n == total:
                log.info("  hashing %d/%d (%.0f%%)", n, total, 100 * n / total)


def _resolution_stats(rows: list[dict], raw_dir: Path, n: int) -> dict:
    rng = random.Random(0)
    sample = rng.sample(rows, min(n, len(rows)))
    sizes, unreadable = Counter(), 0
    for r in sample:
        img = cv2.imread(str(raw_dir / r["rel_path"]), cv2.IMREAD_UNCHANGED)
        if img is None:
            unreadable += 1
            continue
        h, w = img.shape[:2]
        c = 1 if img.ndim == 2 else img.shape[2]
        sizes[f"{w}x{h}x{c}"] += 1
    return {"sampled": len(sample), "unreadable": unreadable,
            "resolutions_WxHxC": dict(sizes.most_common())}


def _summaries(rows: list[dict]) -> tuple[dict, dict]:
    per_split = defaultdict(lambda: {"videos": set(), "frames": 0, "bytes": 0})
    per_cat = defaultdict(lambda: {"videos": set(), "frames": 0, "bytes": 0, "lines": []})
    for r in rows:
        key = f"{r['split']}/{r['category']}"
        for d in (per_split[r["split"]], per_cat[key]):
            d["videos"].add(r["video_id"])
            d["frames"] += 1
            d["bytes"] += int(r["bytes"])
        per_cat[key]["lines"].append(f"{r['rel_path']}\t{r['sha256'] or r['bytes']}")

    split_out = {s: {"videos": len(d["videos"]), "frames": d["frames"], "bytes": d["bytes"]}
                 for s, d in sorted(per_split.items())}
    cat_out = {k: {"videos": len(d["videos"]), "frames": d["frames"], "bytes": d["bytes"],
                   "sha256": sha256_lines(d["lines"])}
               for k, d in sorted(per_cat.items())}
    return split_out, cat_out


def _update_dataset_doc(doc_path: Path, version: dict) -> None:
    if not doc_path.exists():
        log.warning("No existe %s; no se actualizó la documentación.", doc_path)
        return
    text = doc_path.read_text(encoding="utf-8")
    if AUTOGEN_START not in text or AUTOGEN_END not in text:
        log.warning("%s no tiene marcadores AUTOGEN; no se actualizó.", doc_path)
        return

    t = version["totals"]
    lines = [
        AUTOGEN_START,
        f"_Generado automáticamente por `vad ingest` el {version['ingested_at']}. No editar a mano._",
        "",
        "| Campo | Valor |",
        "|---|---|",
        f"| Fuente | {version['source_url']} |",
        f"| Fecha de descarga | {version['download_date'] or '⚠️ SIN REGISTRAR (usar --download-date)'} |",
        f"| Fecha de ingesta | {version['ingested_at']} |",
        f"| Ruta | `{version['raw_dir']}` |",
        f"| Videos | {t['videos']:,} |",
        f"| Frames (archivos) | {t['frames']:,} |",
        f"| Tamaño en disco | {t['bytes_human']} ({t['bytes']:,} bytes) |",
        f"| Resolución (muestra) | {', '.join(version['resolution']['resolutions_WxHxC']) or 'n/d'} |",
        f"| Modo de hash | {version['hash_mode']} |",
        f"| **SHA-256 del dataset** | `{version['dataset_sha256']}` |",
        f"| Anotaciones temporales | {'sí — `' + version['temporal_annotations']['sha256'][:16] + '…`' if version['temporal_annotations'] else 'no encontradas'} |",
        "",
        "| Split / Categoría | Videos | Frames | Tamaño |",
        "|---|---:|---:|---:|",
    ]
    for k, v in version["per_category"].items():
        lines.append(f"| {k} | {v['videos']:,} | {v['frames']:,} | {human_bytes(v['bytes'])} |")
    lines.append(AUTOGEN_END)

    pre = text.split(AUTOGEN_START)[0]
    post = text.split(AUTOGEN_END, 1)[1]
    doc_path.write_text(pre + "\n".join(lines) + post, encoding="utf-8")
    log.info("Documentación actualizada: %s", doc_path)


def run_ingest(cfg: dict, download_date: str | None = None, skip_hash: bool = False) -> dict:
    paths, ds = cfg["paths"], cfg["dataset"]
    raw_dir = Path(paths["raw_dir"])
    manifests_dir = Path(paths["manifests_dir"])
    download_date = download_date or ds.get("download_date")

    log.info("=" * 70)
    log.info("INGESTA — %s", ds["name"])
    log.info("=" * 70)
    log.info("raw_dir: %s", raw_dir.resolve())

    if not raw_dir.exists():
        log.error("No existe %s. Copia ahí los frames de UCF-Crime (Train/ y Test/). "
                  "Ver docs/DATASET.md.", raw_dir)
        sys.exit(1)

    split_names = [p.name for p in raw_dir.iterdir() if p.is_dir()]
    log.info("Splits encontrados: %s", split_names)
    for expected in ("Train", "Test"):
        if expected not in split_names:
            log.warning("Falta el split '%s' en %s", expected, raw_dir)

    # 1. Inventario
    log.info("[1/4] Inventariando frames...")
    rows, issues = _scan(raw_dir, {e.lower() for e in ds["img_exts"]})
    if not rows:
        log.error("No se encontraron imágenes en %s", raw_dir)
        sys.exit(1)
    if issues["unknown_categories"]:
        log.warning("Categorías no reconocidas: %s", issues["unknown_categories"])
    if issues["unparsed_names"]:
        log.warning("%d archivos sin sufijo _<frame> en el nombre", issues["unparsed_names"])
    if issues["ignored_files"]:
        log.info("%d archivos ignorados (extensión no válida)", issues["ignored_files"])

    # 2. Hash
    if skip_hash:
        hash_mode = "size-only (--skip-hash; NO garantiza integridad)"
        log.warning("[2/4] Hash por archivo OMITIDO (--skip-hash). La versión se basa en ruta+tamaño.")
    else:
        hash_mode = "sha256 por archivo"
        log.info("[2/4] Calculando SHA-256 de %d archivos (%d hilos)...",
                 len(rows), cfg["ingest"]["hash_workers"])
        _hash_all(rows, raw_dir, cfg["ingest"]["hash_workers"])
    dataset_sha = sha256_lines(f"{r['rel_path']}\t{r['sha256'] or r['bytes']}" for r in rows)

    # 3. Resolución
    log.info("[3/4] Midiendo resolución en una muestra...")
    resolution = _resolution_stats(rows, raw_dir, cfg["ingest"]["resolution_sample"])
    log.info("  %s", resolution)

    # 4. Resúmenes y salida
    log.info("[4/4] Escribiendo manifest y versión...")
    per_split, per_cat = _summaries(rows)
    total_bytes = sum(int(r["bytes"]) for r in rows)
    n_videos = len({(r["split"], r["category"], r["video_id"]) for r in rows})

    ann_path = Path(paths["temporal_annotations"])
    ann = None
    if ann_path.exists():
        n_lines = sum(1 for l in ann_path.read_text(encoding="utf-8").splitlines() if l.strip())
        ann = {"path": ann_path.as_posix(), "sha256": sha256_file(ann_path), "videos": n_lines}
        log.info("Anotaciones temporales encontradas: %s (%d videos)", ann_path, n_lines)
    else:
        log.info("Sin anotaciones temporales (%s). Solo necesarias para localización.", ann_path)

    version = {
        "dataset": ds["name"],
        "source_url": ds["source_url"],
        "download_date": download_date,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_dir": raw_dir.as_posix(),
        "hash_mode": hash_mode,
        "dataset_sha256": dataset_sha,
        "totals": {"videos": n_videos, "frames": len(rows), "bytes": total_bytes,
                   "bytes_human": human_bytes(total_bytes)},
        "per_split": per_split,
        "per_category": per_cat,
        "resolution": resolution,
        "issues": issues,
        "temporal_annotations": ann,
        "tool_version": __version__,
    }

    version_path = manifests_dir / RAW_VERSION_NAME
    if version_path.exists():
        old = read_json(version_path)
        if old.get("dataset_sha256") == dataset_sha:
            log.info("Dataset SIN cambios respecto a la versión registrada.")
        else:
            log.warning("El dataset CAMBIÓ: %s… → %s…",
                        str(old.get("dataset_sha256"))[:12], dataset_sha[:12])
            write_json(old, manifests_dir / f"raw_dataset_version.prev_{old.get('dataset_sha256', 'x')[:12]}.json")

    write_manifest(rows, manifests_dir / MANIFEST_NAME)
    write_json(version, version_path)
    _update_dataset_doc(Path(paths["dataset_doc"]), version)

    if not download_date:
        log.warning("Fecha de descarga no registrada. Usa: vad ingest --download-date YYYY-MM-DD")

    log.info("-" * 70)
    log.info("Videos: %d | Frames: %d | Tamaño: %s", n_videos, len(rows), human_bytes(total_bytes))
    log.info("SHA-256 dataset: %s", dataset_sha)
    log.info("Manifest: %s", manifests_dir / MANIFEST_NAME)
    log.info("Versión : %s", version_path)
    return version
