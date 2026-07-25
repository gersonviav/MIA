"""
Filtrado UCF-Crime — frames preprocesados → 3 clases
=====================================================
Versión para datasets donde los videos ya están extraídos como frames (.jpg/.png).

Estructura esperada:
    ucf_crime/
        Train/
            Abuse/         ← frames: Abuse028_x264_10.jpg, _20.jpg ...
            Assault/
            Burglary/
            Explosion/
            Fighting/
            NormalVideos/
            Robbery/
            Shoplifting/
            Stealing/
            Vandalism/
            ...
        Test/
            (mismas carpetas)

Uso:
    python filter_dataset_3class.py --dry-run
    python filter_dataset_3class.py
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

UCF_ROOT    = Path("ucf_crime")
OUTPUT_ROOT = Path("ucf_crime_3class")

UCF_MAPPING = {
    "Fighting": ["Assault", "Abuse", "Fighting"],
    "Vandalism": ["Robbery", "Arson", "Burglary", "Explosion",
                  "Shoplifting", "Stealing", "Vandalism"],
    "NormalVideos": ["NormalVideos"],
}

CLASS_TO_IDX = {"NormalVideos": 0, "Fighting": 1, "Vandalism": 2}

BRIGHTNESS_SAMPLE_FRAMES = 5
MIN_BRIGHTNESS           = 30
MAX_PER_CLASS = {"Fighting": 150, "Vandalism": 150, "NormalVideos": 200}
RANDOM_SEED   = 42
IMG_EXTS      = {".jpg", ".jpeg", ".png"}


# ─────────────────────────────────────────────────────────────────────────────
# AGRUPAR FRAMES POR VIDEO
# ─────────────────────────────────────────────────────────────────────────────

def group_frames_by_video(folder: Path) -> dict:
    groups = defaultdict(list)
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in IMG_EXTS:
            parts = f.stem.rsplit("_", 1)
            key   = parts[0] if len(parts) == 2 and parts[1].isdigit() else f.stem
            groups[key].append(f)
    for key in groups:
        groups[key].sort(key=lambda p: int(p.stem.rsplit("_", 1)[-1])
                         if p.stem.rsplit("_", 1)[-1].isdigit() else 0)
    return dict(groups)


def estimate_brightness_frames(frame_paths: list, n: int = BRIGHTNESS_SAMPLE_FRAMES) -> float:
    if not frame_paths:
        return -1.0
    indices = np.linspace(0, len(frame_paths) - 1, min(n, len(frame_paths)), dtype=int)
    vals = []
    for idx in indices:
        img = cv2.imread(str(frame_paths[idx]))
        if img is not None:
            vals.append(float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))))
    return float(np.mean(vals)) if vals else -1.0


# ─────────────────────────────────────────────────────────────────────────────
# ESCANEO
# ─────────────────────────────────────────────────────────────────────────────

def scan_folder(folder: Path, class_name: str, label: int, split_name: str) -> list:
    if not folder.exists():
        print(f"      [AVISO] No existe: {folder}")
        return []
    groups = group_frames_by_video(folder)
    if not groups:
        print(f"      [AVISO] Sin frames en: {folder}")
        return []

    accepted, n_dark, n_short = [], 0, 0
    for video_id, frames in groups.items():
        if len(frames) < 8:
            n_short += 1
            continue
        b = estimate_brightness_frames(frames)
        if b < MIN_BRIGHTNESS:
            n_dark += 1
            continue
        accepted.append({
            "video_id":   video_id,
            "class_name": class_name,
            "label":      label,
            "split":      split_name,
            "frames":     [str(p) for p in frames],
            "brightness": round(b, 1),
            "n_frames":   len(frames),
        })

    print(f"      {folder.name:15s} ({split_name}): "
          f"{len(groups):4d} videos → {len(accepted):3d} ok  "
          f"({n_dark} oscuros, {n_short} cortos)")
    return accepted


# ─────────────────────────────────────────────────────────────────────────────
# PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def build_subset(dry_run: bool = False):
    rng = np.random.default_rng(RANDOM_SEED)

    print("\n" + "=" * 62)
    print("  Filtrado UCF-Crime (frames) → 3 clases")
    print("=" * 62)

    available_splits = [s for s in ["Train", "Test"] if (UCF_ROOT / s).exists()]
    if not available_splits:
        print(f"\n[ERROR] No se encontraron carpetas Train/ ni Test/ en '{UCF_ROOT}/'")
        print(f"  Verifica que el script esté en la misma carpeta que 'ucf_crime/'")
        print(f"  o ajusta UCF_ROOT al inicio del script.")
        return

    print(f"  Splits encontrados: {available_splits}")

    all_by_class = {}
    for class_name, ucf_folders in UCF_MAPPING.items():
        label = CLASS_TO_IDX[class_name]
        print(f"\n[{class_name}]  (label={label})")
        videos = []
        for folder_name in ucf_folders:
            for split in available_splits:
                folder = UCF_ROOT / split / folder_name
                videos.extend(scan_folder(folder, class_name, label, split))

        max_v = MAX_PER_CLASS.get(class_name, 9999)
        if len(videos) > max_v:
            idx    = rng.choice(len(videos), max_v, replace=False)
            videos = [videos[i] for i in sorted(idx)]
            print(f"    → Limitado a {max_v} (MAX_PER_CLASS)")
        all_by_class[class_name] = videos
        print(f"    Total {class_name}: {len(videos)} videos")

    all_videos   = [v for vl in all_by_class.values() for v in vl]
    total        = len(all_videos)
    total_frames = sum(v["n_frames"] for v in all_videos)

    print("\n" + "─" * 62)
    print(f"  TOTAL: {total} videos")
    for cn, vl in all_by_class.items():
        pct = len(vl) / total * 100 if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {cn:15s}: {len(vl):4d}  {bar} {pct:.0f}%")
    print(f"\n  Total frames en disco : {total_frames:,}")
    vram_gb = (8 * 32 * 3 * 224 * 224 * 2 * 6) / (1024**3)
    print(f"  VRAM estimada (fp16, batch=8): {vram_gb:.1f} GB  ✓ OK para RTX 3060 12GB")

    if dry_run:
        print("\n[DRY RUN] No se copiaron archivos.\n")
        return

    # ── Split ──────────────────────────────────────────────────
    if len(available_splits) == 2:
        train_videos = [v for v in all_videos if v["split"] == "Train"]
        test_videos  = [v for v in all_videos if v["split"] == "Test"]
        print(f"\n  Split original UCF: {len(train_videos)} train / {len(test_videos)} test")
    else:
        idx          = rng.permutation(total)
        n_train      = int(total * 0.8)
        train_videos = [all_videos[i] for i in idx[:n_train]]
        test_videos  = [all_videos[i] for i in idx[n_train:]]
        print(f"\n  Split aleatorio 80/20: {len(train_videos)} train / {len(test_videos)} test")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "Annotations").mkdir(exist_ok=True)

    def write_split_file(videos, filename):
        lines = []
        for v in videos:
            frames_str = "|".join(v["frames"])
            lines.append(f"{v['video_id']} {v['label']} {v['class_name']} {frames_str}")
        (OUTPUT_ROOT / "Annotations" / filename).write_text("\n".join(lines))
        print(f"  Guardado: {filename}  ({len(lines)} entradas)")

    print("\nGenerando split files...")
    write_split_file(train_videos, "train_split.txt")
    write_split_file(test_videos,  "test_split.txt")

    class_counts = {cn: len(vl) for cn, vl in all_by_class.items()}
    weight_map   = {cn: total / (len(UCF_MAPPING) * max(cnt, 1))
                    for cn, cnt in class_counts.items()}
    sample_weights_train = [weight_map[v["class_name"]] for v in train_videos]

    metadata = {
        "class_to_idx":         CLASS_TO_IDX,
        "ucf_mapping":          UCF_MAPPING,
        "ucf_root":             str(UCF_ROOT),
        "total_videos":         total,
        "train_videos":         len(train_videos),
        "test_videos":          len(test_videos),
        "class_counts":         class_counts,
        "class_weights":        weight_map,
        "sample_weights_train": sample_weights_train,
        "total_frames":         total_frames,
    }
    (OUTPUT_ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\n✓ Listo en: {OUTPUT_ROOT}/")
    print("  Próximo paso: python train_3class.py --mode train\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build_subset(dry_run=args.dry_run)