"""Utilidades compartidas por ingesta y preprocesamiento (sin dependencia de torch)."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# 13 categorías de anomalía + NormalVideos del paper original (Sultani et al., 2018)
UCF_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    "NormalVideos", "RoadAccidents", "Robbery", "Shooting", "Shoplifting",
    "Stealing", "Vandalism",
]

MANIFEST_NAME = "raw_manifest.csv.gz"
RAW_VERSION_NAME = "raw_dataset_version.json"
PROCESSED_VERSION_NAME = "processed_dataset_version.json"
MANIFEST_COLUMNS = ["split", "category", "video_id", "frame_idx", "rel_path", "bytes", "sha256"]


def parse_frame_name(stem: str) -> tuple[str, int | None]:
    """'Abuse028_x264_120' → ('Abuse028_x264', 120). Sin sufijo numérico → (stem, None)."""
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return stem, None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path) -> list[dict]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_manifest_by_video(rows: list[dict]) -> dict[tuple[str, str, str], list[tuple[int, str]]]:
    """(split, category, video_id) → [(frame_idx, rel_path), ...] ordenado por frame_idx."""
    groups: dict[tuple[str, str, str], list[tuple[int, str]]] = defaultdict(list)
    for r in rows:
        idx = int(r["frame_idx"]) if r["frame_idx"] not in ("", None) else 0
        groups[(r["split"], r["category"], r["video_id"])].append((idx, r["rel_path"]))
    for k in groups:
        groups[k].sort(key=lambda t: (t[0], t[1]))
    return dict(groups)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
