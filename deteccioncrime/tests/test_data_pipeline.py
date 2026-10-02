"""Prueba end-to-end de ingesta + preprocesamiento sobre un dataset sintético pequeño."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from vad.config import load_config
from vad.data.common import read_json, read_jsonl
from vad.data.ingest import run_ingest
from vad.data.preprocess import run_preprocess

ROOT = Path(__file__).resolve().parents[1]


def _make_video(folder: Path, vid: str, n: int, value: int, rng):
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        img = np.clip(rng.normal(value, 10, (32, 32, 3)), 0, 255).astype(np.uint8)
        cv2.imwrite(str(folder / f"{vid}_{k * 10}.png"), img)


@pytest.fixture()
def cfg(tmp_path):
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw" / "ucf_crime"
    for split, n_vid in (("Train", 6), ("Test", 2)):
        for cat in ("NormalVideos", "Fighting", "Abuse", "Burglary", "Arrest"):
            for i in range(n_vid):
                _make_video(raw / split / cat, f"{cat}{split}{i:03d}_x264", 10, 120, rng)
    _make_video(raw / "Train" / "Fighting", "FightingShort_x264", 4, 120, rng)   # corto
    _make_video(raw / "Train" / "Burglary", "BurglaryDark_x264", 10, 5, rng)     # oscuro
    (tmp_path / "ann.txt").write_text("FightingTest000_x264.mp4 Fighting 20 60 -1 -1\n")
    doc = tmp_path / "DATASET.md"
    doc.write_text("# x\n<!-- AUTOGEN:START -->\n<!-- AUTOGEN:END -->\n")

    c = load_config(ROOT / "configs" / "default.yaml")
    c["paths"].update(raw_dir=str(raw), temporal_annotations=str(tmp_path / "ann.txt"),
                      manifests_dir=str(tmp_path / "manifests"),
                      processed_dir=str(tmp_path / "processed"), dataset_doc=str(doc))
    c["preprocess"]["max_per_class_train"] = {"NormalVideos": 5, "Fighting": 5, "Vandalism": 5}
    return c


def test_pipeline_end_to_end(cfg, tmp_path):
    v1 = run_ingest(cfg, download_date="2025-01-01")
    assert v1["totals"]["videos"] == 5 * 8 + 2
    assert v1["temporal_annotations"]["videos"] == 1
    assert "SHA-256 del dataset" in (tmp_path / "DATASET.md").read_text()

    meta = run_preprocess(cfg)
    out = Path(cfg["paths"]["processed_dir"])
    splits = {k: read_jsonl(out / "splits" / f"{k}.jsonl") for k in ("train", "val", "test")}

    # tope por clase aplicado solo a train+val
    assert len(splits["train"]) + len(splits["val"]) == 15
    # test = oficial completo (Arrest fuera del mapeo)
    assert len(splits["test"]) == 4 * 2
    # sin fuga
    ids = [{v["video_id"] for v in s} for s in splits.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    # exclusiones registradas
    excl = (out / "excluded_videos.csv").read_text()
    assert "FightingShort_x264" in excl and "BurglaryDark_x264" in excl
    # linaje
    pv = read_json(Path(cfg["paths"]["manifests_dir"]) / "processed_dataset_version.json")
    assert pv["raw_dataset_sha256"] == v1["dataset_sha256"]

    # reproducibilidad: mismos hashes al repetir
    v2 = run_ingest(cfg, download_date="2025-01-01")
    assert v2["dataset_sha256"] == v1["dataset_sha256"]
    assert run_preprocess(cfg)["processed_dataset_sha256"] == meta["processed_dataset_sha256"]
