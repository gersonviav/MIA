"""Smoke test etapa 2 (train → evaluate → localize) con features sintéticas (sin backbones)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vad.config import load_config
from vad.data.common import write_json, write_jsonl
from vad.evaluate import run_evaluate
from vad.localize import run_localize
from vad.train import run_train

ROOT = Path(__file__).resolve().parents[1]
D = 16


def test_mil_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    c = load_config(ROOT / "configs" / "default.yaml")
    feat = tmp_path / "feat"
    feat.mkdir()
    c["paths"].update(processed_dir=str(tmp_path / "proc"), manifests_dir=str(tmp_path / "man"),
                      checkpoints_dir=str(tmp_path / "ckpt"), reports_dir=str(tmp_path / "rep"),
                      temporal_annotations=str(tmp_path / "ann.txt"))
    c["features"]["dir"] = str(feat)
    c["model"].update(d_model=32, nhead=4, num_layers=1)
    c["mil"].update(num_segments=8, window=8, stride=4)
    c["train"].update(epochs=3, batch_size=8)
    write_json({"dim_rgb": D, "dim_mot": D}, feat / "index.json")

    ann, splits = [], {"train": [], "val": [], "test": []}
    for split, n in (("train", 8), ("val", 3), ("test", 3)):
        for label, cls in enumerate(["NormalVideos", "Fighting", "Vandalism"]):
            for i in range(n):
                vid = f"{cls}{split}{i}"
                nf = int(rng.integers(40, 90))
                rgb = rng.normal(0, 1, (nf, D))
                if label:
                    a = int(rng.integers(5, nf - 15))
                    rgb[a:a + 10] += 3 * label
                    if split == "test":
                        ann.append(f"{vid}.mp4 {cls} {a * 10} {(a + 10) * 10} -1 -1")
                fi = np.arange(nf) * 10
                np.savez(feat / f"{vid}.npz", rgb=rgb.astype(np.float16), mot=rgb.astype(np.float16),
                         frame_indices=fi)
                splits[split].append({"video_id": vid, "label": label, "class_name": cls,
                                      "n_frames": nf, "frame_indices": fi.tolist(), "frames": []})
    for k, v in splits.items():
        write_jsonl(v, tmp_path / "proc" / "splits" / f"{k}.jsonl")
    (tmp_path / "ann.txt").write_text("\n".join(ann) + "\n")

    assert run_train(c)["best_epoch"] >= 1
    ckpt = str(tmp_path / "ckpt" / "best_model.pth")
    m = run_evaluate(c, ckpt)
    assert "auc" in m["frame_level"] and 0 <= m["frame_level"]["auc"] <= 1
    loc = run_localize(c, ckpt)
    assert loc["anomalous_with_gt"] == 6
    assert (tmp_path / "rep" / "localization_test" / "summary.csv").exists()
