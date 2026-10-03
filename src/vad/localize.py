"""
Localización temporal: ¿en qué segundos ocurre el evento?
=========================================================
Para cada video:
  1. score por snippet (≈1 s) con ventanas deslizantes (vad.inference.score_video)
  2. snippets con score ≥ umbral, unidos si son contiguos → intervalos [inicio_s, fin_s]
  3. si hay anotación temporal (solo Test UCF-Crime): tIoU entre lo detectado y lo real

Salidas en artifacts/reports/localization_<split>/:
  snippets.csv   (video, snippet, inicio_s, fin_s, score)
  summary.csv    (video, clase real/predicha, intervalos detectados, intervalos reales, tIoU)
  metrics.json   <video_id>.png (con --video-id o --plot-all)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vad.config import class_names
from vad.data.common import write_json

log = logging.getLogger("vad.localize")


def load_temporal_annotations(path: Path) -> dict[str, list[tuple[int, int]]]:
    """Formato UCF-Crime: <video>.mp4 <Categoria> s1 e1 s2 e2 (frames originales; -1 = sin intervalo)."""
    ann = {}
    if not path.exists():
        return ann
    for line in path.read_text(encoding="utf-8").splitlines():
        tok = line.split()
        if len(tok) < 4:
            continue
        vid = tok[0].rsplit(".", 1)[0]
        nums = [int(x) for x in tok[2:]]
        ann[vid] = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)
                    if nums[i] >= 0 and nums[i + 1] >= 0]
    return ann


def temporal_iou(pred: list[tuple[float, float]], gt: list[tuple[float, float]], res: float = 0.1) -> float:
    """IoU entre la unión de intervalos predichos y la de intervalos reales (en segundos)."""
    if not pred and not gt:
        return 1.0
    end = max([e for _, e in pred + gt] + [0.0])
    grid = np.arange(0, end + res, res)

    def mask(iv):
        m = np.zeros_like(grid, dtype=bool)
        for a, b in iv:
            m |= (grid >= a) & (grid < b)
        return m

    p, g = mask(pred), mask(gt)
    union = (p | g).sum()
    return float((p & g).sum() / union) if union else 0.0


def _plot(vid, label, pred, times, scores, detected, gt_s, thr, path, raw=None):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for a, b in gt_s:
        ax.axvspan(a, b, color="#4ECDC4", alpha=0.3, label="evento real")
    for a, b, _ in detected:
        ax.axvspan(a, b, ymin=0, ymax=0.06, color="#E24B4A", label="detectado")
    if raw is not None and not np.allclose(raw, scores):
        ax.plot(times, raw, color="#E24B4A", lw=0.8, alpha=0.35, label="score sin suavizar")
    ax.plot(times, scores, color="#E24B4A", lw=1.8, label="score anomalía")
    ax.axhline(thr, color="gray", ls="--", lw=1, label=f"umbral {thr}")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("segundo del video")
    ax.set_ylabel("score")
    ax.set_title(f"{vid} — real: {label} | pred: {pred}")
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def run_localize(cfg: dict, checkpoint: str, split: str = "test", video_id: str | None = None,
                 plot_all: bool = False) -> dict:
    from vad.data.feature_dataset import load_split
    from vad.inference import get_device, load_model, score_video
    from vad.temporal import intervals_from_scores

    device = get_device()
    names = class_names(cfg)
    fps = float(cfg["dataset"]["fps"])
    lc = cfg["localize"]
    out_dir = Path(cfg["paths"]["reports_dir"]) / f"localization_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _ = load_model(cfg, checkpoint, device)
    samples = load_split(cfg, split)
    if video_id:
        samples = [s for s in samples if s["video_id"] == video_id]
        if not samples:
            raise SystemExit(f"video_id '{video_id}' no está en el split {split}")
    ann = load_temporal_annotations(Path(cfg["paths"]["temporal_annotations"]))
    log.info("Localizando %d videos | umbral %.2f | suavizado %s snippets | unión de huecos ≤ %s s | "
             "anotaciones: %d", len(samples), lc["threshold"], lc.get("smooth_snippets", 1),
             lc.get("merge_gap_s", 0.0), len(ann))

    snip_rows, sum_rows, tious, fp_normals = [], [], [], []
    for s in samples:
        r = score_video(model, cfg, s["video_id"], device)
        starts, ends = r["ranges"][:, 0] / fps, r["ranges"][:, 1] / fps
        detected = intervals_from_scores(r["scores"], r["ranges"], fps, lc["threshold"], lc["min_duration_s"],
                                         lc.get("merge_gap_s", 0.0))
        gt_s = [(a / fps, (b + 1) / fps) for a, b in ann.get(s["video_id"], [])]
        tiou = None
        if s["label"] != 0 and gt_s:
            tiou = temporal_iou([(a, b) for a, b, _ in detected], gt_s)
            tious.append(tiou)
        if s["label"] == 0:
            fp_normals.append(len(detected) > 0)

        for k, (a, b, sc) in enumerate(zip(starts, ends, r["scores"])):
            snip_rows.append([s["video_id"], k, f"{a:.2f}", f"{b:.2f}", f"{sc:.4f}"])
        sum_rows.append([s["video_id"], s["class_name"], names[r["pred"]], f"{r['probs'].max():.3f}",
                         f"{starts[int(np.argmax(r['scores']))]:.1f}", f"{r['scores'].max():.3f}",
                         "; ".join(f"{a:.1f}-{b:.1f}s" for a, b, _ in detected),
                         "; ".join(f"{a:.1f}-{b:.1f}s" for a, b in gt_s),
                         "" if tiou is None else f"{tiou:.3f}"])
        if video_id or (plot_all and s["label"] != 0):
            _plot(s["video_id"], s["class_name"], names[r["pred"]], (starts + ends) / 2, r["scores"],
                  detected, gt_s, lc["threshold"], out_dir / f"{s['video_id']}.png", raw=r["scores_raw"])
        if video_id:
            log.info("%s | real %s | pred %s | detectado: %s | real: %s", s["video_id"], s["class_name"],
                     names[r["pred"]], sum_rows[-1][6] or "nada", sum_rows[-1][7] or "sin anotación")

    with open(out_dir / "snippets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "snippet", "start_s", "end_s", "score"])
        w.writerows(snip_rows)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label", "pred", "pred_prob", "peak_time_s", "peak_score",
                    "detected_intervals", "gt_intervals", "tiou"])
        w.writerows(sum_rows)

    metrics = {"videos": len(samples), "threshold": lc["threshold"],
               "smooth_snippets": lc.get("smooth_snippets", 1), "merge_gap_s": lc.get("merge_gap_s", 0.0),
               "anomalous_with_gt": len(tious),
               "mean_tiou": float(np.mean(tious)) if tious else None,
               "tiou_ge_0.5": float(np.mean(np.array(tious) >= 0.5)) if tious else None,
               "normal_videos_with_false_alarm": float(np.mean(fp_normals)) if fp_normals else None}
    write_json(metrics, out_dir / "metrics.json")
    log.info("tIoU medio %s | videos con tIoU≥0.5 %s | normales con falsa alarma %s",
             metrics["mean_tiou"], metrics["tiou_ge_0.5"], metrics["normal_videos_with_false_alarm"])
    log.info("Salidas en %s", out_dir)
    return metrics
