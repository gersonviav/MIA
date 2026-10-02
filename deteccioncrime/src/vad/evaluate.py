"""
Evaluación final sobre TEST ciego
=================================
1. Nivel video (clasificación 3 clases): accuracy, P/R/F1, macro AUC, matriz de confusión, ROC.
   + binario normal vs anómalo (accuracy, recall, especificidad, AUC).
2. Nivel frame (¿CUÁNDO?): AUC y AP frame a frame con las anotaciones temporales de UCF-Crime.
   Cada frame original recibe el score del snippet que lo contiene (ventanas deslizantes).

Salidas en artifacts/reports/<split>/:
  metrics.json  predictions.csv  confusion_matrix.png  roc_curves.png  frame_roc.png
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (auc, average_precision_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)

from vad.config import class_names
from vad.data.common import write_json
from vad.data.feature_dataset import load_split
from vad.inference import get_device, load_model, score_video
from vad.localize import load_temporal_annotations
from vad.temporal import frame_level, gt_frame_labels

log = logging.getLogger("vad.evaluate")

COLORS = ["#4ECDC4", "#FF6B6B", "#FFD93D"]
BG, PANEL = "#0A1628", "#0F2240"


def plot_confusion_matrix(y_true, y_pred, names, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)
    fig.suptitle("Matriz de Confusión — Detección de Anomalías UCF-Crime",
                 color="white", fontsize=13, fontweight="bold", y=1.02)
    for ax, data, title, fmt in zip(axes, [cm, cm_norm],
                                    ["Conteos absolutos", "Normalizada (recall por clase)"], ["d", ".2f"]):
        ax.set_facecolor(PANEL)
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=max(data.max(), 1e-9))
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, color="white", fontsize=11, rotation=15)
        ax.set_yticklabels(names, color="white", fontsize=11)
        ax.set_xlabel("Predicción", color="#7A9CC4", fontsize=11)
        ax.set_ylabel("Real", color="#7A9CC4", fontsize=11)
        ax.set_title(title, color="#B8D4F0", fontsize=10, pad=10)
        for i in range(len(names)):
            for j in range(len(names)):
                v = data[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=13, fontweight="bold",
                        color="white" if v < data.max() * 0.6 else BG)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    return cm


def plot_roc_curves(y_true, y_probs, names, save_path):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(PANEL)
    for i, (name, color) in enumerate(zip(names, COLORS)):
        y_bin = (y_true == i).astype(int)
        if y_bin.min() == y_bin.max():
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="#7A9CC4", lw=1, linestyle="--", label="Azar (AUC = 0.500)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Tasa de Falsos Positivos", color="#B8D4F0", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos", color="#B8D4F0", fontsize=11)
    ax.set_title("Curvas ROC por clase (One-vs-Rest)", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(loc="lower right", facecolor="#162D4E", edgecolor="#7A9CC4", labelcolor="white", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()


def plot_frame_roc(y, s, value, save_path):
    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#FF6B6B", label=f"AUC frame = {value:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("ROC a nivel de frame (localización temporal)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_evaluate(cfg: dict, checkpoint: str, split: str = "test") -> dict:
    device = get_device()
    names = class_names(cfg)
    out_dir = Path(cfg["paths"]["reports_dir"]) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model(cfg, checkpoint, device)
    samples = load_split(cfg, split)
    ann = load_temporal_annotations(Path(cfg["paths"]["temporal_annotations"]))
    log.info("Evaluando %d videos de %s | anotaciones temporales: %d | suavizado del score: %s snippets",
             len(samples), split, len(ann), cfg.get("localize", {}).get("smooth_snippets", 1))

    y_true, y_pred, probs, rows, max_scores = [], [], [], [], []
    fy, fs, fy_anom, fs_anom, missing_gt = [], [], [], [], []
    for n, s in enumerate(samples, 1):
        r = score_video(model, cfg, s["video_id"], device)
        y_true.append(s["label"]); y_pred.append(r["pred"]); probs.append(r["probs"])
        max_scores.append(float(r["scores"].max()))
        rows.append([s["video_id"], names[s["label"]], names[r["pred"]], *[f"{p:.5f}" for p in r["probs"]],
                     f"{r['scores'].max():.4f}"])

        frames_s = frame_level(r["scores"], r["ranges"])
        if s["label"] == 0:
            gt = np.zeros(len(frames_s), dtype=np.int8)
        elif s["video_id"] in ann:
            gt = gt_frame_labels(len(frames_s), ann[s["video_id"]])
            fy_anom.append(gt); fs_anom.append(frames_s)
        else:
            missing_gt.append(s["video_id"])
            continue
        fy.append(gt); fs.append(frames_s)
        if n % 25 == 0:
            log.info("  %d/%d videos", n, len(samples))

    y_true, y_pred, probs = np.array(y_true), np.array(y_pred), np.array(probs)
    m = {"split": split, "n_videos": len(samples),
         "video_level": {
             "accuracy": float((y_true == y_pred).mean()),
             "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
             "report": classification_report(y_true, y_pred, labels=list(range(len(names))), target_names=names,
                                             digits=3, zero_division=0, output_dict=True)}}
    try:
        m["video_level"]["macro_auc_ovr"] = float(roc_auc_score(y_true, probs, multi_class="ovr",
                                                                average="macro", labels=list(range(len(names)))))
    except ValueError as e:
        log.warning("AUC video no calculable: %s", e)
    yb, pb = (y_true != 0).astype(int), (y_pred != 0).astype(int)
    binary = {"accuracy": float((yb == pb).mean()),
              "recall_anomalia": float(pb[yb == 1].mean()) if yb.any() else float("nan"),
              "especificidad_normal": float(1 - pb[yb == 0].mean()) if (yb == 0).any() else float("nan"),
              "precision_anomalia": float(yb[pb == 1].mean()) if pb.any() else float("nan")}
    if 0 < yb.sum() < len(yb):
        binary["auc_1_menos_p_normal"] = float(roc_auc_score(yb, 1 - probs[:, 0]))
        binary["auc_max_snippet_score"] = float(roc_auc_score(yb, np.array(max_scores)))
    m["video_level_binary"] = binary
    cm = plot_confusion_matrix(y_true, y_pred, names, out_dir / "confusion_matrix.png")
    plot_roc_curves(y_true, probs, names, out_dir / "roc_curves.png")
    m["video_level"]["confusion_matrix"] = cm.tolist()

    fl = {"videos_used": len(fy), "anomalous_without_gt": missing_gt}
    if fy and len(np.unique(np.concatenate(fy))) == 2:
        Y, S = np.concatenate(fy), np.concatenate(fs)
        fl.update(auc=float(roc_auc_score(Y, S)), ap=float(average_precision_score(Y, S)),
                  n_frames=int(len(Y)), anomalous_frame_ratio=float(Y.mean()))
        if fy_anom and len(np.unique(np.concatenate(fy_anom))) == 2:
            fl["auc_anomalous_only"] = float(roc_auc_score(np.concatenate(fy_anom), np.concatenate(fs_anom)))
        plot_frame_roc(Y, S, fl["auc"], out_dir / "frame_roc.png")
    else:
        log.warning("AUC frame no calculable: faltan anotaciones temporales o videos anómalos con GT.")
    if missing_gt:
        log.warning("%d videos anómalos sin anotación temporal (excluidos del AUC frame)", len(missing_gt))
    fl["smooth_snippets"] = cfg.get("localize", {}).get("smooth_snippets", 1)
    m["frame_level"] = fl
    m["checkpoint"] = {"path": str(checkpoint), "epoch": ckpt.get("epoch"),
                       "data_version": ckpt.get("data_version", {}).get("processed_dataset_sha256")}
    write_json(m, out_dir / "metrics.json")

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label", "pred", *[f"p_{n}" for n in names], "max_snippet_score"])
        w.writerows(rows)

    vl = m["video_level"]
    log.info("=" * 60)
    log.info("RESULTADOS (%s)", split.upper())
    log.info("Video  | acc %.4f | macro-F1 %.4f | macro AUC %s", vl["accuracy"], vl["macro_f1"],
             f"{vl.get('macro_auc_ovr', float('nan')):.4f}")
    for n in names:
        r = vl["report"][n]
        log.info("  %-13s P %.3f  R %.3f  F1 %.3f", n, r["precision"], r["recall"], r["f1-score"])
    b = m["video_level_binary"]
    log.info("Binario (normal vs anómalo) | acc %.4f | recall anomalía %.3f | especificidad %.3f | "
             "AUC(1-P normal) %s | AUC(max score) %s", b["accuracy"], b["recall_anomalia"],
             b["especificidad_normal"], f"{b.get('auc_1_menos_p_normal', float('nan')):.4f}",
             f"{b.get('auc_max_snippet_score', float('nan')):.4f}")
    if "auc" in fl:
        log.info("Frame  | AUC %.4f | AP %.4f | AUC solo anómalos %s | %s frames",
                 fl["auc"], fl["ap"], f"{fl.get('auc_anomalous_only', float('nan')):.4f}", f"{fl['n_frames']:,}")
    log.info("Salidas en %s", out_dir)
    return m
