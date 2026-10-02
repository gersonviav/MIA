"""
Evaluación final — Matriz de confusión + métricas completas
============================================================
Carga el mejor checkpoint y genera:
  - Matriz de confusión (imagen PNG)
  - Classification report completo
  - AUC por clase (OvR)
  - Curvas ROC por clase (imagen PNG)

Uso:
    python eval_final.py --checkpoint checkpoints_3class/best_model.pth
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)

# Importar el modelo y dataset desde train_3class
from train import (
    UCFCrime3ClassDataset, AnomalyDetector3Class, CFG
)

CLASS_NAMES  = ["NormalVideos", "Fighting", "Vandalism"]
COLORS       = ["#4ECDC4", "#FF6B6B", "#FFD93D"]
OUTPUT_DIR   = Path("eval_outputs")

# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, loader, device, use_fp16):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for rgb, motion, labels, _ in loader:
            rgb    = rgb.to(device, non_blocking=True)
            motion = motion.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_fp16):
                logits = model(rgb, motion)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()   # ← agrega .float()
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    return (np.array(all_labels), np.array(all_preds), np.array(all_probs))


# ─────────────────────────────────────────────────────────────────────────────
# MATRIZ DE CONFUSIÓN
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             facecolor="#0A1628")
    fig.suptitle("Matriz de Confusión — Detección de Anomalías UCF-Crime",
                 color="white", fontsize=13, fontweight="bold", y=1.02)

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Conteos absolutos", "Normalizada (recall por clase)"],
        ["d", ".2f"]
    ):
        ax.set_facecolor("#0F2240")
        im = ax.imshow(data, cmap="Blues", vmin=0,
                       vmax=data.max())
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, color="white", fontsize=11, rotation=15)
        ax.set_yticklabels(CLASS_NAMES, color="white", fontsize=11)
        ax.set_xlabel("Predicción", color="#7A9CC4", fontsize=11)
        ax.set_ylabel("Real",       color="#7A9CC4", fontsize=11)
        ax.set_title(title, color="#B8D4F0", fontsize=10, pad=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#162D4E")

        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val      = data[i, j]
                text_col = "white" if val < data.max() * 0.6 else "#0A1628"
                ax.text(j, i, format(val, fmt),
                        ha="center", va="center",
                        color=text_col, fontsize=13, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0A1628")
    plt.close()
    print(f"  Guardado: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CURVAS ROC
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(y_true, y_probs, save_path):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0A1628")
    ax.set_facecolor("#0F2240")

    for i, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        y_bin        = (y_true == i).astype(int)
        fpr, tpr, _  = roc_curve(y_bin, y_probs[:, i])
        roc_auc_val  = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC = {roc_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color="#7A9CC4", lw=1,
            linestyle="--", label="Azar (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Tasa de Falsos Positivos", color="#B8D4F0", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos", color="#B8D4F0", fontsize=11)
    ax.set_title("Curvas ROC por clase (One-vs-Rest)",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#162D4E")
    leg = ax.legend(loc="lower right", facecolor="#162D4E",
                    edgecolor="#7A9CC4", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0A1628")
    plt.close()
    print(f"  Guardado: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    OUTPUT_DIR.mkdir(exist_ok=True)
    device   = CFG["device"]
    use_fp16 = CFG["use_fp16"] and device == "cuda"

    # Cargar modelo
    print(f"\nCargando checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = AnomalyDetector3Class(CFG).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Época: {ckpt['epoch']}  |  AUC guardado: {ckpt['auc']:.4f}")

    # Dataset test
    test_ds = UCFCrime3ClassDataset(
        CFG["data_root"], "test_split.txt",
        CFG["num_segments"], CFG["img_size"], augment=False,
    )
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"],
                             shuffle=False, num_workers=0, pin_memory=True)

    # Inferencia
    print("\nEjecutando inferencia en test set...")
    y_true, y_pred, y_probs = run_inference(model, test_loader, device, use_fp16)

    # ── Métricas ──────────────────────────────────────────────
    acc = np.mean(y_true == y_pred)
    print(f"\n{'='*50}")
    print(f"  RESULTADOS FINALES")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")

    try:
        macro_auc = roc_auc_score(y_true, y_probs,
                                  multi_class="ovr", average="macro")
        print(f"  AUC macro: {macro_auc:.4f}")
        for i, name in enumerate(CLASS_NAMES):
            y_bin = (y_true == i).astype(int)
            a     = roc_auc_score(y_bin, y_probs[:, i])
            print(f"    AUC {name:15s}: {a:.4f}")
    except ValueError as e:
        print(f"  AUC: {e}")
        macro_auc = 0.0

    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES,
                                digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Matriz de confusión (filas=real, cols=pred):")
    print(f"  {'':15s} " + "  ".join(f"{n:12s}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:15s} " + "  ".join(f"{cm[i,j]:12d}" for j in range(len(CLASS_NAMES))))

    # ── Gráficos ──────────────────────────────────────────────
    print(f"\nGenerando gráficos en {OUTPUT_DIR}/...")
    plot_confusion_matrix(y_true, y_pred,
                          OUTPUT_DIR / "confusion_matrix.png")
    plot_roc_curves(y_true, y_probs,
                    OUTPUT_DIR / "roc_curves.png")

    print(f"\n✓ Evaluación completa.")
    print(f"  Archivos en: {OUTPUT_DIR}/")
    print(f"    - confusion_matrix.png")
    print(f"    - roc_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="checkpoints_3class/best_model.pth")
    args = parser.parse_args()
    main(args)