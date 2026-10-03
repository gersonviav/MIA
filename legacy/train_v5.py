"""
train_v5.py — Entrenamiento con localización temporal supervisada débilmente
=============================================================================
v5 — Añade a v4 los mecanismos que fuerzan atención puntiaguda + control de semilla.

PÉRDIDA COMPUESTA:
  L = CE(video_logits, y)
    + λ_ent  * H_norm(w)          [solo videos anómalos]
    + λ_seg  * CE(seg_top, y)     [el segmento top debe votar por la clase correcta]
    + λ_smooth * Σ (w_t - w_{t+1})²   [suavidad temporal, estilo Sultani]

  - H_norm(w) = -Σ w log w / log(T)  ∈ [0,1]. 1 = uniforme (malo), 0 = un solo pico.
    Se aplica SOLO a videos anómalos: en un video normal no hay nada que localizar,
    forzar un pico ahí sería incorrecto.
  - CE(seg_top, y) obliga a que el segmento más atendido sea, por sí solo,
    clasificable como la clase del video. Sin esto los seg_logits votan casi
    idéntico en todos los segmentos (observado en v4: ~0.99 en todos).
  - La suavidad evita picos aislados de un solo segmento por ruido.

Ejemplos:
  python train_v5.py --motion diff --seed 42
  python train_v5.py --motion diff --lambda_ent 0.3 --attn_temp 0.5
  python train_v5.py --motion diff --topk 3 --lambda_seg 0.5
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import UCFCrimeDatasetMIL
from models_v5 import BaselineModels

CLASS_NAMES = ["NormalVideos", "Fighting", "Vandalism"]

CFG = {
    "data_root": "ucf_crime_3class",
    "checkpoint_dir": "checkpoints_v5",
    "epochs": 40, "batch_size": 4, "lr": 3e-4, "weight_decay": 1e-2,
    "motion_type": "diff", "chunk_size": 16, "num_layers": 2, "dropout": 0.3,
    "patience": 8, "freeze": True, "num_workers": 0, "seed": 42,
    "attn_temp": 0.5, "topk": 0,
    "lambda_ent": 0.3, "lambda_seg": 0.5, "lambda_smooth": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset, num_classes=3):
    labels = [s["label"] for s in dataset.samples]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32), \
           np.bincount(labels, minlength=num_classes)


def attention_losses(w, seg_logits, labels, cfg):
    """
    w:          (B, T, 1) pesos de atención normalizados
    seg_logits: (B, T, C) logits por segmento
    Devuelve (loss_entropia, loss_segmento, loss_suavidad, entropia_media_anomalos)
    """
    device = w.device
    B, T, _ = w.shape
    wf = w.squeeze(-1).float().clamp_min(1e-8)          # (B, T)

    # --- Entropía normalizada, solo en videos anómalos ---
    ent = -(wf * wf.log()).sum(dim=1) / np.log(T)       # (B,) en [0,1]
    anom_mask = (labels != 0)
    if anom_mask.any():
        loss_ent = ent[anom_mask].mean()
        ent_report = loss_ent.detach()
    else:
        loss_ent = torch.zeros((), device=device)
        ent_report = torch.zeros((), device=device)

    # --- El segmento más atendido debe votar por la clase del video ---
    top_idx = wf.argmax(dim=1)                                        # (B,)
    seg_top = seg_logits[torch.arange(B, device=device), top_idx]     # (B, C)
    loss_seg = F.cross_entropy(seg_top.float(), labels)

    # --- Suavidad temporal (evita picos aislados por ruido) ---
    loss_smooth = ((wf[:, 1:] - wf[:, :-1]) ** 2).sum(dim=1).mean()

    return (cfg["lambda_ent"] * loss_ent,
            cfg["lambda_seg"] * loss_seg,
            cfg["lambda_smooth"] * loss_smooth,
            ent_report)


@torch.no_grad()
def evaluate_split(model, loader, criterion, device, amp_device):
    model.eval()
    total_loss, y_true, y_pred, ents, peaks = 0.0, [], [], [], []
    for rgb, motion, labels, _, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        motion = motion.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device_type=amp_device):
            logits, w, _, _ = model(rgb, motion)
            total_loss += criterion(logits, labels).item()
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        if w is not None:
            wf = w.squeeze(-1).float().clamp_min(1e-8)
            T = wf.shape[1]
            e = (-(wf * wf.log()).sum(dim=1) / np.log(T)).cpu().numpy()
            anom = (labels != 0).cpu().numpy()
            ents.extend(e[anom].tolist())
            peaks.extend(wf.max(dim=1).values.cpu().numpy()[anom].tolist())
    total_loss /= max(len(loader), 1)
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ent_mean = float(np.mean(ents)) if ents else float("nan")
    peak_mean = float(np.mean(peaks)) if peaks else float("nan")
    return total_loss, acc, macro_f1, y_pred, ent_mean, peak_mean


def train():
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    set_seed(CFG["seed"])
    device = CFG["device"]
    amp_device = "cuda" if device == "cuda" else "cpu"
    torch.backends.cudnn.benchmark = True

    print("\n" + "=" * 72)
    print(f" ENTRENO v5 | motion={CFG['motion_type'].upper()} | seed={CFG['seed']} | {device}")
    print(f" Localización: τ={CFG['attn_temp']} | topk={CFG['topk'] or 'off'} | "
          f"λ_ent={CFG['lambda_ent']} λ_seg={CFG['lambda_seg']} λ_smooth={CFG['lambda_smooth']}")
    print("=" * 72 + "\n")

    train_ds = UCFCrimeDatasetMIL(CFG["data_root"], "train_split.txt",
                                  augment=True, motion_type=CFG["motion_type"])
    val_ds = UCFCrimeDatasetMIL(CFG["data_root"], "val_split.txt",
                                augment=False, motion_type=CFG["motion_type"])

    lk = {"num_workers": CFG["num_workers"], "pin_memory": (device == "cuda")}
    if CFG["num_workers"] > 0:
        lk["persistent_workers"] = True
    g = torch.Generator(); g.manual_seed(CFG["seed"])
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, generator=g, **lk)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, **lk)

    class_w, counts = compute_class_weights(train_ds)
    print("Distribución en TRAIN:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:15s}: {counts[i]:4d} | peso = {class_w[i]:.3f}")

    model = BaselineModels(
        model_type="full_model", num_classes=3, chunk_size=CFG["chunk_size"],
        freeze_backbone=CFG["freeze"], num_layers=CFG["num_layers"],
        dropout=CFG["dropout"], attn_temp=CFG["attn_temp"], topk=CFG["topk"]
    ).to(device)

    total_p, trainable_p = model.trainable_summary()
    print(f"\nParámetros: {total_p/1e6:.1f}M totales | {trainable_p/1e6:.2f}M entrenables\n")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device))
    scaler = GradScaler()

    best_f1, best_epoch, no_improve, history = -1.0, 0, 0, []
    ckpt = f"{CFG['checkpoint_dir']}/best_full_model_{CFG['motion_type']}_s{CFG['seed']}_v5.pth"

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        tl = tl_ce = tl_ent = tl_seg = 0.0
        ent_tr = []
        for rgb, motion, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{CFG['epochs']}", leave=False):
            rgb = rgb.to(device, non_blocking=True)
            motion = motion.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=amp_device):
                logits, w, seg_logits, _ = model(rgb, motion)
                loss_ce = criterion(logits, labels)

            l_ent, l_seg, l_sm, ent_val = attention_losses(w, seg_logits, labels, CFG)
            loss = loss_ce.float() + l_ent + l_seg + l_sm

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            tl += loss.item(); tl_ce += loss_ce.item()
            tl_ent += l_ent.item(); tl_seg += l_seg.item()
            ent_tr.append(ent_val.item())

        scheduler.step()
        n = len(train_loader)
        tl, tl_ce, tl_ent, tl_seg = tl/n, tl_ce/n, tl_ent/n, tl_seg/n

        val_loss, val_acc, val_f1, preds, val_ent, val_peak = evaluate_split(
            model, val_loader, criterion, device, amp_device)
        dist = np.bincount(preds, minlength=3)

        print(f"Ep {epoch:02d} | train {tl:.4f} (ce {tl_ce:.3f} ent {tl_ent:.3f} seg {tl_seg:.3f}) "
              f"|| val loss {val_loss:.4f} F1 {val_f1:.4f}")
        print(f"         atención: H_norm {val_ent:.3f} | pico medio {val_peak:.4f} "
              f"(uniforme = {1/16:.4f}) | preds N:{dist[0]} F:{dist[1]} V:{dist[2]}")

        history.append({"epoch": epoch, "train_loss": tl, "train_ce": tl_ce,
                        "train_ent": tl_ent, "train_seg": tl_seg,
                        "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                        "val_attn_entropy": val_ent, "val_attn_peak": val_peak})

        if val_f1 > best_f1:
            best_f1, best_epoch, no_improve = val_f1, epoch, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_f1": val_f1, "val_attn_peak": val_peak,
                        "cfg": CFG, "history": history,
                        "arch": {"model_type": "full_model", "num_classes": 3,
                                 "num_layers": CFG["num_layers"], "dropout": CFG["dropout"],
                                 "freeze_backbone": CFG["freeze"], "chunk_size": CFG["chunk_size"],
                                 "attn_temp": CFG["attn_temp"], "topk": CFG["topk"]}}, ckpt)
            print(f"         → mejor checkpoint (F1 {val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= CFG["patience"]:
                print(f"\n[EARLY STOPPING] mejor época {best_epoch} (F1 {best_f1:.4f})")
                break

    hist_path = f"{CFG['checkpoint_dir']}/history_{CFG['motion_type']}_s{CFG['seed']}_v5.csv"
    with open(hist_path, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=history[0].keys()); wr.writeheader(); wr.writerows(history)

    print("\n" + "=" * 72)
    print(f" Mejor macro-F1 val: {best_f1:.4f} (época {best_epoch}) | semilla {CFG['seed']}")
    print(f" Checkpoint: {ckpt}")
    print("=" * 72 + "\n")
    return best_f1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Entrenamiento v5 con localización temporal.")
    p.add_argument("--motion", default="diff", choices=["diff", "flow"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn_temp", type=float, default=0.5,
                   help="Temperatura del softmax de atención. <1 agudiza. 1.0 = como v4")
    p.add_argument("--topk", type=int, default=0,
                   help="Top-k pooling estilo Sultani. 0 = desactivado")
    p.add_argument("--lambda_ent", type=float, default=0.3, help="Peso de la entropía de atención")
    p.add_argument("--lambda_seg", type=float, default=0.5, help="Peso del MIL por segmento top")
    p.add_argument("--lambda_smooth", type=float, default=0.1, help="Peso de la suavidad temporal")
    p.add_argument("--no_freeze", action="store_true")
    a = p.parse_args()

    CFG.update({k: getattr(a, k) for k in
                ["epochs", "batch_size", "lr", "weight_decay", "dropout", "num_layers",
                 "chunk_size", "num_workers", "seed", "attn_temp", "topk",
                 "lambda_ent", "lambda_seg", "lambda_smooth"]})
    CFG["motion_type"] = a.motion
    CFG["patience"] = a.patience if a.patience > 0 else 10 ** 9
    CFG["freeze"] = not a.no_freeze
    train()
