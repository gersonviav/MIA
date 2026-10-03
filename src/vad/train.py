"""
Entrenamiento de la cabeza MIL (etapa 2) sobre features pre-extraídas
=====================================================================
Pérdida = CE video                                          (v5)
        + λ_ent    · entropía de atención (solo anómalos)   (v5)
        + λ_seg    · CE del segmento más atendido           (v5)
        + λ_smooth · suavidad de la atención                (v5)
        + λ_rank   · ranking MIL: top-k s(anómalo) > top-k s(normal) + margen  (Sultani 2018, top-k tipo RTFM)
        + λ_normal · BCE(s, 0) en TODOS los snippets de videos normales
        + λ_topk   · BCE(top-k s, 1) en videos anómalos
        + λ_spars  · media de s en anómalos (pocos snippets anómalos)          (Sultani)
        + λ_ssmooth· suavidad temporal de s                                    (Sultani)
Diagnóstico en val (sin anotación temporal): "medio anómalos" debe quedar BAJO aunque sep(max) sea alto;
si el score medio de los anómalos ≈ 1, el modelo marca todo el video y no localiza.
Selección del mejor modelo: macro-F1 en VALIDACIÓN (sale de Train). El test no se toca.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from vad.config import class_names, seed_everything, snapshot
from vad.data.common import PROCESSED_VERSION_NAME, read_json
from vad.data.feature_dataset import BalancedBatchSampler, SegmentDataset
from vad.inference import build_model, get_device

log = logging.getLogger("vad.train")


def topk_for(T: int, tc: dict) -> int:
    ratio = tc.get("rank_topk_ratio", 0) or 0
    k = int(round(ratio * T)) if ratio > 0 else tc.get("rank_topk", 1)
    return max(1, min(k, T))


def mil_losses(out: dict, s: torch.Tensor, y: torch.Tensor, tc: dict) -> dict:
    """out: salida del Transformer (vista 32); s: scores [B,Ts] (vista fina del scorer)."""
    w = out["w"].squeeze(-1).float().clamp_min(1e-8)          # [B,T]
    B, T = w.shape
    anom = y != 0
    zero = torch.zeros((), device=y.device)

    ent = -(w * w.log()).sum(1) / np.log(T)
    l_ent = ent[anom].mean() if anom.any() else zero
    top = w.argmax(1)
    l_seg = F.cross_entropy(out["seg_logits"][torch.arange(B, device=y.device), top].float(), y)
    l_smooth = ((w[:, 1:] - w[:, :-1]) ** 2).sum(1).mean()

    k = topk_for(s.shape[1], tc)
    sc = s.float().clamp(1e-6, 1 - 1e-6)
    if anom.any() and (~anom).any():
        top_a = sc[anom].topk(k, dim=1).values.mean(1)          # los k snippets más anómalos
        top_n = sc[~anom].topk(k, dim=1).values.mean(1)
        l_rank = F.relu(tc["rank_margin"] - top_a[:, None] + top_n[None, :]).mean()
    else:
        l_rank = zero
    # Video normal ⇒ TODOS sus snippets son normales (única supervisión por snippet que es segura)
    l_normal = F.binary_cross_entropy(sc[~anom], torch.zeros_like(sc[~anom])) if (~anom).any() else zero
    # Video anómalo ⇒ al menos sus k snippets más altos son anómalos
    l_topk = (F.binary_cross_entropy(sc[anom].topk(k, dim=1).values, torch.ones_like(sc[anom][:, :k]))
              if anom.any() else zero)
    l_sparsity = sc[anom].mean() if anom.any() else zero
    l_ssmooth = ((sc[:, 1:] - sc[:, :-1]) ** 2).mean()

    return {"ent": tc["lambda_ent"] * l_ent, "seg": tc["lambda_seg"] * l_seg,
            "smooth": tc["lambda_smooth"] * l_smooth, "rank": tc["lambda_rank"] * l_rank,
            "normal": tc.get("lambda_normal", 0.0) * l_normal,
            "topk": tc.get("lambda_topk", 0.0) * l_topk,
            "sparsity": tc["lambda_sparsity"] * l_sparsity,
            "score_smooth": tc["lambda_score_smooth"] * l_ssmooth}


def _scores(model, out, rgb_s, mot_s, device) -> torch.Tensor:
    """Scorer local → vista fina; score desde el Transformer (ablación) → vista de 32."""
    if getattr(model, "score_head", "transformer") == "local":
        return model.snippet_scores(rgb_s.to(device), mot_s.to(device))
    return out["scores"]


@torch.no_grad()
def evaluate_val(model, loader, criterion, device) -> dict:
    model.eval()
    loss, ys, ps, max_s, mean_s = 0.0, [], [], [], []
    for rgb, mot, y, _, rgb_s, mot_s in loader:
        rgb, mot, y = rgb.to(device), mot.to(device), y.to(device)
        out = model(rgb, mot)
        s = _scores(model, out, rgb_s, mot_s, device)
        loss += criterion(out["video_logits"], y).item()
        ys += y.tolist()
        ps += out["video_logits"].argmax(1).tolist()
        max_s += s.max(1).values.tolist()
        mean_s += s.mean(1).tolist()
    ys, ps, max_s, mean_s = np.array(ys), np.array(ps), np.array(max_s), np.array(mean_s)
    sep = float(max_s[ys != 0].mean() - max_s[ys == 0].mean()) if (ys != 0).any() and (ys == 0).any() else float("nan")
    return {"loss": loss / max(len(loader), 1), "acc": float((ys == ps).mean()),
            "f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
            "pred_dist": np.bincount(ps, minlength=3).tolist(), "score_separation": sep,
            "mean_score_anom": float(mean_s[ys != 0].mean()) if (ys != 0).any() else float("nan"),
            "mean_score_norm": float(mean_s[ys == 0].mean()) if (ys == 0).any() else float("nan")}


def run_train(cfg: dict) -> dict:
    tc, paths = cfg["train"], cfg["paths"]
    seed_everything(tc["seed"])
    device = get_device()
    names = class_names(cfg)
    ckpt_dir = Path(paths["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pv = Path(paths["manifests_dir"]) / PROCESSED_VERSION_NAME
    data_version = read_json(pv) if pv.exists() else {}

    log.info("=" * 70)
    log.info("ENTRENAMIENTO MIL | %s | snippet=%d frames | %d segmentos (Transformer) / %d (scorer) | "
             "top-k %s | seed %d", device, cfg["snippet"]["frames_per_snippet"], cfg["mil"]["num_segments"],
             cfg["mil"].get("scorer_segments", cfg["mil"]["num_segments"]),
             f"{tc.get('rank_topk_ratio')}·T" if tc.get("rank_topk_ratio") else tc.get("rank_topk"), tc["seed"])
    log.info("λ: %s", {k: v for k, v in tc.items() if k.startswith("lambda") or k.startswith("rank")})
    log.info("=" * 70)

    log.info("Cargando features...")
    train_ds, val_ds = SegmentDataset(cfg, "train"), SegmentDataset(cfg, "val")
    train_loader = DataLoader(train_ds, batch_sampler=BalancedBatchSampler(train_ds.labels, tc["batch_size"], tc["seed"]),
                              num_workers=tc["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False, num_workers=tc["num_workers"])

    counts = np.bincount(train_ds.labels, minlength=len(names)).astype(float)
    class_w = torch.tensor(counts.sum() / (len(names) * np.maximum(counts, 1)), dtype=torch.float32)
    log.info("Pesos CE por clase: %s", dict(zip(names, class_w.round(decimals=3).tolist())))

    model = build_model(cfg, device)
    log.info("Parámetros entrenables: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device))

    best_f1, best_epoch, no_improve, history = -1.0, 0, 0, []
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        sums = {}
        for rgb, mot, y, _, rgb_s, mot_s in train_loader:
            rgb, mot, y = rgb.to(device), mot.to(device), y.to(device)
            out = model(rgb, mot)
            s = _scores(model, out, rgb_s, mot_s, device)
            parts = {"ce": criterion(out["video_logits"].float(), y), **mil_losses(out, s, y, tc)}
            loss = sum(parts.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v.item()
        scheduler.step()
        tr = {k: v / len(train_loader) for k, v in sums.items()}
        val = evaluate_val(model, val_loader, criterion, device)
        log.info("Ép %02d | train %.4f %s", epoch, sum(tr.values()), {k: round(v, 3) for k, v in tr.items()})
        log.info("       val F1 %.4f acc %.3f | preds %s | score: sep(max) %.3f | medio anómalos %.3f normales %.3f",
                 val["f1"], val["acc"], val["pred_dist"], val["score_separation"],
                 val["mean_score_anom"], val["mean_score_norm"])
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in val.items() if k != "pred_dist"}})

        if val["f1"] > best_f1:
            best_f1, best_epoch, no_improve = val["f1"], epoch, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_f1": val["f1"],
                        "val_score_separation": val["score_separation"], "cfg": snapshot(cfg),
                        "data_version": data_version}, ckpt_dir / "best_model.pth")
            log.info("        → mejor checkpoint (val F1 %.4f)", best_f1)
        else:
            no_improve += 1
            if no_improve >= tc["patience"]:
                log.info("[EARLY STOPPING] mejor época %d (F1 %.4f)", best_epoch, best_f1)
                break

    with open(ckpt_dir / "history.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=history[0].keys())
        wr.writeheader()
        wr.writerows(history)
    log.info("Mejor val macro-F1 %.4f (época %d) | %s", best_f1, best_epoch, ckpt_dir / "best_model.pth")
    log.info("Siguiente paso: vad evaluate")
    return {"best_val_f1": best_f1, "best_epoch": best_epoch}
