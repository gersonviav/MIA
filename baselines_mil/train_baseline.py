"""
Entrena y evalúa un baseline MIL adaptado a 3 clases (Normal / Aggression / Theft) sobre el MISMO
manifiesto (split train/val/test) que el modelo propuesto.

  python train_baseline.py --model rtfm --manifest data/manifest_3class.csv \
      --feat_dir features/i3d_320x240_10crop --out_dir runs/rtfm --seed 0 \
      --annotations Temporal_Anomaly_Annotation.txt

Selección de modelo y parada temprana: F1-macro en VALIDACIÓN (nunca se mira el test).
Al final: carga el mejor checkpoint, evalúa val y test, guarda predicciones y métricas.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from data import BalancedBagSampler, FeatureStore, crop_view, load_manifest
from metrics import (frame_level_auc, load_temporal_annotations, video_metrics, write_predictions,
                     write_segment_scores)
from models.rtfm import RTFM
from models.sultani import SultaniMIL
from models.urdmu import URDMU

# Hiperparámetros por defecto = los de cada repositorio oficial
DEFAULTS = {
    "sultani": dict(cls=SultaniMIL, opt="adagrad", lr=1e-2, wd=1e-3, batch_size=30, iters=20000,
                    n_seg=32, train_crop="mean", test_crop="mean"),
    "rtfm": dict(cls=RTFM, opt="adam", lr=1e-3, wd=5e-3, batch_size=32, iters=15000,
                 n_seg=32, train_crop="all", test_crop="all"),
    "urdmu": dict(cls=URDMU, opt="adam", lr=1e-4, wd=5e-5, batch_size=64, iters=3000,
                  n_seg=200, train_crop="random", test_crop="all"),
}


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model, df, store, test_crop, device, rng):
    model.eval()
    probs, ascore, seg, nfr = [], [], {}, {}
    for vid in df["video_id"]:
        if model.test_segments:
            arr = store.segments(vid, model.test_segments)
        else:
            arr = np.asarray(store.raw(vid))
        x = torch.from_numpy(crop_view(arr, test_crop, rng).astype(np.float32))[None].to(device)
        out = model.infer(x)
        probs.append(torch.softmax(out["logits"].float(), -1).cpu().numpy())
        ascore.append(float(out["video_score"]))
        seg[vid] = out["snippet_scores"].float().cpu().numpy()
        nfr[vid] = store.n_frames(vid)
    return np.stack(probs), np.array(ascore), seg, nfr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(DEFAULTS))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--annotations", default=None, help="Temporal_Anomaly_Annotation.txt (AUC por frame)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lam_cls", type=float, default=1.0, help="peso de la CE de 3 clases")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20, help="evaluaciones sin mejora en val")
    ap.add_argument("--d_in", type=int, default=1024)
    ap.add_argument("--col_video", default="video_id")
    ap.add_argument("--col_label", default="label")
    ap.add_argument("--col_split", default="split")
    args = ap.parse_args()

    cfg = dict(DEFAULTS[args.model])
    for k in ("iters", "lr", "batch_size"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir) / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    df = load_manifest(args.manifest, args.col_video, args.col_label, args.col_split)
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    print(f"train={len(tr)} val={len(va)} test={len(te)} | por clase train: {tr.label.value_counts().to_dict()}")
    store = FeatureStore(args.feat_dir)

    # Pesos de clase inversos a la frecuencia en TRAIN (mismo criterio que el modelo propuesto)
    cnt = np.bincount(tr["y"], minlength=3).astype(float)
    w = torch.tensor(cnt.sum() / (3 * cnt), dtype=torch.float32, device=device)

    model = cfg["cls"](d_in=args.d_in).to(device)
    if cfg["opt"] == "adagrad":
        opt = torch.optim.Adagrad(model.parameters(), lr=cfg["lr"], eps=1e-8, weight_decay=cfg["wd"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sampler = BalancedBagSampler(tr, store, cfg["n_seg"], cfg["train_crop"], cfg["batch_size"], args.seed)
    rng = np.random.default_rng(args.seed)

    best, best_it, bad, log = -1.0, 0, 0, []
    t0 = time.time()
    for it in range(1, cfg["iters"] + 1):
        model.train()
        xn, yn, xa, ya = sampler.next()
        tt = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
        loss, parts = model.train_loss(tt(xn), tt(yn, torch.long), tt(xa), tt(ya, torch.long), w, args.lam_cls)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % args.eval_every == 0 or it == cfg["iters"]:
            p, s, _, _ = predict(model, va, store, cfg["test_crop"], device, rng)
            m = video_metrics(va["y"], p.argmax(1), s)
            log.append({"it": it, "loss": loss.item(), **parts, "val_f1_macro": m["f1_macro"],
                        "val_f1_anomaly": m["f1_anomaly"], "val_auc": m.get("auc_video")})
            print(f"it {it:5d} loss {loss.item():.4f} (mil {parts['mil']:.4f} cls {parts['cls']:.4f}) "
                  f"| val F1m {m['f1_macro']:.4f} F1anom {m['f1_anomaly']:.4f} AUC {m.get('auc_video', float('nan')):.4f} "
                  f"| {time.time() - t0:.0f}s")
            score = m["f1_macro"] + 1e-3 * m.get("auc_video", 0)  # desempate por AUC
            if score > best:
                best, best_it, bad = score, it, 0
                torch.save(model.state_dict(), out / "best.pt")
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"Parada temprana en it {it} (mejor it {best_it})")
                    break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    ann = load_temporal_annotations(args.annotations) if args.annotations else None
    results = {"model": args.model, "seed": args.seed, "best_iter": best_it, "config":
               {k: v for k, v in cfg.items() if k != "cls"} | {"lam_cls": args.lam_cls}}
    for name, part in (("val", va), ("test", te)):
        p, s, seg, nfr = predict(model, part, store, cfg["test_crop"], device, rng)
        write_predictions(out / f"preds_{name}.csv", part["video_id"], name, part["y"], p, s)
        write_segment_scores(out / f"segscores_{name}.npz", seg, nfr, model.score_kind)
        m = video_metrics(part["y"], p.argmax(1), s)
        if ann is not None and name == "test":
            m |= frame_level_auc(seg, nfr, model.score_kind, ann, dict(zip(part.video_id, part.is_anom)))
        results[name] = m
        print(f"[{name}] F1-macro {m['f1_macro']:.4f} | F1-anomalía {m['f1_anomaly']:.4f} | "
              f"AUC-video {m.get('auc_video', float('nan')):.4f}" +
              (f" | AUC-frame {m['auc_frame']:.4f}" if "auc_frame" in m else ""))
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    (out / "train_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
