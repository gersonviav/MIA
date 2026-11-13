"""
Baseline clásico B0: regresión logística sobre las mismas features que M0, con predicciones por video
en el formato de compare.py (para poder hacerle la prueba pareada contra M0).

Úsalo solo si `vad baseline` no guardó predicciones por video. Ejecutar desde la raíz de MIA:

  python baselines_mil/b0_logistic.py --manifest data/splits/manifest_3class_v2.csv ^
      --feat_dir data/processed/features/convnext_s_t_diff_320 --out_dir artifacts/runs_baselines/logistica

Embedding por video = promedio temporal de cada stream (rgb y mot), concatenado, igual que el
análisis de separabilidad del notebook 02. Estandarización ajustada solo con train; C elegido en
validación por F1-macro; pesos de clase balanceados. El test se evalúa una sola vez.

Localización (opcional): con --annotations y --nframes_meta el mismo clasificador se aplica a cada
snippet (rgb_t, mot_t) y 1 − p(Normal) por snippet da la curva de anomalía. Con eso se calcula el
AUC/AP a nivel de frame igual que para M0 y los baselines MIL:

  python baselines_mil/b0_logistic.py --manifest data/splits/manifest_3class_v2.csv ^
      --feat_dir data/processed/features/convnext_s_t_diff_320 --out_dir artifacts/runs_baselines/logistica ^
      --annotations data/annotations/Temporal_Anomaly_Annotation.txt ^
      --nframes_meta data/processed/i3d_320x240_10crop/meta.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_manifest  # noqa: E402
from metrics import (frame_level_auc, load_temporal_annotations, video_metrics,  # noqa: E402
                     write_predictions, write_segment_scores)


def embed(path, keys):
    z = np.load(path, allow_pickle=True)
    parts = [np.asarray(z[k], dtype=np.float32).reshape(len(z[k]), -1).mean(0) for k in keys if k in z.files]
    if not parts:
        raise KeyError(f"{path.name}: no tiene {keys} (arrays: {z.files})")
    return np.concatenate(parts)


def snippets(path, keys):
    """Matriz [T, D] con las features de cada snippet (streams concatenados)."""
    z = np.load(path, allow_pickle=True)
    parts = [np.asarray(z[k], dtype=np.float32).reshape(len(z[k]), -1) for k in keys if k in z.files]
    T = min(len(p) for p in parts)
    return np.concatenate([p[:T] for p in parts], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out_dir", default="artifacts/runs_baselines/logistica")
    ap.add_argument("--keys", nargs="+", default=["rgb", "mot"])
    ap.add_argument("--annotations", default=None, help="Temporal_Anomaly_Annotation.txt")
    ap.add_argument("--nframes_meta", default=None, help="meta.json con n_frames por video (el de I3D)")
    ap.add_argument("--Cs", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    args = ap.parse_args()

    df = load_manifest(args.manifest)
    fd = Path(args.feat_dir)
    X = np.stack([embed(fd / f"{v}.npz", args.keys) for v in df.video_id])
    parts = {s: (df.split == s).values for s in ("train", "val", "test")}
    sc = StandardScaler().fit(X[parts["train"]])
    Xs = sc.transform(X)
    y = df["y"].values

    best = None
    for C in args.Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        clf.fit(Xs[parts["train"]], y[parts["train"]])
        pv = clf.predict_proba(Xs[parts["val"]])
        f1 = video_metrics(y[parts["val"]], pv.argmax(1), 1 - pv[:, 0])["f1_macro"]
        print(f"C={C:<6} val F1-macro {f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    f1v, C, clf = best
    print(f"Elegido C={C} (val F1-macro {f1v:.4f})")

    out = Path(args.out_dir) / "seed1"
    out.mkdir(parents=True, exist_ok=True)
    res = {"C": C}
    for s in ("val", "test"):
        p = clf.predict_proba(Xs[parts[s]])
        write_predictions(out / f"preds_{s}.csv", df.video_id[parts[s]], s, y[parts[s]], p, 1 - p[:, 0])
        res[s] = video_metrics(y[parts[s]], p.argmax(1), 1 - p[:, 0])
        print(f"[{s}] F1-macro {res[s]['f1_macro']:.4f} | F1-anomalía {res[s]['f1_anomaly']:.4f} | "
              f"AUC-video {res[s].get('auc_video', float('nan')):.4f}")
    if args.annotations and args.nframes_meta:
        meta = json.loads(Path(args.nframes_meta).read_text())
        ann = load_temporal_annotations(args.annotations)
        for s in ("val", "test"):
            sub = df[parts[s]]
            seg = {v: 1 - clf.predict_proba(sc.transform(snippets(fd / f"{v}.npz", args.keys)))[:, 0]
                   for v in sub.video_id}
            nfr = {v: meta[v]["n_frames"] for v in sub.video_id}
            # los snippets de M0 cubren el video de forma uniforme (1 de cada 10 frames)
            write_segment_scores(out / f"segscores_{s}.npz", seg, nfr, "uniform")
            if s == "test":
                fr = frame_level_auc(seg, nfr, "uniform", ann, dict(zip(sub.video_id, sub.is_anom)))
                res["test"] |= fr
                print(f"[test] AUC-frame {fr['auc_frame']:.4f} | AP-frame {fr['ap_frame']:.4f} "
                      f"({fr['n_videos_frame']} videos)")
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
