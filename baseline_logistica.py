"""
Baseline mínimo: regresión logística sobre las MISMAS features que usa el modelo
==============================================================================
Independiente de `vad` (sirve con la versión 0.6.0 del repo). Usa la config para saber
qué splits y qué features leer.

Tres referencias, todas evaluadas en el TEST oficial:
  1. Clase mayoritaria (de train)             → piso trivial
  2. Logística por video  (3 clases)           → accuracy, macro-F1, F1 por clase, AUC binario
     embedding del video = [media temporal ; máximo temporal] de las features rgb+mot
  3. Logística por frame  (normal vs anómalo)  → AUC / AP a nivel de frame con las anotaciones
     temporales. Etiqueta débil: todos los frames de un video anómalo = 1 (sin MIL).
La regularización C se elige con VALIDACIÓN (nunca con test).

Uso (desde la raíz del repo):
    python baseline_logistica.py --config configs/hr320_cnn_v2.yaml --run cnn_v2
Salida:
    artifacts/seeds/<run>/baseline_logistica/metrics.json   (lo lee notebooks/03_experimentos.ipynb)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CS = [0.01, 0.1, 1.0, 10.0]
MAX_FRAMES_PER_VIDEO = 64      # frames por video para entrenar la logística por frame
SEED = 42


def load_npz(path: Path) -> np.ndarray | None:
    """Concatena todos los arrays float 2D del .npz (rgb, mot…) → (T, D)."""
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    arrs = [z[k].astype(np.float32).reshape(z[k].shape[0], -1)
            for k in sorted(z.files) if z[k].dtype.kind == "f" and z[k].ndim >= 2]
    if not arrs:
        return None
    t = min(a.shape[0] for a in arrs)
    return np.concatenate([a[:t] for a in arrs], axis=1)


def load_annotations(path: Path) -> dict[str, list[tuple[int, int]]]:
    ann = {}
    if not path.exists():
        return ann
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 6:
            continue
        vid = Path(p[0]).stem
        nums = [int(x) for x in p[2:6]]
        ann[vid] = [(a, b) for a, b in ((nums[0], nums[1]), (nums[2], nums[3])) if a >= 0 and b >= 0]
    return ann


def frame_numbers(row: dict, t: int, every_n: int) -> np.ndarray:
    fi = row.get("frame_indices")
    if isinstance(fi, list) and len(fi) == t:
        return np.asarray(fi)
    return np.arange(t) * every_n


def fit_best(X_tr, y_tr, X_va, y_va, score_fn):
    best = (-np.inf, None, None)
    for C in CS:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf.fit(X_tr, y_tr)
        s = score_fn(clf, X_va, y_va)
        print(f"    C={C:<5} val = {s:.4f}")
        if s > best[0]:
            best = (s, C, clf)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, help="nombre del run, p. ej. cnn_v2")
    args = ap.parse_args()
    t0 = time.time()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    proc = Path(cfg["paths"]["processed_dir"])
    feat_dir = Path(cfg["features"]["dir"])
    every_n = int(cfg.get("videos", {}).get("every_n", 10))
    c2i = cfg["preprocess"]["class_to_idx"]
    classes = sorted(c2i, key=c2i.get)
    normal = classes[0]
    ann = load_annotations(Path(cfg["paths"].get("temporal_annotations", "")))

    print(f"Config: {args.config}\nFeatures: {feat_dir}\nSplits: {proc}")
    splits = {s: [json.loads(l) for l in (proc / "splits" / f"{s}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
              for s in ("train", "val", "test")}

    # ── Cargar features ──────────────────────────────────────────────────────
    data, faltan = {}, []
    for s, rows in splits.items():
        for r in rows:
            F = load_npz(feat_dir / f"{r['video_id']}.npz")
            if F is None:
                faltan.append(r["video_id"]); continue
            data[r["video_id"]] = (s, r, F)
    print(f"Videos con features: {len(data)} | sin features: {len(faltan)}")
    if faltan:
        print("  [AVISO] faltan .npz (corre `vad --config ... extract`):", faltan[:10], "…" if len(faltan) > 10 else "")

    def subset(s):
        return [(r, F) for (sp, r, F) in data.values() if sp == s]

    # ── 1. Clase mayoritaria ─────────────────────────────────────────────────
    y_train = [r["class_name"] for r, _ in subset("train")]
    maj = pd.Series(y_train).mode()[0]
    y_test = [r["class_name"] for r, _ in subset("test")]
    majority = {"class": maj, "video_level": {
        "accuracy": accuracy_score(y_test, [maj] * len(y_test)),
        "macro_f1": f1_score(y_test, [maj] * len(y_test), labels=classes, average="macro", zero_division=0)}}
    print(f"\n[1] Mayoritaria ({maj}): acc {majority['video_level']['accuracy']:.4f} | macro-F1 {majority['video_level']['macro_f1']:.4f}")

    # ── 2. Logística por video ───────────────────────────────────────────────
    def video_xy(s):
        X = np.stack([np.concatenate([F.mean(0), F.max(0)]) for _, F in subset(s)])
        y = np.array([c2i[r["class_name"]] for r, _ in subset(s)])
        return X, y

    Xtr, ytr = video_xy("train"); Xva, yva = video_xy("val"); Xte, yte = video_xy("test")
    print("\n[2] Logística por video (C elegido con macro-F1 en val):")
    _, C_v, clf_v = fit_best(Xtr, ytr, Xva, yva,
                             lambda c, X, y: f1_score(y, c.predict(X), average="macro", zero_division=0))
    pred = clf_v.predict(Xte); prob = clf_v.predict_proba(Xte)
    lab = list(range(len(classes)))
    video_lr = {
        "C": C_v,
        "video_level": {
            "accuracy": accuracy_score(yte, pred),
            "macro_f1": f1_score(yte, pred, labels=lab, average="macro", zero_division=0),
            "f1_per_class": dict(zip(classes, f1_score(yte, pred, labels=lab, average=None, zero_division=0).tolist())),
            "auc_macro_ovr": roc_auc_score(yte, prob, multi_class="ovr", average="macro", labels=lab),
        },
        "video_level_binary": {"auc_1_menos_p_normal": roc_auc_score((yte != c2i[normal]).astype(int), 1 - prob[:, c2i[normal]])},
    }
    print(f"    TEST acc {video_lr['video_level']['accuracy']:.4f} | macro-F1 {video_lr['video_level']['macro_f1']:.4f} "
          f"| AUC binario {video_lr['video_level_binary']['auc_1_menos_p_normal']:.4f}")

    # ── 3. Logística por frame (normal vs anómalo) ───────────────────────────
    rng = np.random.default_rng(SEED)

    def frame_xy(s):
        Xs, ys = [], []
        for r, F in subset(s):
            idx = rng.choice(len(F), min(MAX_FRAMES_PER_VIDEO, len(F)), replace=False)
            Xs.append(F[idx]); ys.append(np.full(len(idx), int(r["class_name"] != normal)))
        return np.concatenate(Xs), np.concatenate(ys)

    Xtr, ytr = frame_xy("train"); Xva, yva = frame_xy("val")
    print("\n[3] Logística por frame (C elegido con AUC débil en val):")
    _, C_s, clf_s = fit_best(Xtr, ytr, Xva, yva, lambda c, X, y: roc_auc_score(y, c.predict_proba(X)[:, 1]))

    scores, labels, is_anom_video = [], [], []
    sin_anot = 0
    for r, F in subset("test"):
        sc = clf_s.predict_proba(F)[:, 1]
        fr = frame_numbers(r, len(F), every_n)
        anom = r["class_name"] != normal
        lb = np.zeros(len(F), dtype=int)
        if anom:
            iv = ann.get(r["video_id"])
            if iv is None:
                sin_anot += 1; continue
            for a, b in iv:
                lb |= ((fr >= a) & (fr <= b)).astype(int)
        scores.append(sc); labels.append(lb); is_anom_video.append(np.full(len(F), anom))
    S, L, A = np.concatenate(scores), np.concatenate(labels), np.concatenate(is_anom_video)
    snippet_lr = {"C": C_s, "frame_level": {
        "auc": roc_auc_score(L, S) if L.any() else None,
        "ap": average_precision_score(L, S) if L.any() else None,
        "auc_anomalous_only": roc_auc_score(L[A], S[A]) if L[A].any() and (L[A] == 0).any() else None,
        "n_frames": int(len(S)), "anomalous_videos_without_annotation": sin_anot}}
    fl = snippet_lr["frame_level"]
    print(f"    TEST AUC frame {fl['auc']} | AP {fl['ap']} | AUC solo anómalos {fl['auc_anomalous_only']}")
    if sin_anot:
        print(f"    [AVISO] {sin_anot} videos anómalos de test sin anotación temporal (se omiten)")

    # ── Guardar ──────────────────────────────────────────────────────────────
    out = Path("artifacts/seeds") / args.run / "baseline_logistica"
    out.mkdir(parents=True, exist_ok=True)
    m = {"config": args.config, "features_dir": str(feat_dir), "processed_dir": str(proc), "classes": classes,
         "n_videos": {s: len(subset(s)) for s in splits}, "missing_features": faltan,
         "majority": majority, "video_logreg": video_lr, "snippet_logreg": snippet_lr,
         "summary": {"video_level": {"accuracy": video_lr["video_level"]["accuracy"],
                                     "macro_f1": video_lr["video_level"]["macro_f1"]},
                     "video_level_binary": video_lr["video_level_binary"],
                     "frame_level": {k: fl[k] for k in ("auc", "ap", "auc_anomalous_only")}}}
    (out / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nGuardado: {out / 'metrics.json'}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
