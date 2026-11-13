"""Métricas comunes para TODOS los modelos (propuesto y baselines) + formato de predicciones.

Formato de predicciones (un CSV por modelo y semilla):
  video_id, split, y_true, y_pred, p_Normal, p_Aggression, p_Theft, anomaly_score
  anomaly_score: puntuación continua de anomalía a nivel de video
     - baselines MIL: score de su cabeza binaria (max / top-k medio de snippets)
     - modelo propuesto: 1 - p_Normal
Opcional (para AUC a nivel de frame): NPZ con un array de scores por video + JSON con
  {"kind": "snippet16" | "uniform", "n_frames": {video_id: n}}.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score, roc_auc_score)

CLASSES = ["Normal", "Aggression", "Theft"]


def video_metrics(y_true, y_pred, anomaly_score=None):
    """F1 por clase, F1-macro (3 clases), F1-macro de anomalías (Aggression, Theft),
    F1-anomalía binaria ({Aggression ∪ Theft} vs Normal), AUC/AP de anomalía a nivel de video."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rep = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASSES,
                                output_dict=True, zero_division=0)
    yb, pb = (y_true > 0).astype(int), (y_pred > 0).astype(int)
    m = {
        "f1_macro": f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0),
        "f1_macro_anom": f1_score(y_true, y_pred, labels=[1, 2], average="macro", zero_division=0),
        "f1_anomaly": f1_score(yb, pb, pos_label=1, zero_division=0),
        "precision_anomaly": rep_bin(yb, pb)[0],
        "recall_anomaly": rep_bin(yb, pb)[1],
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
    for c in CLASSES:
        m[f"f1_{c}"] = rep[c]["f1-score"]
        m[f"recall_{c}"] = rep[c]["recall"]
        m[f"precision_{c}"] = rep[c]["precision"]
    if anomaly_score is not None and len(np.unique(yb)) == 2:
        m["auc_video"] = roc_auc_score(yb, anomaly_score)
        m["ap_video"] = average_precision_score(yb, anomaly_score)
    return m


def rep_bin(yb, pb):
    tp = int(((yb == 1) & (pb == 1)).sum())
    p = tp / max(int(pb.sum()), 1)
    r = tp / max(int(yb.sum()), 1)
    return p, r


# ---------------------------------------------------------------- AUC a nivel de frame
def load_temporal_annotations(path):
    """Temporal_Anomaly_Annotation.txt (repo de Sultani): nombre clase s1 e1 s2 e2 (-1 = no hay)."""
    ann = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) < 6:
            continue
        ann[Path(p[0]).stem] = [(int(p[i]), int(p[i + 1])) for i in (2, 4) if int(p[i]) >= 0]
    return ann


def frame_gt(vid, n_frames, ann, is_anom):
    gt = np.zeros(n_frames, dtype=np.uint8)
    if is_anom:
        for s, e in ann.get(vid, []):
            gt[max(s, 0):min(e + 1, n_frames)] = 1
    return gt


def scores_to_frames(scores, n_frames, kind):
    scores = np.asarray(scores, dtype=np.float32)
    if kind == "snippet16":  # un score por cada 16 frames consecutivos
        f = np.repeat(scores, 16)
        if len(f) < n_frames:
            f = np.concatenate([f, np.full(n_frames - len(f), f[-1])])
        return f[:n_frames]
    if kind == "uniform":  # S segmentos uniformes sobre el video (Sultani 32, modelo propuesto L=16)
        r = np.linspace(0, n_frames, len(scores) + 1).astype(int)
        f = np.empty(n_frames, dtype=np.float32)
        for i, sc in enumerate(scores):
            f[r[i]:r[i + 1]] = sc
        return f
    raise ValueError(kind)


def frame_level_auc(seg_scores, n_frames, kind, ann, labels):
    """labels: dict video_id -> is_anom. Solo se usan videos anómalos con anotación temporal
    y videos normales (convención UCF-Crime). Devuelve AUC y AP a nivel de frame."""
    ys, ss, used = [], [], 0
    for vid, is_anom in labels.items():
        if vid not in seg_scores or (is_anom and vid not in ann):
            continue
        n = int(n_frames[vid])
        ys.append(frame_gt(vid, n, ann, is_anom))
        ss.append(scores_to_frames(seg_scores[vid], n, kind))
        used += 1
    y, s = np.concatenate(ys), np.concatenate(ss)
    return {"auc_frame": roc_auc_score(y, s), "ap_frame": average_precision_score(y, s), "n_videos_frame": used}


# ---------------------------------------------------------------- E/S
def write_predictions(path, video_ids, split, y_true, probs, anomaly_score):
    probs = np.asarray(probs)
    df = pd.DataFrame({"video_id": video_ids, "split": split, "y_true": np.asarray(y_true).astype(int),
                       "y_pred": probs.argmax(1)})
    for i, c in enumerate(CLASSES):
        df[f"p_{c}"] = probs[:, i]
    df["anomaly_score"] = np.asarray(anomaly_score, dtype=float)
    df.to_csv(path, index=False)
    return df


def write_segment_scores(path_npz, seg_scores, n_frames, kind):
    np.savez_compressed(path_npz, **{k: np.asarray(v, dtype=np.float32) for k, v in seg_scores.items()})
    Path(str(path_npz).replace(".npz", ".json")).write_text(
        json.dumps({"kind": kind, "n_frames": {k: int(v) for k, v in n_frames.items()}}))


def read_segment_scores(path_npz):
    z = np.load(path_npz)
    meta = json.loads(Path(str(path_npz).replace(".npz", ".json")).read_text())
    return {k: z[k] for k in z.files}, meta["n_frames"], meta["kind"]
