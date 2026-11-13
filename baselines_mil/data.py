"""Manifiesto de videos (mismo split que el modelo propuesto) y carga de características I3D."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Normal", "Aggression", "Theft"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

# Mapeo de subclases UCF-Crime -> macro-clase (por si el manifiesto trae la subclase)
SUB2MACRO = {
    "NormalVideos": "Normal", "Normal": "Normal", "Testing_Normal_Videos_Anomaly": "Normal",
    "Training_Normal_Videos_Anomaly": "Normal", "Normal_Videos_event": "Normal",
    "Abuse": "Aggression", "Assault": "Aggression", "Fighting": "Aggression",
    "Robbery": "Theft", "Burglary": "Theft", "Stealing": "Theft", "Shoplifting": "Theft",
}


def load_manifest(path, col_video="video_id", col_label="label", col_split="split"):
    """CSV con una fila por video. Columnas mínimas: id de video, clase (macro o subclase) y split
    (train/val/test). Debe ser el MISMO archivo de split usado por el modelo propuesto."""
    df = pd.read_csv(path)
    df = df.rename(columns={col_video: "video_id", col_label: "label", col_split: "split"})
    df["video_id"] = df["video_id"].astype(str).map(lambda s: Path(s).stem)
    df["label"] = df["label"].map(lambda s: s if s in CLASSES else SUB2MACRO.get(s, s))
    bad = set(df["label"]) - set(CLASSES)
    assert not bad, f"Etiquetas no mapeadas: {bad}"
    df["split"] = df["split"].str.lower()
    assert set(df["split"]) <= {"train", "val", "test"}, set(df["split"])
    df["y"] = df["label"].map(CLS2IDX).astype(int)
    df["is_anom"] = (df["y"] > 0).astype(int)
    # Control de fuga: un video no puede estar en dos splits
    dup = df["video_id"].duplicated(keep=False)
    assert not dup.any(), f"Videos duplicados en el manifiesto: {df.loc[dup, 'video_id'].tolist()[:5]}"
    return df.reset_index(drop=True)


def resample_segments(feat, n_seg):
    """Promedia snippets en n_seg segmentos uniformes (process_feat de RTFM / Sultani 32 seg).
    feat: [T, D] -> [n_seg, D]"""
    T = feat.shape[0]
    r = np.linspace(0, T, n_seg + 1).astype(int)
    out = np.empty((n_seg, feat.shape[1]), dtype=np.float32)
    for i in range(n_seg):
        a, b = r[i], r[i + 1]
        out[i] = feat[a:b].mean(0) if b > a else feat[min(a, T - 1)]
    return out


class FeatureStore:
    """Carga perezosa de [ncrops, T, D] por video con caché de la versión re-muestreada."""

    def __init__(self, feat_dir, cache=True):
        self.dir = Path(feat_dir)
        mp = self.dir / "meta.json"
        self.meta = json.loads(mp.read_text()) if mp.exists() else {}
        self.cache = {} if cache else None

    def raw(self, vid):
        return np.load(self.dir / f"{vid}.npy", mmap_mode="r")  # [N, T, D]

    def n_frames(self, vid):
        if vid in self.meta:
            return self.meta[vid]["n_frames"]
        return self.raw(vid).shape[1] * 16

    def segments(self, vid, n_seg):
        """[N, n_seg, D] float16 (cacheado)."""
        key = (vid, n_seg)
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        raw = self.raw(vid)
        arr = np.stack([resample_segments(np.asarray(raw[c], dtype=np.float32), n_seg)
                        for c in range(raw.shape[0])]).astype(np.float16)
        if self.cache is not None:
            self.cache[key] = arr
        return arr


def crop_view(arr, mode, rng):
    """arr [N,T,D] -> según modo: 'all' [N,T,D], 'mean' [1,T,D], 'random' [1,T,D], 'center' [1,T,D]."""
    if mode == "all":
        return arr
    if mode == "mean":
        return arr.astype(np.float32).mean(0, keepdims=True)
    if mode == "random":
        return arr[rng.integers(arr.shape[0])][None]
    if mode == "center":
        return arr[2 if arr.shape[0] == 10 else 0][None]  # índice 2 = recorte central en 10-crop
    raise ValueError(mode)


class BalancedBagSampler:
    """Genera lotes con B videos normales + B anómalos (Aggression ∪ Theft), como en
    Sultani/RTFM/UR-DMU. Las etiquetas de clase (0/1/2) se devuelven para la cabeza de 3 clases."""

    def __init__(self, df_train, store, n_seg, crop_mode, batch_size, seed=0):
        self.store, self.n_seg, self.crop_mode, self.B = store, n_seg, crop_mode, batch_size
        self.rng = np.random.default_rng(seed)
        self.norm = df_train[df_train.is_anom == 0][["video_id", "y"]].values.tolist()
        self.anom = df_train[df_train.is_anom == 1][["video_id", "y"]].values.tolist()
        assert len(self.norm) >= batch_size and len(self.anom) >= batch_size, \
            "batch_size mayor que el número de videos de una partición"
        self._pn, self._pa = [], []

    def _draw(self, pool, buf):
        if len(buf) < self.B:
            buf.extend(self.rng.permutation(len(pool)).tolist())
        idx, buf[:] = buf[:self.B], buf[self.B:]
        vids = [pool[i] for i in idx]
        x = np.stack([crop_view(self.store.segments(v, self.n_seg), self.crop_mode, self.rng) for v, _ in vids])
        y = np.array([c for _, c in vids])
        return x.astype(np.float32), y

    def next(self):
        xn, yn = self._draw(self.norm, self._pn)
        xa, ya = self._draw(self.anom, self._pa)
        return xn, yn, xa, ya  # x: [B, N, S, D]
