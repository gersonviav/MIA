"""Datasets sobre features pre-extraídas (etapa 2)."""

from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from vad.data.common import read_jsonl
from vad.temporal import snippet_frame_ranges, to_segments, to_snippets

log = logging.getLogger("vad.dataset")


def load_split(cfg: dict, split: str) -> list[dict]:
    p = Path(cfg["paths"]["processed_dir"]) / "splits" / f"{split}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"No existe {p}. Ejecuta: vad preprocess")
    return read_jsonl(p)


def load_video_snippets(cfg: dict, video_id: str) -> dict:
    """Features por snippet de un video + rango de frames originales de cada snippet."""
    path = Path(cfg["features"]["dir"]) / f"{video_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Faltan features de {video_id}. Ejecuta: vad extract")
    z = np.load(path)
    k = cfg["snippet"]["frames_per_snippet"]
    return {"rgb": to_snippets(z["rgb"], k), "mot": to_snippets(z["mot"], k),
            "ranges": snippet_frame_ranges(z["frame_indices"], k)}


class SegmentDataset(Dataset):
    """
    Entrenamiento/validación. Cada video se entrega en dos resoluciones:
      - num_segments (32):     vista gruesa para el Transformer / clasificación del video
      - scorer_segments (128): vista fina para el scorer de anomalía (≈1–2 s por segmento),
                               más cerca de los snippets de 1 s que se puntúan en inferencia
    """

    def __init__(self, cfg: dict, split: str):
        self.samples = load_split(cfg, split)
        t = cfg["mil"]["num_segments"]
        ts = cfg["mil"].get("scorer_segments", t)
        self.rgb, self.mot, self.rgb_s, self.mot_s, n_snips = [], [], [], [], []
        for s in self.samples:
            v = load_video_snippets(cfg, s["video_id"])
            n_snips.append(len(v["rgb"]))
            self.rgb.append(torch.from_numpy(to_segments(v["rgb"], t)))
            self.mot.append(torch.from_numpy(to_segments(v["mot"], t)))
            self.rgb_s.append(torch.from_numpy(to_segments(v["rgb"], ts)).half())
            self.mot_s.append(torch.from_numpy(to_segments(v["mot"], ts)).half())
        self.labels = [s["label"] for s in self.samples]
        log.info("  %s: %d videos | dist %s | %d segmentos (Transformer) / %d (scorer) | snippets/video "
                 "mediana %d", split, len(self.samples), dict(sorted(Counter(self.labels).items())), t, ts,
                 int(np.median(n_snips)) if n_snips else 0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        return self.rgb[i], self.mot[i], self.labels[i], i, self.rgb_s[i].float(), self.mot_s[i].float()


class BalancedBatchSampler(Sampler):
    """Cada batch: mitad videos normales, mitad anómalos (requisito del ranking MIL)."""

    def __init__(self, labels: list[int], batch_size: int, seed: int = 0):
        self.norm = [i for i, y in enumerate(labels) if y == 0]
        self.anom = [i for i, y in enumerate(labels) if y != 0]
        if not self.norm or not self.anom:
            raise ValueError("El ranking MIL necesita videos normales y anómalos en train.")
        self.half = max(batch_size // 2, 1)
        self.rng = random.Random(seed)
        self.n_batches = int(np.ceil(max(len(self.norm), len(self.anom)) / self.half))

    def _stream(self, pool: list[int]):
        while True:
            p = pool[:]
            self.rng.shuffle(p)
            yield from p

    def __iter__(self):
        sn, sa = self._stream(self.norm), self._stream(self.anom)
        for _ in range(self.n_batches):
            yield [next(sn) for _ in range(self.half)] + [next(sa) for _ in range(self.half)]

    def __len__(self) -> int:
        return self.n_batches
