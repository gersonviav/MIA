"""
Lógica temporal (sin torch): frames → snippets → segmentos / ventanas → segundos.

    frames extraídos   f0 f1 f2 | f3 f4 f5 | f6 ...        (1 frame cada ~10 originales ≈ 0.33 s)
    snippets           s0       | s1       | s2 ...        (promedio de k frames ≈ 1 s)
    ENTRENAMIENTO      N snippets → 32 segmentos (promedio de los snippets que caen en cada uno)
    INFERENCIA         ventanas de W snippets que avanzan S; score final = promedio de las ventanas
"""

from __future__ import annotations

import numpy as np


def to_snippets(x: np.ndarray, k: int) -> np.ndarray:
    """[n_frames, D] → [ceil(n/k), D] promediando bloques de k frames consecutivos."""
    n = len(x)
    n_snip = max(int(np.ceil(n / k)), 1)
    out = np.empty((n_snip, x.shape[1]), dtype=np.float32)
    for i in range(n_snip):
        out[i] = x[i * k:(i + 1) * k].astype(np.float32).mean(0)
    return out


def snippet_frame_ranges(frame_indices: np.ndarray, k: int) -> np.ndarray:
    """[n_snip, 2] rango [inicio, fin) en frames ORIGINALES que cubre cada snippet."""
    fi = np.asarray(frame_indices, dtype=np.int64)
    step = int(np.median(np.diff(fi))) if len(fi) > 1 else 1
    n_snip = max(int(np.ceil(len(fi) / k)), 1)
    r = np.empty((n_snip, 2), dtype=np.int64)
    for i in range(n_snip):
        blk = fi[i * k:(i + 1) * k]
        r[i] = (blk[0], blk[-1] + max(step, 1))
    return r


def segment_bounds(n: int, t: int) -> list[tuple[int, int]]:
    """Divide n snippets en t segmentos [a, e). Si n < t, se repiten snippets."""
    b = np.linspace(0, n, t + 1)
    out = []
    for i in range(t):
        a = min(int(b[i]), n - 1)
        e = max(int(b[i + 1]), a + 1)
        out.append((a, min(e, n)))
    return out


def to_segments(x: np.ndarray, t: int) -> np.ndarray:
    """[n_snip, D] → [t, D] (promedio de snippets por segmento)."""
    return np.stack([x[a:e].mean(0) for a, e in segment_bounds(len(x), t)])


def window_starts(n: int, w: int, s: int) -> list[int]:
    """Inicios de ventana de tamaño w y paso s que cubren n snippets (la última se alinea al final)."""
    if n <= w:
        return [0]
    starts = list(range(0, n - w + 1, s))
    if starts[-1] + w < n:
        starts.append(n - w)
    return starts


def merge_window_scores(n: int, w: int, starts: list[int], window_scores: np.ndarray) -> np.ndarray:
    """Promedia los scores de ventanas solapadas → un score por snippet."""
    acc, cnt = np.zeros(n), np.zeros(n)
    for st, sc in zip(starts, window_scores):
        L = min(w, n - st)
        acc[st:st + L] += sc[:L]
        cnt[st:st + L] += 1
    return acc / np.maximum(cnt, 1)


def frame_level(scores: np.ndarray, ranges: np.ndarray, n_frames: int | None = None) -> np.ndarray:
    """Expande scores por snippet a un score por frame ORIGINAL del video."""
    n_frames = int(n_frames or ranges[-1, 1])
    out = np.zeros(n_frames, dtype=np.float32)
    for sc, (a, e) in zip(scores, ranges):
        out[a:min(e, n_frames)] = sc
    if ranges[0, 0] > 0:
        out[:ranges[0, 0]] = scores[0]
    return out


def gt_frame_labels(n_frames: int, intervals: list[tuple[int, int]]) -> np.ndarray:
    y = np.zeros(n_frames, dtype=np.int8)
    for a, e in intervals:
        y[max(a, 0):min(e + 1, n_frames)] = 1
    return y


def smooth_scores(scores: np.ndarray, win: int) -> np.ndarray:
    """Media móvil centrada de `win` snippets (win ≤ 1 → sin cambios)."""
    if win <= 1 or len(scores) < 2:
        return np.asarray(scores, dtype=np.float64)
    pad = win // 2
    x = np.pad(np.asarray(scores, dtype=np.float64), (pad, win - 1 - pad), mode="edge")
    return np.convolve(x, np.ones(win) / win, mode="valid")


def intervals_from_scores(scores: np.ndarray, ranges: np.ndarray, fps: float, thr: float,
                          min_dur_s: float, merge_gap_s: float = 0.0) -> list[tuple[float, float, float]]:
    """Snippets con score ≥ thr, unidos si son contiguos o separados por ≤ merge_gap_s
    → [(inicio_s, fin_s, score_max)]."""
    out, cur = [], None
    for sc, (a, e) in zip(scores, ranges):
        if sc >= thr:
            cur = [a, e, sc] if cur is None else [cur[0], e, max(cur[2], sc)]
        elif cur is not None:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    merged = []
    for a, e, m in out:
        if merged and (a - merged[-1][1]) / fps <= merge_gap_s:
            merged[-1] = [merged[-1][0], e, max(merged[-1][2], m)]
        else:
            merged.append([a, e, m])
    return [(a / fps, e / fps, float(m)) for a, e, m in merged if (e - a) / fps >= min_dur_s]
