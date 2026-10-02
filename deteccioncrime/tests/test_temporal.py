"""Lógica temporal: snippets, segmentos, ventanas deslizantes y mapeo a frames."""

import numpy as np

from vad.temporal import (frame_level, gt_frame_labels, intervals_from_scores, merge_window_scores,
                          segment_bounds, snippet_frame_ranges, to_segments, to_snippets, window_starts)


def test_ejemplo_del_readme_120_snippets():
    starts = window_starts(120, 32, 16)
    assert starts == [0, 16, 32, 48, 64, 80, 88]          # 7 ventanas, la última alineada al final
    covered = np.zeros(120)
    for s in starts:
        covered[s:s + 32] += 1
    assert covered.min() >= 1                              # ningún snippet queda sin ver
    scores = merge_window_scores(120, 32, starts, np.ones((7, 32)))
    assert scores.shape == (120,) and np.allclose(scores, 1)


def test_video_corto_una_sola_ventana():
    assert window_starts(10, 32, 16) == [0]


def test_snippets_y_rangos():
    x = np.arange(10, dtype=np.float32)[:, None]
    fi = np.arange(0, 100, 10)                             # frames originales 0,10,...,90
    assert to_snippets(x, 3).ravel().tolist() == [1.0, 4.0, 7.0, 9.0]
    assert snippet_frame_ranges(fi, 3).tolist() == [[0, 30], [30, 60], [60, 90], [90, 100]]


def test_segmentos_cubren_todo_y_rellenan_si_faltan():
    b = segment_bounds(100, 32)
    assert b[0][0] == 0 and b[-1][1] == 100 and len(b) == 32
    assert to_segments(np.random.rand(5, 4), 32).shape == (32, 4)   # 5 snippets → 32 segmentos


def test_frame_level_e_intervalos():
    ranges = np.array([[0, 30], [30, 60], [60, 90]])
    s = np.array([0.1, 0.9, 0.2])
    f = frame_level(s, ranges)
    assert f.shape == (90,) and f[45] == np.float32(0.9) and f[10] == np.float32(0.1)
    assert gt_frame_labels(90, [(30, 59)]).sum() == 30
    iv = intervals_from_scores(s, ranges, fps=30, thr=0.5, min_dur_s=0.5)
    assert iv == [(1.0, 2.0, 0.9)]
