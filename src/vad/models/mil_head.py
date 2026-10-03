"""
SnippetMIL — cabeza v5 (clasificación) + scorer LOCAL de anomalía por snippet.

    rgb[T,768] → Linear ─┐
                         ├─ z[T,512] ──► + pos_emb → Transformer → attention pooling → video_logits [3]
    mot[T,768] → Linear ─┘      │                         └──────► head por posición → seg_logits [T,3]
                                │
                                └──► scorer local (Conv1d k=3 → Conv1d 1) → sigmoid → score_t ∈ [0,1]

¿Por qué el score NO sale del Transformer? Con self-attention global cada posición "ve" todo el
video: si el video es una pelea, todas las posiciones suben juntas y el score no localiza
(AUC solo-anómalos ≈ 0.47 en la v0.4). El scorer local solo mira el snippet y sus vecinos
inmediatos, así que solo sube donde las features cambian.

score_head: "transformer" reproduce el comportamiento anterior (score = 1 − P(Normal)).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalScorer(nn.Module):
    def __init__(self, d: int, hidden: int, kernel: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, hidden, kernel, padding=kernel // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:        # [B,T,d] → [B,T] logits
        return self.net(z.transpose(1, 2)).squeeze(1)


class SnippetMIL(nn.Module):
    def __init__(self, dim_rgb: int, dim_mot: int, cfg: dict, num_classes: int = 3):
        super().__init__()
        m = cfg["model"]
        d = m["d_model"]
        self.attn_temp = m["attn_temp"]
        self.topk = m["topk"]
        self.score_head = m.get("score_head", "local")
        max_len = max(cfg["mil"]["num_segments"], cfg["mil"]["window"])
        self.proj_spat = nn.Linear(dim_rgb, d)
        self.proj_mot = nn.Linear(dim_mot, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=m["nhead"], batch_first=True,
                                           dim_feedforward=d * 2, dropout=m["dropout"])
        self.transformer = nn.TransformerEncoder(layer, num_layers=m["num_layers"],
                                                 enable_nested_tensor=False)
        self.attn = nn.Linear(d, 1)
        self.head = nn.Linear(d, num_classes)
        if self.score_head == "local":
            self.scorer = LocalScorer(d, m.get("scorer_hidden", 256), m.get("scorer_kernel", 3), m["dropout"])

    def _attention_pool(self, f: torch.Tensor):
        logits = self.attn(f) / self.attn_temp
        if self.topk and 0 < self.topk < f.shape[1]:
            thr = logits.topk(self.topk, dim=1).values[:, -1:, :]
            logits = logits.masked_fill(logits < thr, float("-inf"))
        w = F.softmax(logits, dim=1)
        return (f * w).sum(1), w

    def snippet_scores(self, rgb: torch.Tensor, mot: torch.Tensor) -> torch.Tensor:
        """Score por posición sin pasar por el Transformer (solo score_head='local'); acepta cualquier longitud."""
        return torch.sigmoid(self.scorer(self.proj_spat(rgb) + self.proj_mot(mot)))

    def forward(self, rgb: torch.Tensor, mot: torch.Tensor) -> dict:
        T = rgb.shape[1]
        z = self.proj_spat(rgb) + self.proj_mot(mot)
        f = self.transformer(z + self.pos_emb[:, :T])
        pooled, w = self._attention_pool(f)
        seg_logits = self.head(f)
        if self.score_head == "local":
            scores = torch.sigmoid(self.scorer(z))
        else:
            scores = 1.0 - F.softmax(seg_logits.float(), dim=-1)[..., 0]
        return {"video_logits": self.head(pooled), "w": w, "seg_logits": seg_logits, "scores": scores}
