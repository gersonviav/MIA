"""Componentes compartidos por los tres baselines adaptados a 3 clases."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassHead(nn.Module):
    """Única modificación arquitectónica sobre los baselines: cabeza de 3 clases
    (Normal / Aggression / Theft) aplicada al promedio de los k segmentos que el propio
    método selecciona como más anómalos (misma regla top-k del paper original)."""

    def __init__(self, dim, n_classes=3, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, n_classes))

    def forward(self, emb):
        return self.net(emb)


def gather_topk_mean(feats, idx):
    """feats [B,T,E], idx [B,k] -> [B,E] (promedio de los segmentos seleccionados)."""
    g = torch.gather(feats, 1, idx.unsqueeze(-1).expand(-1, -1, feats.size(-1)))
    return g.mean(1)


def ce_loss(logits, y, weight):
    return F.cross_entropy(logits, y, weight=weight)
