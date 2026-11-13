"""
Sultani, Chen & Shah (CVPR 2018) "Real-world Anomaly Detection in Surveillance Videos".
Re-implementación en PyTorch fiel a TrainingAnomalyDetector_public.py (Keras/Theano):
  FC(D->512, ReLU) - Dropout(0.6) - FC(512->32, lineal) - Dropout(0.6) - FC(32->1, sigmoid)
  pérdida de ranking MIL con bisagra sobre el máximo de cada bolsa,
  + 8e-5 * suavidad temporal + 8e-5 * dispersión (solo bolsas anómalas); 32 segmentos L2-normalizados.
Adaptaciones (declaradas en la tesis):
  (a) entrada I3D-1024 en lugar de C3D-4096 (mismo extractor para todos los baselines);
  (b) cabeza de 3 clases sobre el embedding de 512-d del segmento top-1 (máximo MIL).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import ClassHead, gather_topk_mean, ce_loss


class SultaniMIL(nn.Module):
    test_segments = 32  # en el original, también el test usa 32 segmentos
    score_kind = "uniform"

    def __init__(self, d_in=1024, n_classes=3, dropout=0.6, lam=8e-5, k=1, l2norm=True):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(d_in, 512), nn.Linear(512, 32), nn.Linear(32, 1)
        self.drop = nn.Dropout(dropout)
        self.lam, self.k, self.l2norm = lam, k, l2norm
        self.cls_head = ClassHead(512, n_classes)
        for m in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_normal_(m.weight)  # glorot_normal
            nn.init.zeros_(m.bias)

    def encode(self, x):
        """x [B,N,T,D] -> promedio de crops (Sultani no usa crops) -> scores [B,T], h [B,T,512]"""
        x = x.mean(1)
        if self.l2norm:
            x = F.normalize(x, dim=-1)
        h = F.relu(self.fc1(x))
        s = torch.sigmoid(self.fc3(self.drop(self.fc2(self.drop(h))))).squeeze(-1)
        return s, h

    def logits(self, s, h):
        idx = torch.topk(s, min(self.k, s.size(1)), dim=1)[1]
        return self.cls_head(gather_topk_mean(h, idx))

    def train_loss(self, xn, yn, xa, ya, ce_weight, lam_cls):
        s, h = self.encode(torch.cat([xn, xa]))
        B = xn.size(0)
        sn, sa = s[:B], s[B:]
        mn, ma = sn.max(1)[0], sa.max(1)[0]
        # Para cada video normal i: sum_j max(0, 1 - max(a_j) + max(n_i)); luego promedio
        rank = F.relu(1.0 - ma.unsqueeze(0) + mn.unsqueeze(1)).sum(1).mean()
        smooth = ((sa[:, 1:] - sa[:, :-1]) ** 2).sum()
        sparse = sa.sum()
        l_mil = rank + self.lam * (smooth + sparse)
        l_cls = ce_loss(self.logits(s, h), torch.cat([yn, ya]), ce_weight)
        loss = l_mil + lam_cls * l_cls
        return loss, {"mil": l_mil.item(), "cls": l_cls.item()}

    @torch.no_grad()
    def infer(self, x):
        s, h = self.encode(x)
        return {"snippet_scores": s[0], "video_score": s[0].max(), "logits": self.logits(s, h)[0]}
