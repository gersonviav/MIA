"""
Tian et al. (ICCV 2021) "Weakly-supervised Video Anomaly Detection with Robust Temporal Feature
Magnitude Learning" (RTFM). Adaptado de https://github.com/tianyu0207/RTFM (model.py, train.py):
  módulo MTN (convoluciones dilatadas 1/2/4 + bloque no-local), FC 512-128-1, dropout 0.7,
  selección top-k (k=3) por magnitud de característica con dropout sobre la selección,
  pérdida BCE + 1e-4 * RTFM(margen=100) + 8e-4 * suavidad + 8e-3 * dispersión.
Adaptaciones (declaradas en la tesis):
  (a) dimensión de entrada parametrizable (I3D Inception = 1024; el original asumía 2048);
  (b) código independiente del dispositivo (sin .cuda() fijo);
  (c) cabeza de 3 clases sobre el promedio de los k segmentos de mayor magnitud.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import ClassHead, gather_topk_mean, ce_loss


class NonLocal1D(nn.Module):
    """_NonLocalBlockND (dimension=1, sub_sample=False, bn_layer=True) del repo oficial."""

    def __init__(self, c):
        super().__init__()
        ic = c // 2
        self.ic = ic
        self.g = nn.Conv1d(c, ic, 1)
        self.theta = nn.Conv1d(c, ic, 1)
        self.phi = nn.Conv1d(c, ic, 1)
        self.W = nn.Sequential(nn.Conv1d(ic, c, 1), nn.BatchNorm1d(c))
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self, x):
        b = x.size(0)
        g = self.g(x).view(b, self.ic, -1).permute(0, 2, 1)
        th = self.theta(x).view(b, self.ic, -1).permute(0, 2, 1)
        ph = self.phi(x).view(b, self.ic, -1)
        f = torch.matmul(th, ph)
        f = f / f.size(-1)
        y = torch.matmul(f, g).permute(0, 2, 1).contiguous().view(b, self.ic, *x.size()[2:])
        return self.W(y) + x


class Aggregate(nn.Module):
    def __init__(self, d):
        super().__init__()

        def dil(dl):
            return nn.Sequential(nn.Conv1d(d, 512, 3, 1, padding=dl, dilation=dl), nn.ReLU(), nn.BatchNorm1d(512))

        self.conv_1, self.conv_2, self.conv_3 = dil(1), dil(2), dil(4)
        self.conv_4 = nn.Sequential(nn.Conv1d(d, 512, 1, bias=False), nn.ReLU())
        self.conv_5 = nn.Sequential(nn.Conv1d(2048, d, 3, padding=1, bias=False), nn.ReLU(), nn.BatchNorm1d(d))
        self.non_local = NonLocal1D(512)

    def forward(self, x):  # x [B,T,D]
        out = x.permute(0, 2, 1)
        res = out
        out_d = torch.cat([self.conv_1(out), self.conv_2(out), self.conv_3(out)], 1)
        out = self.non_local(self.conv_4(out))
        out = self.conv_5(torch.cat([out_d, out], 1)) + res
        return out.permute(0, 2, 1)


def _xavier(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class RTFM(nn.Module):
    test_segments = None  # test sobre todos los snippets (como el original)
    score_kind = "snippet16"

    def __init__(self, d_in=1024, n_classes=3, k=3, dropout=0.7, alpha=1e-4, margin=100.0):
        super().__init__()
        self.agg = Aggregate(d_in)
        self.fc1, self.fc2, self.fc3 = nn.Linear(d_in, 512), nn.Linear(512, 128), nn.Linear(128, 1)
        self.drop = nn.Dropout(dropout)
        self.k, self.alpha, self.margin = k, alpha, margin
        self.apply(_xavier)
        # NonLocal: el original re-inicializa W[1] a 0 (BN) -> se respeta porque _xavier no toca BN
        self.cls_head = ClassHead(d_in, n_classes)

    def encode(self, x):
        """x [B,N,T,D] -> feats [B,N,T,D'], scores [B,T] (media crops), mag [B,T] (media crops)"""
        B, N, T, D = x.shape
        f = self.drop(self.agg(x.reshape(B * N, T, D)))
        s = F.relu(self.fc1(f))
        s = F.relu(self.fc2(self.drop(s)))
        s = torch.sigmoid(self.fc3(self.drop(s))).view(B, N, T).mean(1)
        f = f.view(B, N, T, -1)
        mag = torch.norm(f, p=2, dim=3).mean(1)
        return f, s, mag

    def _select(self, mag, train):
        if train:
            mag = mag * self.drop(torch.ones_like(mag))
        return torch.topk(mag, min(self.k, mag.size(1)), dim=1)[1]

    def _logits(self, f, idx):
        emb = gather_topk_mean(f.mean(1), idx)  # media de crops, luego top-k
        return self.cls_head(emb)

    def train_loss(self, xn, yn, xa, ya, ce_weight, lam_cls):
        B = xn.size(0)
        f, s, mag = self.encode(torch.cat([xn, xa]))
        idx_n, idx_a = self._select(mag[:B], True), self._select(mag[B:], True)
        score_n = torch.gather(s[:B], 1, idx_n).mean(1)
        score_a = torch.gather(s[B:], 1, idx_a).mean(1)
        N = f.size(1)

        def sel(feat, idx):  # [b,N,T,D] -> [b*N,k,D] (por crop, como el original)
            ii = idx[:, None, :, None].expand(-1, N, -1, feat.size(-1))
            return torch.gather(feat, 2, ii).reshape(-1, idx.size(1), feat.size(-1))

        fa, fn = sel(f[B:], idx_a), sel(f[:B], idx_n)
        label = torch.cat([torch.zeros(B), torch.ones(B)]).to(s.device)
        l_bce = F.binary_cross_entropy(torch.cat([score_n, score_a]).clamp(1e-6, 1 - 1e-6), label)
        l_abn = torch.abs(self.margin - torch.norm(fa.mean(1), p=2, dim=1))
        l_nor = torch.norm(fn.mean(1), p=2, dim=1)
        l_rtfm = torch.mean((l_abn + l_nor) ** 2)
        abn = s[B:].reshape(-1)  # igual que el original: vector aplanado de scores anómalos
        smooth = 8e-4 * torch.sum((torch.cat([abn[1:], abn[-1:]]) - abn) ** 2)
        sparse = 8e-3 * torch.mean(torch.norm(abn, dim=0))
        l_mil = l_bce + self.alpha * l_rtfm + smooth + sparse
        idx_all = torch.cat([idx_n, idx_a])
        l_cls = ce_loss(self._logits(f, idx_all), torch.cat([yn, ya]), ce_weight)
        return l_mil + lam_cls * l_cls, {"mil": l_mil.item(), "cls": l_cls.item()}

    @torch.no_grad()
    def infer(self, x):
        f, s, mag = self.encode(x)
        idx = self._select(mag, False)
        return {"snippet_scores": s[0], "video_score": torch.gather(s, 1, idx).mean(),
                "logits": self._logits(f, idx)[0]}
