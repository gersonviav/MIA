"""
Zhou, Yu & Yang (AAAI 2023) "Dual Memory Units with Uncertainty Regulation for Weakly Supervised
Video Anomaly Detection" (UR-DMU). Adaptado de https://github.com/henrryzh1/UR-DMU (MIT):
  model.py (WSAD), memory.py (Memory_Unit), translayer.py (Transformer GL-MHSA), train.py (AD_Loss).
  Hiperparámetros originales UCF: a_nums = n_nums = 60, Transformer(512, depth 2, 4 cabezas, 128),
  200 segmentos en entrenamiento, Adam lr 1e-4, wd 5e-5, k = T//16 + 1.
Adaptaciones (declaradas en la tesis):
  (a) código independiente del dispositivo (el original fija .cuda() en la atención y la pérdida);
  (b) cabeza de 3 clases sobre el promedio de los k segmentos con mayor puntuación de anomalía,
      usando el mismo vector (1024-d: embedding + memoria aumentada) que alimenta a la cabeza binaria.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import ClassHead, gather_topk_mean, ce_loss


def l2n(x):
    return x / torch.norm(x, p=2, dim=-1, keepdim=True)


class MemoryUnit(nn.Module):
    def __init__(self, nums, dim):
        super().__init__()
        self.dim, self.nums = dim, nums
        self.memory_block = nn.Parameter(torch.empty(nums, dim))
        stdv = 1.0 / math.sqrt(dim)
        self.memory_block.data.uniform_(-stdv, stdv)

    def forward(self, data):  # B,T,D
        att = torch.sigmoid(torch.einsum("btd,kd->btk", data, self.memory_block) / (self.dim ** 0.5))
        temporal_att = torch.topk(att, self.nums // 16 + 1, dim=-1)[0].mean(-1)
        augment = torch.einsum("btk,kd->btd", att, self.memory_block)
        return temporal_att, augment


class GLAttention(nn.Module):
    """Atención global (softmax) + local (núcleo de distancia temporal) del repo oficial."""

    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner = dim_head * heads
        self.h, self.scale = heads, dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner * 4, bias=False)
        self.to_out = nn.Sequential(nn.Linear(2 * inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v, t = [z.reshape(b, n, self.h, -1).transpose(1, 2) for z in self.to_qkv(x).chunk(4, -1)]
        a1 = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, -1)
        pos = torch.linspace(1, n, n, device=x.device)
        a2 = torch.exp(-torch.abs(pos[None, :] - pos[:, None]) / math.e)
        a2 = (a2 / a2.sum(-1)).unsqueeze(0).unsqueeze(0)
        out = torch.cat([torch.matmul(a1, v), torch.matmul(a2.expand(b, self.h, n, n), t)], -1)
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Sequential(nn.LayerNorm(dim), GLAttention(dim, heads, dim_head, dropout)),
                nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                              nn.Linear(mlp_dim, dim), nn.Dropout(dropout))]))

    def forward(self, x):
        for att, ff in self.layers:
            x = att(x) + x
            x = ff(x) + x
        return x


class URDMU(nn.Module):
    test_segments = None
    score_kind = "snippet16"

    def __init__(self, d_in=1024, n_classes=3, a_nums=60, n_nums=60):
        super().__init__()
        self.embedding = nn.Sequential(nn.Conv1d(d_in, 512, 3, padding=1), nn.ReLU())
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.5)
        self.Amemory, self.Nmemory = MemoryUnit(a_nums, 512), MemoryUnit(n_nums, 512)
        self.encoder_mu, self.encoder_var = nn.Linear(512, 512), nn.Linear(512, 512)
        self.cls_bin = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.triplet = nn.TripletMarginLoss(margin=1)
        self.cls_head = ClassHead(1024, n_classes)

    def _embed(self, x):  # [B,T,D] -> [B,T,512]
        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.selfatt(x)

    @staticmethod
    def _k(t):
        return t // 16 + 1

    def train_loss(self, xn, yn, xa, ya, ce_weight, lam_cls):
        # crops: el original entrena con un crop por muestra -> sampler en modo 'random' (N=1)
        x = torch.cat([xn, xa])[:, 0]
        b, t, _ = x.shape
        B = b // 2
        k = self._k(t)
        x = self._embed(x)
        N_x, A_x = x[:B], x[B:]
        A_att, A_aug = self.Amemory(A_x)
        N_Aatt, N_Aaug = self.Nmemory(A_x)
        A_Natt, A_Naug = self.Amemory(N_x)
        N_att, N_aug = self.Nmemory(N_x)

        def pick(src, att):
            idx = torch.topk(att, k, dim=-1)[1]
            return torch.gather(src, 1, idx.unsqueeze(2).expand(-1, -1, src.size(-1))).mean(1), idx

        negative_ax, A_index = pick(A_x, A_att)
        anchor_nx, N_index = pick(N_x, N_att)
        positive_nx, _ = pick(A_x, N_Aatt)
        triplet = self.triplet(l2n(anchor_nx), l2n(positive_nx), l2n(negative_ax))

        N_mu, N_var = self.encoder_mu(N_aug), self.encoder_var(N_aug)
        N_aug_new = N_mu + torch.randn_like(N_var) * torch.exp(N_var).sqrt()
        anchor_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand(-1, -1, 512)).mean(1)
        A_aug_new = self.encoder_mu(A_aug)
        negative_new = torch.gather(A_aug_new, 1, A_index.unsqueeze(2).expand(-1, -1, 512)).mean(1)
        kl = torch.mean(-0.5 * torch.sum(1 + N_var - N_mu ** 2 - N_var.exp(), dim=1))
        A_Naug, N_Aaug = self.encoder_mu(A_Naug), self.encoder_mu(N_Aaug)
        distance = torch.relu(100 - torch.norm(negative_new, p=2, dim=-1) + torch.norm(anchor_new, p=2, dim=-1)).mean()
        xc = torch.cat([x, torch.cat([N_aug_new + A_Naug, A_aug_new + N_Aaug], 0)], -1)  # [b,t,1024]
        frame = self.cls_bin(xc).squeeze(-1)  # [b,t]

        # AD_Loss original
        bce = nn.BCELoss()
        lbl = torch.cat([torch.zeros(B), torch.ones(B)]).to(x.device)
        ones = torch.ones(B, device=x.device)
        anomaly_loss = bce(torch.topk(frame, k, -1)[0].mean(-1), lbl)
        pan_loss = bce(torch.topk(1 - N_Aatt, k, -1)[0].mean(-1), ones)
        A_loss = bce(torch.topk(A_att, k, -1)[0].mean(-1), ones)
        N_loss = bce(N_att, torch.ones_like(N_att))
        A_Nloss = bce(A_Natt, torch.zeros_like(A_Natt))
        l_mil = anomaly_loss + 0.1 * (A_loss + pan_loss + N_loss + A_Nloss) + 0.1 * triplet \
            + 0.001 * kl + 0.0001 * distance

        idx = torch.topk(frame, k, dim=1)[1]
        l_cls = ce_loss(self.cls_head(gather_topk_mean(xc, idx)), torch.cat([yn, ya]), ce_weight)
        return l_mil + lam_cls * l_cls, {"mil": l_mil.item(), "cls": l_cls.item()}

    @torch.no_grad()
    def infer(self, x):
        # x [1,N,T,D]: cada crop se procesa por separado y se promedia (como ucf_test.py)
        x = x[0]
        t = x.size(1)
        e = self._embed(x)
        _, A_aug = self.Amemory(e)
        _, N_aug = self.Nmemory(e)
        xc = torch.cat([e, self.encoder_mu(A_aug) + self.encoder_mu(N_aug)], -1)  # [N,T,1024]
        frame = self.cls_bin(xc).squeeze(-1).mean(0)  # [T]
        idx = torch.topk(frame, self._k(t))[1]
        emb = xc.mean(0)[idx].mean(0, keepdim=True)
        return {"snippet_scores": frame, "video_score": frame[idx].mean(), "logits": self.cls_head(emb)[0]}
