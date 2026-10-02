# models_v5.py
"""
v5 — Localización temporal real (corrige el colapso de atención de v4)
=======================================================================
Diagnóstico del problema en v4:
  w = softmax(attn(f)) y pooled = (f*w).sum(). Nada en la pérdida incentiva
  que w sea puntiaguda. Con w uniforme (1/16 = 0.0625) el pooling se reduce a
  un mean pooling, que ya minimiza la CE a nivel de video. Resultado observado:
  todos los pesos entre 0.062 y 0.087 -> la atención no localiza nada.

Tres mecanismos añadidos (combinables):
  1. TEMPERATURA (--attn_temp): softmax(logits/τ). τ<1 agudiza la distribución.
  2. REGULARIZACIÓN DE ENTROPÍA (en train_v5.py): penaliza H(w) alta.
     IMPORTANTE: solo se aplica a videos anómalos. En un video normal no hay
     segmento anómalo que señalar, así que forzar atención puntiaguda ahí sería
     incorrecto conceptualmente.
  3. TOP-K POOLING (--topk): estilo Sultani et al. — se agregan solo los k
     segmentos de mayor puntaje, renormalizando el softmax entre ellos.
     Fuerza estructuralmente que la decisión dependa de pocos segmentos.

forward() devuelve 4 valores: (video_logits, w, seg_logits, attn_logits)
El cuarto es necesario para calcular la entropía en la pérdida.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange


class BaselineModels(nn.Module):
    def __init__(self, model_type="full_model", num_classes=3, d_model=512,
                 chunk_size=16, freeze_backbone=True, num_layers=2, num_segments=16,
                 dropout=0.3, attn_temp=1.0, topk=0):
        super().__init__()
        self.model_type = model_type
        self.chunk_size = chunk_size
        self.freeze_backbone = freeze_backbone
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_segments = num_segments
        self.attn_temp = attn_temp      # τ < 1 agudiza; 1.0 = comportamiento v4
        self.topk = topk                # 0 = desactivado (usa todos los segmentos)

        self.spatial_enc = timm.create_model("convnext_small", pretrained=True, num_classes=0)
        self.proj_spat = nn.Linear(self.spatial_enc.num_features, d_model)
        if "dual" in model_type or model_type == "full_model":
            self.motion_enc = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
            self.proj_mot = nn.Linear(self.motion_enc.num_features, d_model)

        if model_type == "cnn_lstm":
            self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
            self.head = nn.Linear(d_model, num_classes)

        elif model_type in ["single_trans", "full_model"]:
            self.pos_emb = nn.Parameter(torch.zeros(1, num_segments, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, batch_first=True,
                dim_feedforward=d_model * 2, dropout=dropout
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.attn = nn.Linear(d_model, 1)
            self.head = nn.Linear(d_model, num_classes)

        elif model_type in ["cnn_rgb", "dual_no_trans", "dual_avg_pool"]:
            in_dim = d_model * 2 if model_type == "dual_no_trans" else d_model
            self.head = nn.Linear(in_dim, num_classes)

        if self.freeze_backbone:
            self._freeze_backbones()

    # ---------------- congelado ----------------
    def _freeze_backbones(self):
        for p in self.spatial_enc.parameters():
            p.requires_grad = False
        self.spatial_enc.eval()
        if hasattr(self, "motion_enc"):
            for p in self.motion_enc.parameters():
                p.requires_grad = False
            self.motion_enc.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.spatial_enc.eval()
            if hasattr(self, "motion_enc"):
                self.motion_enc.eval()
        return self

    def trainable_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    # ---------------- encoding ----------------
    def _run_encoder(self, inp, encoder):
        if self.freeze_backbone:
            with torch.no_grad():
                return encoder(inp)
        return encoder(inp)

    def _encode_chunked(self, x, encoder, proj, chunk_size):
        if chunk_size is None or x.shape[0] <= chunk_size:
            return proj(self._run_encoder(x, encoder))
        outs = []
        for i in range(0, x.shape[0], chunk_size):
            outs.append(proj(self._run_encoder(x[i:i + chunk_size], encoder)))
        return torch.cat(outs, dim=0)

    # ---------------- pooling con atención ----------------
    def _attention_pool(self, f):
        """
        f: (B, T, D)
        Devuelve (pooled, w, attn_logits)
          pooled      : (B, D)
          w           : (B, T, 1) pesos normalizados que suman 1
          attn_logits : (B, T, 1) logits crudos, necesarios para la entropía
        """
        attn_logits = self.attn(f)                       # (B, T, 1)
        scaled = attn_logits / self.attn_temp            # temperatura

        if self.topk and 0 < self.topk < f.shape[1]:
            # Top-k pooling: solo los k segmentos de mayor puntaje participan.
            # Los demás reciben -inf antes del softmax -> peso exactamente 0.
            k = self.topk
            thresh = scaled.topk(k, dim=1).values[:, -1:, :]   # k-ésimo valor
            masked = scaled.masked_fill(scaled < thresh, float("-inf"))
            w = F.softmax(masked, dim=1)
        else:
            w = F.softmax(scaled, dim=1)

        pooled = (f * w).sum(dim=1)                      # (B, D)
        return pooled, w, attn_logits

    # ---------------- forward ----------------
    def forward(self, rgb, motion=None, chunk_size=None):
        """
        Devuelve SIEMPRE 4 valores:
          video_logits : (B, num_classes)
          w            : (B, T, 1) o None
          seg_logits   : (B, T, num_classes) o None
          attn_logits  : (B, T, 1) o None  -> para regularización de entropía
        """
        cs = chunk_size if chunk_size is not None else self.chunk_size
        B, T, C, H, W = rgb.shape

        x_spat = rearrange(rgb, "b t c h w -> (b t) c h w")
        f_spat = self._encode_chunked(x_spat, self.spatial_enc, self.proj_spat, cs)
        f_spat = rearrange(f_spat, "(b t) d -> b t d", b=B, t=T)

        if "dual" in self.model_type or self.model_type == "full_model":
            x_mot = rearrange(motion, "b t c h w -> (b t) c h w")
            f_mot = self._encode_chunked(x_mot, self.motion_enc, self.proj_mot, cs)
            f_mot = rearrange(f_mot, "(b t) d -> b t d", b=B, t=T)

        if self.model_type == "cnn_rgb":
            return self.head(f_spat.mean(dim=1)), None, None, None

        elif self.model_type == "cnn_lstm":
            out, _ = self.lstm(f_spat)
            return self.head(out[:, -1, :]), None, None, None

        elif self.model_type == "dual_no_trans":
            f_cat = torch.cat([f_spat.mean(dim=1), f_mot.mean(dim=1)], dim=-1)
            return self.head(f_cat), None, None, None

        elif self.model_type == "dual_avg_pool":
            return self.head((f_spat + f_mot).mean(dim=1)), None, None, None

        elif self.model_type == "single_trans":
            f = self.transformer(f_spat + self.pos_emb)
            pooled, w, attn_logits = self._attention_pool(f)
            return self.head(pooled), w, self.head(f), attn_logits

        elif self.model_type == "full_model":
            f = self.transformer(f_spat + f_mot + self.pos_emb)
            pooled, w, attn_logits = self._attention_pool(f)
            seg_logits = self.head(f)                    # voto por segmento
            return self.head(pooled), w, seg_logits, attn_logits
