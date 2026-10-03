"""
Entrenamiento 3 clases — v2.1 con Attention Pooling y Diagnóstico
=================================================================
Cambios respecto a v2:
  - Reemplazado el promedio simple (.mean(dim=1)) por una capa de Attention Pooling.
  - La importancia de cada frame se calcula dinámicamente mediante Softmax temporal.
  - Los parámetros de la atención se entrenan con la tasa de la cabeza (lr_head = 1e-3).
"""

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision.transforms as T
import timm
from einops import rearrange
from sklearn.metrics import classification_report, roc_auc_score

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "data_root":        "ucf_crime_3class",
    "checkpoint_dir":   "checkpoints_3class",
    "class_to_idx":     {"NormalVideos": 0, "Fighting": 1, "Vandalism": 2},
    "num_classes":      3,

    "num_segments":     16,
    "img_size":         224,

    "spatial_backbone": "convnext_small",
    "d_model":          512,
    "nhead":            8,
    "num_layers":       4,

    # Differential LR: backbone mucho más lento que la cabeza
    "lr_backbone":      1e-5,   # ← bajo para no destruir pesos preentrenados
    "lr_head":          1e-3,   # ← más alto para el Transformer, Atención y clasificador

    "weight_decay":     1e-4,
    "label_smoothing":  0.1,    
    "warmup_epochs":    3,      # épocas sin actualizar backbone
    "batch_size":       4,
    "epochs":           50,
    "use_fp16":         True,
    "grad_accum":       2,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class UCFCrime3ClassDataset(Dataset):
    def __init__(self, data_root, split_file, num_segments=16,
                 img_size=224, augment=False):
        self.num_segments = num_segments
        self.augment = augment

        aug = [T.RandomHorizontalFlip(),
               T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
               T.RandomGrayscale(p=0.05)] if augment else []
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            *aug,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

        split_path = Path(data_root) / "Annotations" / split_file
        self.samples = []
        for line in split_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split(" ", 3)
            if len(parts) < 4:
                continue
            video_id   = parts[0]
            label      = int(parts[1])
            frames_str = parts[3]
            frame_paths = [Path(p) for p in frames_str.split("|") if p]
            if frame_paths:
                self.samples.append((video_id, label, frame_paths))

        labels = [s[1] for s in self.samples]
        from collections import Counter
        print(f"  {split_file}: {len(self.samples)} videos | "
              f"dist: {dict(sorted(Counter(labels).items()))}")

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, frame_paths):
        n       = len(frame_paths)
        indices = np.linspace(0, n - 1, self.num_segments, dtype=int)
        frames, last = [], None
        for idx in indices:
            img = cv2.imread(str(frame_paths[idx]))
            if img is not None:
                last = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(last if last is not None
                          else np.zeros((224, 224, 3), dtype=np.uint8))
        return frames

    def __getitem__(self, idx):
        video_id, label, frame_paths = self.samples[idx]
        frames = self._load_frames(frame_paths)

        rgb_list, motion_list = [], []
        for i, f in enumerate(frames):
            rgb_list.append(self.transform(f))
            diff = cv2.absdiff(frames[i], frames[i-1]) if i > 0 else np.zeros_like(f)
            motion_list.append(self.transform(diff))

        return (torch.stack(rgb_list), torch.stack(motion_list),
                label, video_id)


def make_weighted_sampler(dataset):
    labels      = [s[1] for s in dataset.samples]
    class_count = np.bincount(labels, minlength=CFG["num_classes"]).astype(float)
    w_per_class = 1.0 / np.maximum(class_count, 1)
    weights     = torch.tensor([w_per_class[l] for l in labels], dtype=torch.float)
    print(f"  Sampler pesos: { {i: round(w_per_class[i],4) for i in range(len(w_per_class))} }")
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULOS DE RED
# ─────────────────────────────────────────────────────────────────────────────

class SpatialStream(nn.Module):
    def __init__(self, backbone, d_model):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.num_features, d_model),
            nn.LayerNorm(d_model), nn.GELU(),
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        return rearrange(self.proj(self.encoder(x)), "(b t) d -> b t d", b=B, t=T)

    def backbone_params(self):
        return list(self.encoder.parameters())

    def head_params(self):
        return list(self.proj.parameters())


class MotionStream(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoder = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.num_features, d_model),
            nn.LayerNorm(d_model), nn.GELU(),
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        return rearrange(self.proj(self.encoder(x)), "(b t) d -> b t d", b=B, t=T)

    def backbone_params(self):
        return list(self.encoder.parameters())

    def head_params(self):
        return list(self.proj.parameters())


class TemporalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, max_len=64):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        return self.transformer(x + self.pos_emb(pos))


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [B, T, d_model]
        scores = self.attn_layer(x)  # [B, T, 1]
        weights = F.softmax(scores, dim=1)  # [B, T, 1]
        pooled = torch.sum(x * weights, dim=1)  # [B, d_model]
        return pooled


# ─────────────────────────────────────────────────────────────────────────────
# MODELO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetector3Class(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.spatial  = SpatialStream(cfg["spatial_backbone"], d)
        self.motion   = MotionStream(d)
        self.temporal = TemporalTransformer(d, cfg["nhead"], cfg["num_layers"])
        self.pool     = AttentionPooling(d)  # Reemplaza al promedio crudo
        self.classifier = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, cfg["num_classes"]),
        )

    def forward(self, rgb, motion):
        f = self.spatial(rgb) + self.motion(motion)
        f = self.temporal(f)
        f = self.pool(f)  # Ponderación dinámica basada en relevancia temporal
        return self.classifier(f)

    def get_param_groups(self, lr_backbone, lr_head, weight_decay):
        """Grupos separados: backbone lento, head (incluyendo atención) rápido."""
        backbone_params = (self.spatial.backbone_params() +
                           self.motion.backbone_params())
        head_params     = (self.spatial.head_params() +
                           self.motion.head_params() +
                           list(self.temporal.parameters()) +
                           list(self.pool.parameters()) +  # Entrenado con lr_head
                           list(self.classifier.parameters()))
        return [
            {"params": backbone_params, "lr": lr_backbone,
             "weight_decay": weight_decay, "name": "backbone"},
            {"params": head_params,     "lr": lr_head,
             "weight_decay": weight_decay, "name": "head"},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL LOOP
# ─────────────────────────────────────────────────────────────────────────────

def set_backbone_grad(model, requires_grad: bool):
    """Congela/descongela los backbones de imagen y movimiento."""
    for p in model.spatial.encoder.parameters():
        p.requires_grad = requires_grad
    for p in model.motion.encoder.parameters():
        p.requires_grad = requires_grad


def train_epoch(model, loader, optimizer, scaler, device, use_fp16,
                grad_accum, criterion, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pred_counts = np.zeros(CFG["num_classes"], dtype=int)
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="  Train", leave=False, ncols=90)
    for step, (rgb, motion, labels, _) in enumerate(pbar):
        rgb    = rgb.to(device, non_blocking=True)
        motion = motion.to(device, non_blocking=True)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_fp16):
            logits = model(rgb, motion)
            loss   = criterion(logits, labels) / grad_accum

        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        for p in preds.cpu().numpy():
            pred_counts[p] += 1
        pbar.set_postfix(loss=f"{total_loss/(step+1):.3f}",
                         acc=f"{correct/total:.2%}")

    print(f"  Pred dist train → "
          f"Normal:{pred_counts[0]} Fighting:{pred_counts[1]} Vandalism:{pred_counts[2]}")
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device, use_fp16, class_names):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for rgb, motion, labels, _ in tqdm(loader, desc="  Eval", leave=False, ncols=90):
            rgb    = rgb.to(device, non_blocking=True)
            motion = motion.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_fp16):
                logits = model(rgb, motion)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n  Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, digits=3,
                                zero_division=0))
    
    unique_classes = np.unique(all_labels)
    if len(unique_classes) >= 2:
        try:
            auc = roc_auc_score(all_labels, np.array(all_probs),
                                multi_class="ovr", average="macro")
            print(f"  AUC macro (OvR): {auc:.4f}")
        except ValueError as e:
            print(f"  AUC no calculable: {e}")
            auc = acc
    else:
        print(f"  AUC no calculable: solo una clase en test ({unique_classes})")
        auc = acc
    return acc, auc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device   = CFG["device"]
    use_fp16 = CFG["use_fp16"] and device == "cuda"
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

    class_names = sorted(CFG["class_to_idx"], key=CFG["class_to_idx"].get)
    print(f"\nDispositivo : {device}")
    print(f"fp16        : {use_fp16}")
    print(f"LR backbone : {CFG['lr_backbone']}  |  LR head: {CFG['lr_head']}")
    print(f"Clases      : {class_names}\n")

    print("Cargando datasets...")
    train_ds = UCFCrime3ClassDataset(
        CFG["data_root"], "train_split.txt",
        CFG["num_segments"], CFG["img_size"], augment=True,
    )
    test_ds = UCFCrime3ClassDataset(
        CFG["data_root"], "test_split.txt",
        CFG["num_segments"], CFG["img_size"], augment=False,
    )

    sampler      = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              sampler=sampler, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=0, pin_memory=True)

    model = AnomalyDetector3Class(CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}\n")

    param_groups = model.get_param_groups(
        CFG["lr_backbone"], CFG["lr_head"], CFG["weight_decay"]
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"] - CFG["warmup_epochs"], eta_min=1e-7,
    )
    scaler    = torch.amp.GradScaler("cuda", enabled=use_fp16)
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

    if args.mode == "eval":
        if args.checkpoint is None:
            raise ValueError("Debes especificar la ruta de un --checkpoint para evaluar.")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Checkpoint cargado: época {ckpt['epoch']}, AUC={ckpt['auc']:.4f}")
        evaluate(model, test_loader, device, use_fp16, class_names)
        return

    best_auc = 0.0
    print(f"Iniciando entrenamiento — {CFG['epochs']} épocas")
    print(f"Warmup: backbone congelado las primeras {CFG['warmup_epochs']} épocas\n")

    for epoch in range(1, CFG["epochs"] + 1):

        # Control del congelamiento / descongelamiento del Backbone
        if epoch <= CFG["warmup_epochs"]:
            set_backbone_grad(model, False)
            if epoch == 1:
                print("  [Warmup] Backbone congelado — solo entrenando Transformer, Atención y Clasificador")
        elif epoch == CFG["warmup_epochs"] + 1:
            set_backbone_grad(model, True)
            print("  [Fine-tune] Backbone descongelado — entrenamiento de extremo a extremo")

        print(f"Época {epoch}/{CFG['epochs']}")
        loss, acc = train_epoch(model, train_loader, optimizer, scaler,
                                device, use_fp16, CFG["grad_accum"],
                                criterion, epoch)

        if epoch > CFG["warmup_epochs"]:
            scheduler.step()

        # Monitor de Learning Rates por grupo
        lr_bb  = optimizer.param_groups[0]["lr"]
        lr_hd  = optimizer.param_groups[1]["lr"]
        print(f"  Loss: {loss:.4f}  |  Acc: {acc:.2%}  |  "
              f"LR backbone={lr_bb:.2e}  head={lr_hd:.2e}")

        if epoch % 5 == 0 or epoch == CFG["epochs"]:
            val_acc, auc = evaluate(model, test_loader, device, use_fp16, class_names)

            # Guardar último estado de entrenamiento
            torch.save({
                "epoch": epoch, "auc": auc, "acc": val_acc,
                "model_state": model.state_dict(), "cfg": CFG,
            }, os.path.join(CFG["checkpoint_dir"], "last_model.pth"))

            # Guardar el modelo óptimo (Mejor AUC)
            if auc > best_auc:
                best_auc = auc
                torch.save({
                    "epoch": epoch, "auc": auc, "acc": val_acc,
                    "model_state": model.state_dict(), "cfg": CFG,
                }, os.path.join(CFG["checkpoint_dir"], "best_model.pth"))
                print(f"  → Mejor modelo guardado (AUC={best_auc:.4f}, Acc={val_acc:.4f})")

    print(f"\nEntrenamiento completo. Mejor AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["train", "eval"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    main(args)


# .\mi_entorno\Scripts\activate  