"""
Extracción de características I3D (RGB, Inception-v1, Kinetics-400) para los baselines MIL
(Sultani, RTFM, UR-DMU), usando EXACTAMENTE los mismos videos y la misma resolución que
el modelo propuesto (frames 320x240 -> recortes de 224x224).

Protocolo (igual al de RTFM/UR-DMU, salvo la resolución de origen):
  * snippets de 16 frames consecutivos, sin solapamiento, a la tasa nativa del video;
  * el último snippet se completa repitiendo el último frame;
  * 10-crop (5 recortes 224x224 + sus espejos horizontales) o 1-crop central;
  * salida por video: array float16 de forma [n_crops, T, 1024] + metadatos (n_frames, fps).

Uso típico (lee los .mp4 originales y los reescala a 320x240, como en tu extracción de frames):
  python extract_i3d.py --manifest data/manifest_3class.csv --video_root D:/UCF_Crime/videos \
      --weights weights/model_rgb.pth --out_dir features/i3d_320x240_10crop --crops 10

Alternativa (si tus frames 320x240 están a fps nativo):
  python extract_i3d.py --manifest ... --frames_root D:/UCF_frames_320x240 ...
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from data import load_manifest
from models.i3dpt import I3D

CROP = 224


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def frame_stream_from_video(path, size_wh):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def gen():
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if (fr.shape[1], fr.shape[0]) != size_wh:
                fr = cv2.resize(fr, size_wh, interpolation=cv2.INTER_AREA)
            yield fr[:, :, ::-1]  # BGR -> RGB
        cap.release()

    return gen(), fps


def frame_stream_from_dir(path, size_wh):
    import cv2
    files = sorted([f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                   key=natural_key)
    if not files:
        raise IOError(f"Sin frames en {path}")

    def gen():
        for f in files:
            fr = cv2.imread(os.path.join(path, f))
            if (fr.shape[1], fr.shape[0]) != size_wh:
                fr = cv2.resize(fr, size_wh, interpolation=cv2.INTER_AREA)
            yield fr[:, :, ::-1]

    return gen(), None


def chunk_stream(frames, chunk=16):
    """Agrupa frames en snippets de `chunk`; rellena el último repitiendo el último frame."""
    buf, n = [], 0
    for fr in frames:
        buf.append(fr)
        n += 1
        if len(buf) == chunk:
            yield np.stack(buf), n
            buf = []
    if buf:
        buf += [buf[-1]] * (chunk - len(buf))
        yield np.stack(buf), n


def crop_offsets(h, w, ncrops):
    """Coordenadas (y, x) de los recortes 224x224 sobre un frame h x w."""
    cy, cx = (h - CROP) // 2, (w - CROP) // 2
    if ncrops == 1:
        return [(cy, cx)], False
    five = [(0, 0), (0, w - CROP), (cy, cx), (h - CROP, 0), (h - CROP, w - CROP)]
    return five, True  # 5 recortes + 5 espejos = 10


def make_crops(batch, ncrops):
    """batch: [B,16,H,W,3] uint8 -> lista de ncrops tensores [B,3,16,224,224] en [-1,1]."""
    _, _, h, w, _ = batch.shape
    offs, flip = crop_offsets(h, w, ncrops)
    views = [batch[:, :, y:y + CROP, x:x + CROP] for y, x in offs]
    if flip:
        views += [v[:, :, :, ::-1] for v in views]
    out = []
    for v in views:
        t = torch.from_numpy(np.ascontiguousarray(v)).float()
        t = t * (2.0 / 255.0) - 1.0
        out.append(t.permute(0, 4, 1, 2, 3))  # B,C,T,H,W
    return out


@torch.no_grad()
def extract_video(net, frames, ncrops, batch_size, device):
    feats = [[] for _ in range(ncrops)]
    pending, n_frames = [], 0

    def flush():
        b = np.stack(pending)
        for i, x in enumerate(make_crops(b, ncrops)):
            f, _ = net(x.to(device), feature_layer=5)
            feats[i].append(f[:, :, 0, 0, 0].float().cpu().numpy())

    for clip, n in chunk_stream(frames):
        pending.append(clip)
        n_frames = n
        if len(pending) == batch_size:
            flush()
            pending = []
    if pending:
        flush()
    arr = np.stack([np.concatenate(f, 0) for f in feats])  # [ncrops, T, 1024]
    return arr.astype(np.float16), n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--video_root", default=None, help="Carpeta con los .mp4 originales (busca recursivo)")
    ap.add_argument("--frames_root", default=None, help="Carpeta con subcarpetas de frames por video")
    ap.add_argument("--weights", required=True, help="model_rgb.pth (repo UR-DMU, feature_extract/)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--crops", type=int, choices=[1, 10], default=10)
    ap.add_argument("--batch_size", type=int, default=16, help="snippets por forward")
    ap.add_argument("--col_video", default="video_id")
    ap.add_argument("--col_label", default="label")
    ap.add_argument("--col_split", default="split")
    args = ap.parse_args()
    assert (args.video_root is None) != (args.frames_root is None), "Usa --video_root O --frames_root"
    assert args.height >= CROP and args.width >= CROP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = I3D(400, modality="rgb", dropout_prob=0, name="inception")
    net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    net.eval().to(device)

    df = load_manifest(args.manifest, args.col_video, args.col_label, args.col_split)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    index = {}
    if args.video_root:
        for p in Path(args.video_root).rglob("*"):
            if p.suffix.lower() in (".mp4", ".avi", ".mkv"):
                index[p.stem] = p

    size = (args.width, args.height)
    for i, vid in enumerate(df["video_id"]):
        dst = out / f"{vid}.npy"
        if dst.exists() and vid in meta:
            continue
        if args.video_root:
            if vid not in index:
                print(f"[WARN] no encontrado: {vid}")
                continue
            frames, fps = frame_stream_from_video(index[vid], size)
        else:
            frames, fps = frame_stream_from_dir(Path(args.frames_root) / vid, size)
        arr, n_frames = extract_video(net, frames, args.crops, args.batch_size, device)
        np.save(dst, arr)
        meta[vid] = {"n_frames": int(n_frames), "fps": fps, "T": int(arr.shape[1]),
                     "crops": args.crops, "size": [args.width, args.height]}
        if i % 20 == 0:
            meta_path.write_text(json.dumps(meta, indent=1))
        print(f"[{i + 1}/{len(df)}] {vid}: frames={n_frames} -> {arr.shape}")
    meta_path.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
