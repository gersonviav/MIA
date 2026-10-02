"""
Extracción de features (etapa 1, se corre UNA vez)
==================================================
Los backbones de v5 están congelados, así que su salida no cambia entre épocas.
Se calcula una sola vez para TODOS los frames de cada video (cobertura completa):

    frame_t (RGB)            → ConvNeXt-Small → rgb[t]  (768)
    |frame_t - frame_{t-1}|  → ConvNeXt-Tiny  → mot[t]  (768)

Salida: data/processed/features/<nombre>/<video_id>.npz con
    rgb [n_frames, D] float16 · mot [n_frames, D] float16 · frame_indices [n_frames]
y index.json con backbones, dimensiones y el hash del dataset procesado.

Rendimiento (para que la GPU no espere al disco):
  - Un hilo productor lee y redimensiona frames por adelantado (cola) mientras la GPU procesa.
  - Las imágenes viajan a la GPU como uint8; la diferencia, la conversión a float y la
    normalización se hacen en la GPU (antes se hacían en la CPU con numpy).
Se puede interrumpir: al reanudar salta los videos ya extraídos.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import timm
import torch

from vad.data.common import PROCESSED_VERSION_NAME, read_json, read_jsonl, write_json

log = logging.getLogger("vad.extract")

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SPLITS = ("train", "val", "test")
# Parámetros estándar de Farnebäck (documentación de OpenCV); se pueden cambiar con features.farneback_params
FARNEBACK_DEFAULTS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
_END = object()


def _read_rgb(path: Path, size: int) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_frames(pool: ThreadPoolExecutor, raw_dir: Path, rel_paths: list[str], size: int,
                fallback: np.ndarray | None = None) -> np.ndarray:
    """Lee frames (uint8 [N,H,W,3]); un frame ilegible se reemplaza por el anterior."""
    frames = list(pool.map(lambda p: _read_rgb(raw_dir / p, size), rel_paths))
    last = fallback if fallback is not None else next(
        (f for f in frames if f is not None), np.zeros((size, size, 3), np.uint8))
    out = []
    for f in frames:
        last = f if f is not None else last
        out.append(last)
    return np.stack(out)


def _producer(videos, raw_dir, size, chunk, workers, q, stop):
    """Lee los videos por bloques y los encola: (video, bloque uint8, prev uint8|None, es_último)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for v in videos:
            prev = None
            n = len(v["frames"])
            for i in range(0, n, chunk):
                if stop.is_set():
                    return
                fr = load_frames(pool, raw_dir, v["frames"][i:i + chunk], size, fallback=prev)
                q.put((v, fr, prev, i + chunk >= n))
                prev = fr[-1]
    q.put(_END)


def _model_norm(model, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Media/desv. con que se pre-entrenó cada backbone (ImageNet para ConvNeXt, otras para CLIP)."""
    try:
        dc = timm.data.resolve_model_data_config(model)
        mean, std = dc.get("mean", MEAN), dc.get("std", STD)
    except Exception:
        mean, std = MEAN, STD
    return (torch.tensor(mean, device=device).view(1, 3, 1, 1),
            torch.tensor(std, device=device).view(1, 3, 1, 1))


class FrameEncoder:
    """
    Stream de apariencia: frame RGB → spatial_backbone (ConvNeXt, CLIP, … cualquier modelo de timm).
    Stream de movimiento: imagen de movimiento → motion_backbone, donde la imagen es
        motion_input = "diff": |frame_t − frame_t−1|              (por defecto)
        motion_input = "raft":      flujo óptico RAFT (red neuronal, GPU)
        motion_input = "farneback": flujo óptico de Farnebäck (OpenCV, clásico, CPU, sin pesos)
    En ambos flujos la imagen codifica R = desplazamiento horizontal, G = vertical, B = magnitud
    (escala fija: ±flow_max_px píxeles → 0..255), así son comparables entre sí.
    """

    def __init__(self, fcfg: dict, device: str):
        self.device = device
        self.fp16 = device == "cuda"
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
        self.spatial = timm.create_model(fcfg["spatial_backbone"], pretrained=fcfg["pretrained"],
                                         num_classes=0).to(device).eval()
        self.motion = timm.create_model(fcfg["motion_backbone"], pretrained=fcfg["pretrained"],
                                        num_classes=0).to(device).eval()
        self.dims = (self.spatial.num_features, self.motion.num_features)
        self.bs = fcfg["batch_frames"]
        self.norm = {id(self.spatial): _model_norm(self.spatial, device),
                     id(self.motion): _model_norm(self.motion, device)}
        self.motion_input = fcfg.get("motion_input", "diff")
        self.raft = None
        if self.motion_input == "raft":
            from torchvision.models import optical_flow as of

            if fcfg["img_size"] < 128 or fcfg["img_size"] % 8:
                raise ValueError("RAFT necesita img_size ≥ 128 y múltiplo de 8 (usa 224).")

            name = fcfg.get("raft_model", "raft_large")
            weights = None
            if fcfg["pretrained"]:
                weights = (of.Raft_Large_Weights.DEFAULT if name == "raft_large"
                           else of.Raft_Small_Weights.DEFAULT)
            self.raft = getattr(of, name)(weights=weights).to(device).eval()
            self.raft_bs = fcfg.get("raft_batch", 32)
            self.flow_max = float(fcfg.get("flow_max_px", 20.0))
        elif self.motion_input == "farneback":
            self.flow_max = float(fcfg.get("flow_max_px", 20.0))
            self.fb_params = {**FARNEBACK_DEFAULTS, **(fcfg.get("farneback_params") or {})}
            self.fb_pool = ThreadPoolExecutor(max_workers=fcfg.get("farneback_workers", 8))
        elif self.motion_input != "diff":
            raise ValueError(f"motion_input desconocido: {self.motion_input} (usa diff, raft o farneback)")

    def _norm(self, model, x_u8: torch.Tensor) -> torch.Tensor:
        """uint8 [N,H,W,3] (en GPU) → float normalizado [N,3,H,W] según el backbone."""
        mean, std = self.norm[id(model)]
        x = x_u8.permute(0, 3, 1, 2).float().div_(255.0)
        return (x - mean) / std

    @torch.no_grad()
    def _run(self, model, x_u8: torch.Tensor) -> np.ndarray:
        outs = []
        for i in range(0, len(x_u8), self.bs):
            with torch.autocast(device_type="cuda" if self.fp16 else "cpu", enabled=self.fp16):
                outs.append(model(self._norm(model, x_u8[i:i + self.bs])).float())
        return torch.cat(outs).cpu().numpy().astype(np.float16)

    @torch.no_grad()
    def _flow_images(self, a_u8: torch.Tensor, b_u8: torch.Tensor) -> torch.Tensor:
        """Flujo RAFT de a→b (uint8 [N,H,W,3]) codificado como imagen uint8 [N,H,W,3]."""
        out = []
        for i in range(0, len(a_u8), self.raft_bs):
            a = a_u8[i:i + self.raft_bs].permute(0, 3, 1, 2).float().div(127.5).sub(1.0)   # [-1, 1]
            b = b_u8[i:i + self.raft_bs].permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
            flow = self.raft(a, b)[-1]                                                    # [n,2,H,W] px
            mag = flow.norm(dim=1, keepdim=True)
            img = torch.cat([flow / self.flow_max, mag / self.flow_max * 2 - 1], 1)     # ~[-1, 1]
            img = ((img.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
            out.append(img.permute(0, 2, 3, 1))
        return torch.cat(out)

    def _encode_flow_np(self, flow: np.ndarray) -> np.ndarray:
        """Flujo [H,W,2] en px → imagen uint8 [H,W,3] (misma codificación que RAFT)."""
        mag = np.linalg.norm(flow, axis=2, keepdims=True)
        img = np.concatenate([flow / self.flow_max, mag / self.flow_max * 2 - 1], axis=2)
        return ((np.clip(img, -1, 1) + 1) * 127.5).round().astype(np.uint8)

    def _farneback_images(self, seq: np.ndarray) -> np.ndarray:
        """Flujo de Farnebäck entre frames consecutivos de seq (uint8 [N,H,W,3]) → [N−1,H,W,3]."""
        gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in seq]

        def one(i):
            flow = cv2.calcOpticalFlowFarneback(gray[i], gray[i + 1], None, **self.fb_params)
            return self._encode_flow_np(flow)

        return np.stack(list(self.fb_pool.map(one, range(len(gray) - 1))))

    @torch.no_grad()
    def encode_chunk(self, frames: np.ndarray, prev: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Apariencia y movimiento de un bloque. prev = último frame del bloque anterior."""
        x = torch.from_numpy(frames).to(self.device, non_blocking=True)
        seq = x if prev is None else torch.cat([torch.from_numpy(prev[None]).to(self.device), x])
        if self.motion_input == "raft":
            m = self._flow_images(seq[:-1], seq[1:]) if len(seq) > 1 else torch.zeros_like(x)
        elif self.motion_input == "farneback":
            seq_np = frames if prev is None else np.concatenate([prev[None], frames])
            m = (torch.from_numpy(self._farneback_images(seq_np)).to(self.device)
                 if len(seq_np) > 1 else torch.zeros_like(x))
        else:
            m = (seq[1:].short() - seq[:-1].short()).abs().to(torch.uint8)
        if prev is None:  # primer frame del video: reutiliza el movimiento 0→1
            m = torch.cat([m[:1], m]) if len(m) else torch.zeros_like(x)
        return self._run(self.spatial, x), self._run(self.motion, m)


def run_extract(cfg: dict) -> dict:
    fcfg, paths = cfg["features"], cfg["paths"]
    raw_dir = Path(paths["raw_dir"])
    out_dir = Path(fcfg["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunk = max(fcfg.get("chunk_frames", 256), fcfg["batch_frames"])

    videos = []
    for split in SPLITS:
        p = Path(paths["processed_dir"]) / "splits" / f"{split}.jsonl"
        if not p.exists():
            log.error("No existe %s. Ejecuta primero: vad preprocess", p)
            raise SystemExit(1)
        videos += read_jsonl(p)
    pending = [v for v in videos if not (out_dir / f"{v['video_id']}.npz").exists()]
    total_frames = sum(v["n_frames"] for v in pending)

    log.info("=" * 70)
    log.info("EXTRACCIÓN DE FEATURES — apariencia: %s | movimiento: %s sobre %s | %s",
             fcfg["spatial_backbone"], fcfg["motion_backbone"],
             {"raft": "RAFT (" + fcfg.get("raft_model", "raft_large") + ")",
              "farneback": "flujo Farnebäck (OpenCV)"}.get(fcfg.get("motion_input", "diff"), "|Δframe|"), device)
    log.info("=" * 70)
    log.info("Videos: %d (pendientes %d, ya extraídos %d) | frames pendientes: %s | destino: %s",
             len(videos), len(pending), len(videos) - len(pending), f"{total_frames:,}", out_dir)
    log.info("batch_frames=%d | chunk_frames=%d | read_workers=%d", fcfg["batch_frames"], chunk, fcfg["read_workers"])
    if device == "cpu":
        log.warning("Sin GPU: la extracción será lenta.")

    enc = FrameEncoder(fcfg, device)
    pv = Path(paths["manifests_dir"]) / PROCESSED_VERSION_NAME
    index = {
        "spatial_backbone": fcfg["spatial_backbone"], "motion_backbone": fcfg["motion_backbone"],
        "pretrained": fcfg["pretrained"], "img_size": fcfg["img_size"],
        "dim_rgb": enc.dims[0], "dim_mot": enc.dims[1], "dtype": "float16",
        "motion_input": fcfg.get("motion_input", "diff"),
        "raft_model": fcfg.get("raft_model") if fcfg.get("motion_input") == "raft" else None,
        "farneback_params": ({**FARNEBACK_DEFAULTS, **(fcfg.get("farneback_params") or {})}
                             if fcfg.get("motion_input") == "farneback" else None),
        "flow_max_px": fcfg.get("flow_max_px") if fcfg.get("motion_input") in ("raft", "farneback") else None,
        "processed_dataset_sha256": read_json(pv).get("processed_dataset_sha256") if pv.exists() else None,
    }
    idx_path = out_dir / "index.json"
    if idx_path.exists():
        prev_idx = read_json(idx_path)
        for k in ("spatial_backbone", "motion_backbone", "motion_input", "img_size", "processed_dataset_sha256"):
            if prev_idx.get(k) != index[k]:
                log.warning("index.json existente difiere en '%s' (%s → %s). "
                            "Usa otro features.dir o borra la carpeta.", k, prev_idx.get(k), index[k])
    write_json(index, idx_path)

    q: queue.Queue = queue.Queue(maxsize=fcfg.get("prefetch_chunks", 4))
    stop = threading.Event()
    th = threading.Thread(target=_producer, daemon=True,
                          args=(pending, raw_dir, fcfg["img_size"], chunk, fcfg["read_workers"], q, stop))
    th.start()

    done, frames_done, t0, wait = 0, 0, time.time(), 0.0
    rgb_parts, mot_parts = [], []
    try:
        while True:
            tw = time.time()
            item = q.get()
            wait += time.time() - tw
            if item is _END:
                break
            v, fr, prev, last = item
            r, m = enc.encode_chunk(fr, prev)
            rgb_parts.append(r)
            mot_parts.append(m)
            frames_done += len(fr)
            if last:
                dst = out_dir / f"{v['video_id']}.npz"
                tmp = dst.with_suffix(".tmp.npz")
                np.savez(tmp, rgb=np.concatenate(rgb_parts), mot=np.concatenate(mot_parts),
                         frame_indices=np.asarray(v["frame_indices"], dtype=np.int64))
                tmp.replace(dst)
                rgb_parts, mot_parts = [], []
                done += 1
                if done % 10 == 0 or done == len(pending):
                    el = time.time() - t0
                    fps = frames_done / max(el, 1e-6)
                    eta = (total_frames - frames_done) / max(fps, 1e-6) / 60
                    log.info("  %d/%d videos | %.0f frames/s | GPU esperando al disco %.0f%% | "
                             "%.1f min | faltan ~%.0f min", done, len(pending), fps,
                             100 * wait / max(el, 1e-6), el / 60, eta)
    finally:
        stop.set()

    log.info("Extraídos: %d | dims rgb=%d mot=%d", done, *enc.dims)
    log.info("Siguiente paso: vad train")
    return {"extracted": done, "skipped": len(videos) - len(pending)}
