"""Inferencia compartida por evaluate y localize."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vad.data.common import read_json
from vad.data.feature_dataset import load_video_snippets
from vad.models.mil_head import SnippetMIL
from vad.temporal import merge_window_scores, smooth_scores, to_segments, window_starts

log = logging.getLogger("vad.inference")


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def feature_dims(cfg: dict) -> tuple[int, int]:
    idx = Path(cfg["features"]["dir"]) / "index.json"
    if not idx.exists():
        raise FileNotFoundError(f"No existe {idx}. Ejecuta: vad extract")
    meta = read_json(idx)
    return meta["dim_rgb"], meta["dim_mot"]


def build_model(cfg: dict, device: str) -> SnippetMIL:
    return SnippetMIL(*feature_dims(cfg), cfg, num_classes=len(cfg["preprocess"]["class_to_idx"])).to(device)


def load_model(cfg: dict, checkpoint: str | Path, device: str) -> tuple[SnippetMIL, dict]:
    """La arquitectura se toma del CHECKPOINT (no del YAML actual), para que un checkpoint
    entrenado con otra config se evalúe con la arquitectura con la que se entrenó."""
    import copy

    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"No existe {checkpoint}. Ejecuta primero: vad train")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ck_cfg = ckpt.get("cfg", {})
    ck_model = dict(ck_cfg.get("model", cfg["model"]))
    ck_model.setdefault("score_head", "transformer")          # checkpoints v0.4 no tenían la clave
    build_cfg = copy.deepcopy(cfg)
    build_cfg["model"] = {**cfg["model"], **ck_model}
    if "mil" in ck_cfg:
        build_cfg["mil"] = {**cfg["mil"], **{k: ck_cfg["mil"][k] for k in ("num_segments", "window")
                                                if k in ck_cfg["mil"]}}

    diffs = {k: (cfg["model"].get(k), v) for k, v in ck_model.items() if cfg["model"].get(k) != v}
    if diffs:
        log.warning("El checkpoint se entrenó con otra config de modelo; se usa la del checkpoint: %s",
                    {k: f"yaml={a} → checkpoint={b}" for k, (a, b) in diffs.items()})
    model = build_model(build_cfg, device)
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as e:
        raise RuntimeError(f"El checkpoint {checkpoint} no coincide con la arquitectura. "
                           f"¿Lo entrenaste con otra versión? Vuelve a correr `vad train`.\n{e}") from None
    model.eval()
    log.info("Checkpoint %s | época %s | val macro-F1 %.4f | score_head=%s", checkpoint, ckpt.get("epoch"),
             float(ckpt.get("val_f1", float("nan"))), ck_model["score_head"])
    return model, ckpt


@torch.no_grad()
def score_video(model: SnippetMIL, cfg: dict, video_id: str, device: str) -> dict:
    """
    - Clasificación del video: vista de 32 segmentos (igual que en entrenamiento).
    - Score por snippet: scorer local sobre todo el video (score_head=local) o ventanas deslizantes
      de W snippets con paso S promediadas (score_head=transformer).
    """
    v = load_video_snippets(cfg, video_id)
    mil = cfg["mil"]
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(device)

    seg = model(t(to_segments(v["rgb"], mil["num_segments"]))[None],
                t(to_segments(v["mot"], mil["num_segments"]))[None])
    probs = F.softmax(seg["video_logits"].float(), dim=-1)[0].cpu().numpy()

    n = len(v["rgb"])
    if getattr(model, "score_head", "transformer") == "local":
        # scorer local: mira cada snippet y sus vecinos → se aplica al video completo de una vez
        scores = model.snippet_scores(t(v["rgb"])[None], t(v["mot"])[None])[0].cpu().numpy()
        n_windows = 0
    else:
        # score desde el Transformer: ventanas deslizantes de W snippets con paso S, promediadas
        starts = window_starts(n, mil["window"], mil["stride"])
        L = min(mil["window"], n)
        rgb_w = torch.stack([t(v["rgb"][s:s + L]) for s in starts])
        mot_w = torch.stack([t(v["mot"][s:s + L]) for s in starts])
        win_scores = model(rgb_w, mot_w)["scores"].cpu().numpy()
        scores = merge_window_scores(n, mil["window"], starts, win_scores)
        n_windows = len(starts)
    raw = scores
    scores = smooth_scores(raw, int(cfg.get("localize", {}).get("smooth_snippets", 1)))
    return {"probs": probs, "pred": int(probs.argmax()), "scores": scores, "scores_raw": raw,
            "ranges": v["ranges"], "n_windows": n_windows}
