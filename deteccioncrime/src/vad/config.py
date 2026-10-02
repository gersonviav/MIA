"""Carga de configuración YAML y utilidades comunes."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración '{path}'. "
            "Ejecuta los comandos desde la raíz del repositorio o usa --config."
        )
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def apply_overrides(cfg: dict[str, Any], items: list[str]) -> dict[str, Any]:
    """Aplica 'a.b.c=valor' (valor parseado como YAML: números, bool, listas)."""
    applied = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set espera CLAVE=VALOR, recibió '{it}'")
        key, raw = it.split("=", 1)
        node = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            if k not in node or not isinstance(node[k], dict):
                raise SystemExit(f"--set: la sección '{k}' no existe en la config")
            node = node[k]
        if parts[-1] not in node:
            raise SystemExit(f"--set: la clave '{key}' no existe en la config")
        node[parts[-1]] = yaml.safe_load(raw)
        applied[key] = node[parts[-1]]
    return applied


def snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    """Copia serializable de la configuración (para guardar en logs/checkpoints)."""
    return copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def class_names(cfg: dict[str, Any]) -> list[str]:
    c2i = cfg["preprocess"]["class_to_idx"]
    return sorted(c2i, key=c2i.get)
