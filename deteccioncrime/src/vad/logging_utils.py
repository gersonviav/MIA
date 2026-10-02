"""Logging a consola + archivo (logs/<paso>_<timestamp>.log)."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from vad import __version__

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(step: str, logs_dir: str | Path = "logs", level: int = logging.INFO) -> logging.Logger:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{step}_{ts}.log"

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    root.addHandler(fh)
    root.addHandler(ch)

    logger = logging.getLogger(f"vad.{step}")
    logger.info("Log de ejecución: %s", log_file)
    return logger


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def log_run_context(logger: logging.Logger, cfg: dict | None = None) -> dict:
    ctx = {
        "vad_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "config_file": (cfg or {}).get("_config_path"),
    }
    try:
        import torch

        ctx["torch"] = torch.__version__
        ctx["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            ctx["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    logger.info("Contexto: %s", json.dumps(ctx, ensure_ascii=False))
    return ctx
