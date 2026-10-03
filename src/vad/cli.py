"""
CLI del proyecto.

    vad frames     (opcional) videos .mp4 originales → frames, mismo formato que data/raw
    vad ingest     [--download-date YYYY-MM-DD] [--skip-hash]
    vad preprocess
    vad pipeline   (= ingest + preprocess)
    vad extract    features por frame con los backbones congelados (una sola vez)
    vad train      cabeza MIL sobre features
    vad evaluate   [--checkpoint ...] [--split test]      nivel video + AUC nivel frame
    vad localize   [--checkpoint ...] [--video-id X] [--plot-all]   intervalos en segundos

También funciona como:  python -m vad <subcomando>
Cualquier parámetro se puede sobrescribir sin editar el YAML:
    vad train --set train.seed=7 --set paths.checkpoints_dir=artifacts/seed7
"""

from __future__ import annotations

import argparse
import time

from vad.config import apply_overrides, load_config
from vad.logging_utils import log_run_context, setup_logging


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vad", description="Detección de anomalías UCF-Crime (3 clases)")
    p.add_argument("--config", default="configs/default.yaml")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR",
                        help="Sobrescribe la config, ej. --set model.score_head=transformer (repetible)")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("ingest", "pipeline"):
        s = sub.add_parser(name, parents=[common])
        s.add_argument("--download-date", default=None, help="Fecha de descarga del dataset (YYYY-MM-DD)")
        s.add_argument("--skip-hash", action="store_true", help="No calcular SHA-256 por archivo (solo pruebas)")
    sub.add_parser("frames", parents=[common])
    sub.add_parser("preprocess", parents=[common])
    sub.add_parser("extract", parents=[common])
    sub.add_parser("train", parents=[common])
    for name in ("evaluate", "localize"):
        s = sub.add_parser(name, parents=[common])
        s.add_argument("--checkpoint", default=None,
                       help="Por defecto: <paths.checkpoints_dir>/best_model.pth del config usado")
        s.add_argument("--split", default="test", choices=["train", "val", "test"])
        if name == "localize":
            s.add_argument("--video-id", default=None)
            s.add_argument("--plot-all", action="store_true", help="Gráfico por cada video anómalo")
    return p.parse_args()


def main() -> None:
    args = _args()
    cfg = load_config(args.config)
    applied = apply_overrides(cfg, args.set)
    logger = setup_logging(args.cmd, cfg["paths"]["logs_dir"])
    log_run_context(logger, cfg)
    if applied:
        logger.info("Overrides: %s", applied)
    if getattr(args, "checkpoint", "x") is None:
        from pathlib import Path

        args.checkpoint = str(Path(cfg["paths"]["checkpoints_dir"]) / "best_model.pth")
    t0 = time.time()

    try:
        if args.cmd == "frames":
            from vad.data.frames import run_frames

            run_frames(cfg)
        if args.cmd in ("ingest", "pipeline"):
            from vad.data.ingest import run_ingest

            run_ingest(cfg, download_date=args.download_date, skip_hash=args.skip_hash)
        if args.cmd in ("preprocess", "pipeline"):
            from vad.data.preprocess import run_preprocess

            run_preprocess(cfg)
        if args.cmd == "extract":
            from vad.features import run_extract

            run_extract(cfg)
        if args.cmd == "train":
            from vad.train import run_train

            run_train(cfg)
        if args.cmd == "evaluate":
            from vad.evaluate import run_evaluate

            run_evaluate(cfg, args.checkpoint, args.split)
        if args.cmd == "localize":
            from vad.localize import run_localize

            run_localize(cfg, args.checkpoint, args.split, args.video_id, args.plot_all)
    except SystemExit:
        logger.error("Proceso '%s' abortado.", args.cmd)
        raise
    except Exception:
        logger.exception("Error no controlado en '%s'", args.cmd)
        raise
    logger.info("'%s' completado en %.1f s", args.cmd, time.time() - t0)


if __name__ == "__main__":
    main()