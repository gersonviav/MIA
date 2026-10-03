"""
Videos originales (.mp4) → frames (paso 0, opcional)
====================================================
Reemplaza la versión de frames de 64×64 por frames a la resolución original (normalmente 320×240),
con EXACTAMENTE el mismo formato que espera el resto del pipeline:

    <out_dir>/<Train|Test>/<Categoria>/<VideoID>_<frame>.jpg      frame = nº de frame ORIGINAL

- Se guarda 1 de cada `every_n` frames (10 → ~3 fps, igual que la versión de 64×64),
  así los tiempos y las anotaciones temporales siguen calzando.
- El split Train/Test sale de los archivos oficiales Anomaly_Train.txt / Anomaly_Test.txt.
- Solo se procesan las categorías del mapeo a 3 clases (ahorra tiempo y disco).
- Reanudable: salta los videos ya terminados (marca `.done` por video).
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

log = logging.getLogger("vad.frames")

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}


def category_of(video_id: str) -> str:
    """'Abuse028_x264' → 'Abuse' · 'Normal_Videos_003_x264' → 'NormalVideos'."""
    if video_id.lower().startswith("normal"):
        return "NormalVideos"
    m = re.match(r"[A-Za-z]+", video_id)
    return m.group(0) if m else "Unknown"


def read_split_file(path: Path) -> set[str]:
    """Lista oficial (una ruta por línea, ej. 'Abuse/Abuse001_x264.mp4') → set de video_ids."""
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.add(Path(line.replace("\\", "/")).stem)
    return ids


def _extract_one(video: str, dst_dir: str, video_id: str, every_n: int, max_side: int,
                 quality: int) -> tuple[str, int, float, str]:
    dst = Path(dst_dir)
    done = dst / f".{video_id}.done"
    if done.exists():
        return video_id, -1, 0.0, "skip"
    dst.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return video_id, 0, 0.0, "no se pudo abrir"
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    idx, saved = 0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % every_n == 0:
            ok, frame = cap.retrieve()
            if ok:
                h, w = frame.shape[:2]
                if max_side and max(h, w) > max_side:
                    s = max_side / max(h, w)
                    frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dst / f"{video_id}_{idx}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                saved += 1
        idx += 1
    cap.release()
    done.write_text(str(saved))
    return video_id, saved, fps, "ok"


def run_frames(cfg: dict) -> dict:
    vc = cfg["videos"]
    src = Path(vc["dir"])
    out = Path(vc["out_dir"])
    if not src.exists():
        log.error("No existe %s. Descarga los videos de UCF-Crime ahí (ver docs/DATASET.md).", src)
        raise SystemExit(1)

    splits = {}
    for name, key in (("Train", "train_list"), ("Test", "test_list")):
        p = Path(vc[key])
        if not p.exists():
            log.error("Falta la lista oficial %s (%s). Viene con el dataset (carpeta de splits).", name, p)
            raise SystemExit(1)
        splits[name] = read_split_file(p)
        log.info("Lista oficial %s: %d videos", name, len(splits[name]))

    wanted = {c for cats in cfg["preprocess"]["class_mapping"].values() for c in cats}
    videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    jobs, unknown, skipped_cat = [], 0, 0
    for v in videos:
        vid = v.stem
        split = "Train" if vid in splits["Train"] else "Test" if vid in splits["Test"] else None
        if split is None:
            unknown += 1
            continue
        cat = category_of(vid)
        if cat not in wanted:
            skipped_cat += 1
            continue
        jobs.append((str(v), str(out / split / cat), vid))

    log.info("=" * 70)
    log.info("VIDEOS → FRAMES | 1 de cada %d frames | lado máx. %s px | JPEG q=%d",
             vc["every_n"], vc["max_side"] or "original", vc["jpeg_quality"])
    log.info("=" * 70)
    log.info("Videos encontrados: %d | a procesar: %d | fuera del mapeo 3 clases: %d | sin split oficial: %d",
             len(videos), len(jobs), skipped_cat, unknown)
    if unknown:
        log.warning("%d videos no aparecen en Anomaly_Train/Test.txt y se ignoran", unknown)

    n_ok = n_skip = n_err = total = 0
    odd_fps = []
    with ProcessPoolExecutor(max_workers=vc["workers"]) as ex:
        futs = [ex.submit(_extract_one, v, d, vid, vc["every_n"], vc["max_side"], vc["jpeg_quality"])
                for v, d, vid in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            vid, saved, fps, status = f.result()
            if status == "skip":
                n_skip += 1
            elif status == "ok":
                n_ok += 1
                total += saved
                if fps and abs(fps - cfg["dataset"]["fps"]) > 1:
                    odd_fps.append((vid, round(fps, 2)))
            else:
                n_err += 1
                log.error("  %s: %s", vid, status)
            if i % 25 == 0 or i == len(jobs):
                log.info("  %d/%d videos | %d frames guardados", i, len(jobs), total)

    if odd_fps:
        log.warning("%d videos con fps distinto de %s (los segundos serán aproximados): %s",
                    len(odd_fps), cfg["dataset"]["fps"], odd_fps[:10])
    log.info("Listo: %d procesados | %d ya existían | %d con error | %d frames nuevos", n_ok, n_skip, n_err, total)
    log.info("Siguiente paso: apunta paths.raw_dir a %s y corre `vad pipeline`", out)
    return {"ok": n_ok, "skipped": n_skip, "errors": n_err, "frames": total}