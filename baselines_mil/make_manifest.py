"""
Convierte tu split (train.jsonl / val.jsonl / test.jsonl) en el manifiesto CSV que usan los baselines.
Así los baselines usan EXACTAMENTE los mismos videos y particiones que tu modelo.

  python baselines_mil/make_manifest.py --splits_dir data/processed/ucf_crime_3class_320_v2/splits ^
      --out data/splits/manifest_3class_v2.csv --video_root <carpeta_mp4>

--video_root es opcional: si lo pasas, verifica que exista el .mp4 original de cada video
(I3D necesita frames consecutivos; tus frames guardados están cada 10 frames, por eso se lee el .mp4).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

KEEP = ["video_id", "class_name", "ucf_category", "split", "source_split", "label", "n_frames", "brightness"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video_root", default=None)
    args = ap.parse_args()

    rows = []
    for name in ("train", "val", "test"):
        f = Path(args.splits_dir) / f"{name}.jsonl"
        if not f.exists():
            raise SystemExit(f"No existe {f}")
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    d.setdefault("split", name)
                    rows.append({k: d.get(k) for k in KEEP})
    df = pd.DataFrame(rows).rename(columns={"class_name": "label_name", "label": "label_idx",
                                            "n_frames": "n_frames_guardados"})
    df = df.rename(columns={"label_name": "label", "ucf_category": "subclass"})

    # Controles de fuga / consistencia
    dup = df[df.video_id.duplicated(keep=False)]
    assert dup.empty, f"Videos en más de un split:\n{dup[['video_id', 'split']].to_string(index=False)}"
    bad_test = df[(df.split == "test") & (df.source_split.str.lower() != "test")]
    bad_trv = df[(df.split != "test") & (df.source_split.str.lower() != "train")]
    if len(bad_test) or len(bad_trv):
        print(f"[WARN] test con videos fuera del test oficial: {len(bad_test)} | "
              f"train/val con videos del test oficial: {len(bad_trv)}")
    else:
        print("OK: test = test oficial UCF-Crime; train/val salen solo del train oficial.")

    if args.video_root:
        stems = {p.stem for p in Path(args.video_root).rglob("*") if p.suffix.lower() in (".mp4", ".avi", ".mkv")}
        miss = sorted(set(df.video_id) - stems)
        print(f"Videos originales encontrados: {len(df) - len(miss)}/{len(df)}")
        if miss:
            print("[WARN] faltan .mp4 de:", miss[:15], "..." if len(miss) > 15 else "")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(pd.crosstab(df.label, df.split, margins=True))
    print(pd.crosstab(df.subclass, df.split, margins=True))
    print(f"\nGuardado: {args.out} ({len(df)} videos)")


if __name__ == "__main__":
    main()
