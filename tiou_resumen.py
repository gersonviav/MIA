"""Resume el tIoU de cada semilla (igual que la sección 6d del notebook 03).

  python tiou_resumen.py                 # cnn_v2
  python tiou_resumen.py cnn_v2 clip_v2  # varios modelos
"""
import sys
from pathlib import Path

import pandas as pd

runs = sys.argv[1:] or ["cnn_v2"]
rows = []
for run in runs:
    for sdir in sorted(Path("artifacts/seeds", run).glob("s[0-9]*")):
        f = next(iter(sdir.rglob("localization_test/summary.csv")), None)
        if f is None:
            print(f"[{run} {sdir.name}] sin summary.csv")
            continue
        d = pd.read_csv(f)
        normal = d["label"].astype(str).str.lower().str.startswith("normal")
        an = d[~normal & d["gt_intervals"].notna()]
        t = an["tiou"].fillna(0)
        rows.append({"modelo": run, "semilla": sdir.name, "videos anómalos": len(an),
                     "tIoU medio": t.mean(), "tIoU>=0.1": (t >= .1).mean(), "tIoU>=0.3": (t >= .3).mean(),
                     "tIoU>=0.5": (t >= .5).mean(),
                     "falsa alarma normales": d.loc[normal, "detected_intervals"].notna().mean()})

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
print("\nMedia ± d.e. entre semillas:")
print(df.drop(columns=["semilla", "videos anómalos"]).groupby("modelo").agg(["mean", "std"]).round(3).T.to_string())
