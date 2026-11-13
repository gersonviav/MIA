"""AUC/AP a nivel de frame para CUALQUIER modelo a partir de sus scores por segmento.

  python frame_auc.py --manifest data/manifest_3class.csv \
      --segscores runs/propuesto/seed0/segscores_test.npz \
      --annotations Temporal_Anomaly_Annotation.txt
"""
import argparse
import json

from data import load_manifest
from metrics import frame_level_auc, load_temporal_annotations, read_segment_scores

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--segscores", required=True)
ap.add_argument("--annotations", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--col_video", default="video_id")
ap.add_argument("--col_label", default="label")
ap.add_argument("--col_split", default="split")
a = ap.parse_args()

df = load_manifest(a.manifest, a.col_video, a.col_label, a.col_split)
df = df[df.split == a.split]
seg, nfr, kind = read_segment_scores(a.segscores)
res = frame_level_auc(seg, nfr, kind, load_temporal_annotations(a.annotations),
                      dict(zip(df.video_id, df.is_anom)))
print(json.dumps(res, indent=1))
