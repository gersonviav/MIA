import glob, json, statistics as st

M = {"AUC binario": lambda m: m["video_level_binary"]["auc_1_menos_p_normal"],
     "macro-F1":    lambda m: m["video_level"]["macro_f1"],
     "AUC frame":   lambda m: m["frame_level"]["auc"],
     "AUC anómalos": lambda m: m["frame_level"]["auc_anomalous_only"]}

for cfg in ("default", "hr320", "hr320_clip", "hr320_farneback", "hr320_clip_farneback"):
    ms = [json.load(open(f)) for f in sorted(glob.glob(f"artifacts/seeds/{cfg}/s*/reports/test/metrics.json"))]
    print(f"{cfg}  ({len(ms)} semillas)")
    for name, get in M.items():
        v = [get(m) for m in ms]
        print(f"  {name:13s} {st.mean(v):.3f} ± {st.stdev(v):.3f}" if len(v) > 1 else f"  {name:13s} {v}")