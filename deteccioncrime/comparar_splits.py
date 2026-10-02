import json

def ids(carpeta, split):
    with open(f"data/processed/{carpeta}/splits/{split}.jsonl", encoding="utf-8") as f:
        return {json.loads(l)["video_id"] for l in f if l.strip()}

for s in ("train", "val", "test"):
    a = ids("ucf_crime_3class", s)
    b = ids("ucf_crime_3class_320", s)
    if a == b:
        print(f"{s:5s} IGUALES ({len(a)} videos)")
    else:
        print(f"{s:5s} distintos: {len(a & b)} en común de {len(a)} | "
              f"solo 64x64: {sorted(a - b)[:5]} | solo 320: {sorted(b - a)[:5]}")