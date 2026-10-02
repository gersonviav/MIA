"""
Análisis por categoría original de UCF-Crime a partir de summary.csv de `vad localize`.

Uso (desde la raíz del repo):
    python analisis_categorias.py                                   # 320×240
    python analisis_categorias.py artifacts/reports/localization_test/summary.csv   # 64×64
"""
import sys

import pandas as pd

ruta = sys.argv[1] if len(sys.argv) > 1 else "artifacts/reports_320/localization_test/summary.csv"
s = pd.read_csv(ruta)
s = s[s.label != "NormalVideos"].copy()
s["cat"] = s.video_id.str.extract(r"^([A-Za-z]+)")
s["loc_ok"] = s.tiou >= 0.5
s["clase_ok"] = s.label == s.pred
s["detectado"] = s.pred != "NormalVideos"

t = s.groupby("cat").agg(videos=("video_id", "count"),
                         tiou_medio=("tiou", "mean"),
                         pct_bien_localizado=("loc_ok", "mean"),
                         pct_detectado=("detectado", "mean"),
                         acierto_clase=("clase_ok", "mean"),
                         score_max_medio=("peak_score", "mean")).round(2).sort_values("tiou_medio")
print(f"Fuente: {ruta}\n")
print(t.to_string())
print(f"\nTotal anómalos: {len(s)} | tIoU medio {s.tiou.mean():.3f} | bien localizados {s.loc_ok.mean():.1%}")
t.to_csv(ruta.replace("summary.csv", "por_categoria.csv"))
print(f"Guardado: {ruta.replace('summary.csv', 'por_categoria.csv')}")