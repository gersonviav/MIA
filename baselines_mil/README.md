# Baselines MIL adaptados a 3 clases (Normal / Aggression / Theft)

Reimplementación y adaptación de **Sultani et al. (CVPR 2018)**, **RTFM (ICCV 2021)** y **UR-DMU (AAAI 2023)**
para compararlos con el modelo propuesto (dual-stream ConvNeXt + Transformer + Attention Pooling) bajo el
**mismo protocolo**: mismos videos, mismo split, misma resolución (frames 320×240 → 224×224), mismas métricas.

## Qué se cambia respecto a los papers (y nada más)

| | Sultani | RTFM | UR-DMU |
|---|---|---|---|
| Arquitectura y pérdida original | se conserva | se conserva | se conserva |
| Hiperparámetros | repo oficial | repo oficial | repo oficial |
| Entrada | I3D-1024 (antes C3D-4096) | I3D-1024 (antes 2048, se parametriza) | I3D-1024 (igual) |
| **Añadido** | cabeza 3 clases sobre top-1 | cabeza 3 clases sobre top-3 por magnitud | cabeza 3 clases sobre top-(T/16+1) |

Pérdida total: `L = L_original + 1.0 · CE_ponderada(cabeza_3clases)`. La salida binaria original se mantiene
y se usa para el AUC; la clase predicha sale de la cabeza de 3 clases.

## Pasos

```bash
pip install -r requirements.txt   # instala torch con CUDA según tu GPU

# 0) Pesos I3D: copia feature_extract/model_rgb.pth de https://github.com/henrryzh1/UR-DMU
#    Anotaciones temporales: Temporal_Anomaly_Annotation.txt de https://github.com/WaqasSultani/AnomalyDetectionCVPR2018

# 1) Características I3D de TUS videos (mismo manifiesto/split que tu modelo)
python extract_i3d.py --manifest data/manifest_3class.csv --video_root D:/UCF_Crime/videos ^
    --weights weights/model_rgb.pth --out_dir features/i3d_320x240_10crop --crops 10

# 2) Entrenar cada baseline con 5 semillas (selección por F1-macro en VAL)
for %s in (0 1 2 3 4) do python train_baseline.py --model sultani --manifest data/manifest_3class.csv --feat_dir features/i3d_320x240_10crop --out_dir runs/sultani --seed %s --annotations Temporal_Anomaly_Annotation.txt
#   idem con --model rtfm y --model urdmu

# 3) Exportar las predicciones de TU modelo en el mismo formato (ver abajo) a runs/propuesto/seed*/

# 4) Comparación estadística en test
python compare.py --ref Propuesto --run "Propuesto=runs/propuesto/seed*/preds_test.csv" ^
    --run "Sultani=runs/sultani/seed*/preds_test.csv" --run "RTFM=runs/rtfm/seed*/preds_test.csv" ^
    --run "UR-DMU=runs/urdmu/seed*/preds_test.csv" --out results/comparacion
```

Si tu CSV de split usa otros nombres de columna: `--col_video video --col_label clase --col_split particion`.
La columna de clase puede traer la macro-clase o la subclase UCF (Abuse, Robbery, …): se mapea sola.

## Exportar tu modelo (añadir al final de `eval_final.py`)

```python
import sys; sys.path.append("ruta/baselines_vad")
from metrics import write_predictions, write_segment_scores
# probs: [N,3] softmax (orden Normal, Aggression, Theft); y_true: [N] enteros 0/1/2
write_predictions(f"runs/propuesto/seed{seed}/preds_test.csv", video_ids, "test", y_true, probs,
                  anomaly_score=1 - probs[:, 0])
# Opcional, AUC por frame: score por segmento (p.ej. 1 - softmax(seg_logits)[:, 0], o alpha_i)
write_segment_scores(f"runs/propuesto/seed{seed}/segscores_test.npz", seg_scores_dict,
                     n_frames_dict, kind="uniform")   # L=16 segmentos uniformes
```
Luego: `python frame_auc.py --manifest ... --segscores runs/propuesto/seed0/segscores_test.npz --annotations ...`

## Salidas por corrida (`runs/<modelo>/seed<k>/`)
`best.pt`, `metrics.json` (val y test: F1 por clase, F1-macro, F1-macro-anom, F1-anomalía, AUC/AP video,
AUC/AP frame, matriz de confusión), `preds_val.csv`, `preds_test.csv`, `segscores_*.npz/.json`, `train_log.json`.

## Notas
- En Windows deja `num_workers` en 0 (aquí no se usa DataLoader); el extractor procesa en streaming.
- `FeatureStore` cachea en RAM los segmentos re-muestreados en float16 (UR-DMU con 200 segmentos y 10 crops
  ≈ 2 GB para ~400 videos de train). Si falta RAM, usa `--crops 1` en la extracción.
- Las cifras publicadas (AUC 75.41 / 84.30 / 86.97) son sobre los 290 videos de test y 13 clases: solo son contexto,
  no comparables directamente con tu tarea de 3 clases.
- Código de RTFM y UR-DMU adaptado de sus repositorios oficiales; `models/i3dpt.py` copiado de UR-DMU (MIT).
