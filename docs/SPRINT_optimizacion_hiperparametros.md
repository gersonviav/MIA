# Sprint — Optimización de hiperparámetros

Notebook: `notebooks/05_optimizacion_hiperparametros.ipynb` · Config base: `configs/hr320_clip_v2.yaml` ·
Fecha: 2026-11-05

## Diseño
- Se optimiza solo la cabeza MIL (Transformer + attention pooling) sobre features congeladas (CLIP ViT-B/16 + ConvNeXt-T sobre |Δframe|).
- Objetivo: **macro-F1 en validación** (criterio de selección de `vad train`). El test no se usa durante la búsqueda.
- Optuna TPE vs búsqueda aleatoria, 50 trials cada uno, poda con MedianPruner. La config actual entra como trial 0.
- Fijos: λ y arquitectura del scorer de localización (val no tiene anotación temporal), `mil.*`, `snippet.*`.
- Robustez: top-3 + default con semillas [42, 43, 44, 45, 46]; t-test pareado vs default.

## Búsqueda
| estudio | trials | completos | podados | mejor val F1 | trial del mejor | mejora vs default | tiempo total (min) |
|---|---|---|---|---|---|---|---|
| tpe | 50 | 23 | 27 | 0.8222 | 22 | 0.0575 | 5.8000 |
| random | 50 | 21 | 29 | 0.7949 | 12 | 0.0303 | 5.3000 |

![convergencia](../artifacts/hpo/hr320_clip_v2/figs/convergencia.png)

## Importancia de hiperparámetros (fANOVA, TPE)
| index | importancia |
|---|---|
| num_layers | 0.473 |
| lr | 0.108 |
| dropout | 0.103 |
| d_model | 0.093 |
| attn_temp | 0.057 |
| lambda_ent | 0.057 |
| weight_decay | 0.031 |
| lambda_smooth | 0.029 |
| batch_size | 0.019 |
| lambda_seg | 0.019 |
| nhead | 0.010 |

![importancia](../artifacts/hpo/hr320_clip_v2/figs/importancia_hiperparametros.png)
![slices](../artifacts/hpo/hr320_clip_v2/figs/slice_plots.png)
![localización](../artifacts/hpo/hr320_clip_v2/figs/chequeo_localizacion.png)

## Robustez (5 semillas)
| candidato | F1 media | F1 std | sep(max) media | score anóm. media | Δ F1 vs default | p (t pareado) | p (Wilcoxon) |
|---|---|---|---|---|---|---|---|
| default | 0.7625 | 0.0141 | 0.6933 | 0.6701 | 0.0000 | – | – |
| TPE#22 | 0.7711 | 0.0425 | 0.6244 | 0.6548 | 0.0086 | 0.7105 | 0.8125 |
| TPE#40 | 0.7797 | 0.0272 | 0.7267 | 0.6262 | 0.0171 | 0.2453 | 0.3125 |
| TPE#23 | 0.7826 | 0.0232 | 0.6897 | 0.6309 | 0.0201 | 0.1828 | 0.1875 |

![robustez](../artifacts/hpo/hr320_clip_v2/figs/robustez_semillas.png)

**Ganadora: TPE#23.** La diferencia con la config actual no es significativa con estas semillas: la config original ya estaba cerca del óptimo de este espacio.

## Cambios de configuración
| index | default | ganadora |
|---|---|---|
| lr | 0.0003 | 8.049e-05 |
| weight_decay | 0.01 | 0.006962 |
| lambda_ent | 0.3 | 0.7158 |
| lambda_seg | 0.5 | 0.4936 |
| lambda_smooth | 0.1 | 0.2891 |
| d_model | 512 | 256 |
| num_layers | 2 | 4 |
| dropout | 0.3 | 0.2647 |
| attn_temp | 0.5 | 0.5203 |

Config resultante: `configs/hr320_clip_v2_hpo.yaml`

## Test (una sola evaluación, config ganadora)
```
18:04:55 | INFO    | Checkpoint artifacts\checkpoints_320_clip_v2_hpo\best_model.pth | época 20 | val macro-F1 0.8109 | score_head=local
18:05:00 | INFO    | Video  | acc 0.8818 | macro-F1 0.7323 | macro AUC 0.9427
18:05:00 | INFO    |   Normal        P 0.929  R 0.953  F1 0.941
18:05:00 | INFO    |   Aggression    P 0.500  R 0.500  F1 0.500
18:05:00 | INFO    |   Theft         P 0.795  R 0.721  F1 0.756
18:05:00 | INFO    | Binario (normal vs anómalo) | acc 0.9113 | recall anomalía 0.792 | especificidad 0.953 | AUC(1-P normal) 0.9321 | AUC(max score) 0.9223
18:05:00 | INFO    | Frame  | AUC 0.9059 | AP 0.4109 | AUC solo anómalos 0.6993 | 870,440 frames
18:05:03 | INFO    | Checkpoint artifacts\checkpoints_320_clip_v2_hpo\best_model.pth | época 20 | val macro-F1 0.8109 | score_head=local
18:05:12 | INFO    | tIoU medio 0.2574631849876485 | videos con tIoU≥0.5 0.22641509433962265 | normales con falsa alarma 0.08
```

## Reproducir
```
jupyter nbconvert --to notebook --execute notebooks/05_optimizacion_hiperparametros.ipynb
```
El estudio queda en `artifacts/hpo/hr320_clip_v2/optuna.db` (se retoma si se interrumpe).
