# legacy/

Scripts originales conservados solo como referencia. **No usar**: los reemplaza el paquete `src/vad/`.

| Original | Reemplazo |
|---|---|
| `filter_dataset_3class.py` | `vad ingest` + `vad preprocess` |
| `train.py` (v2.1, end-to-end) | reemplazado por el esquema v5 + MIL |
| `train_v5.py` / `models_v5.py` | `vad extract` (backbones congelados) + `vad train` (`src/vad/models/mil_head.py`) |
| `eval_final.py` | `vad evaluate` (+ AUC por frame) |

Los checkpoints de `train_v5.py` **no cargan** directamente en `SnippetMIL`: los nombres de capas coinciden
(`proj_spat`, `proj_mot`, `pos_emb`, `transformer`, `attn`, `head`), pero `pos_emb` tenía 16 posiciones y el
entrenamiento era con 16 frames sueltos. Hay que reentrenar con `vad train`.
