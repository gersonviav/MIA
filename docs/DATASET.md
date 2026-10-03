# Dataset — UCF-Crime (frames pre-extraídos)

## 1. Fuente

| Campo | Valor |
|---|---|
| Nombre | UCF-Crime — *Real-world Anomaly Detection in Surveillance Videos* |
| Autores | W. Sultani, C. Chen, M. Shah (CVPR 2018) |
| Página oficial | https://www.crcv.ucf.edu/projects/real-world/ |
| Variante usada | Frames pre-extraídos (JPG/PNG), no los `.mp4` originales |
| URL exacta de la variante | **⚠️ COMPLETAR** (ej. mirror/Kaggle desde donde se descargaron los frames) |
| Licencia / uso | Solo investigación académica, según términos de los autores |

**Dataset original:** 1 900 videos de vigilancia sin recortar (~128 h), 13 categorías de anomalía + videos normales.
Split oficial: 1 610 videos de entrenamiento (800 normales + 810 anómalos) y 290 de test
(150 normales + 140 anómalos). Solo los videos de **test** tienen anotación temporal del evento.

## 2. Versión registrada (fecha de descarga, tamaño, hash)

El bloque siguiente lo escribe `vad ingest` a partir de `data/manifests/raw_dataset_version.json`.
Para registrar la fecha de descarga: `vad ingest --download-date YYYY-MM-DD`.

<!-- AUTOGEN:START -->
_Generado automáticamente por `vad ingest` el 2026-10-01T20:32:32+00:00. No editar a mano._

| Campo | Valor |
|---|---|
| Fuente | https://www.crcv.ucf.edu/projects/real-world/ |
| Fecha de descarga | 2026-10-01 |
| Fecha de ingesta | 2026-10-01T20:32:32+00:00 |
| Ruta | `data/raw/ucf_crime_320` |
| Videos | 1,650 |
| Frames (archivos) | 1,306,972 |
| Tamaño en disco | 40.0 GB (42,938,383,931 bytes) |
| Resolución (muestra) | 320x240x3 |
| Modo de hash | sha256 por archivo |
| **SHA-256 del dataset** | `b5624813ff432dbb39fb6c7050aab133e07f564af938239a1d44a39c1a4e7b0d` |
| Anotaciones temporales | sí — `3b9542413f2ed9e9…` |

| Split / Categoría | Videos | Frames | Tamaño |
|---|---:|---:|---:|
| Test/Abuse | 2 | 297 | 9.0 MB |
| Test/Arson | 9 | 2,793 | 57.2 MB |
| Test/Assault | 3 | 2,657 | 61.0 MB |
| Test/Burglary | 13 | 7,657 | 186.0 MB |
| Test/Explosion | 21 | 6,510 | 143.3 MB |
| Test/Fighting | 5 | 1,231 | 33.7 MB |
| Test/NormalVideos | 150 | 64,952 | 2.0 GB |
| Test/Robbery | 5 | 835 | 24.5 MB |
| Test/Shoplifting | 21 | 7,623 | 288.2 MB |
| Test/Stealing | 5 | 1,984 | 57.6 MB |
| Test/Vandalism | 5 | 1,111 | 28.0 MB |
| Train/Abuse | 48 | 19,076 | 439.0 MB |
| Train/Arson | 41 | 24,421 | 596.0 MB |
| Train/Assault | 47 | 10,360 | 268.2 MB |
| Train/Burglary | 87 | 39,504 | 1.1 GB |
| Train/Explosion | 29 | 18,753 | 492.8 MB |
| Train/Fighting | 45 | 24,684 | 707.9 MB |
| Train/NormalVideos | 800 | 947,768 | 30.0 GB |
| Train/Robbery | 145 | 41,493 | 1.3 GB |
| Train/Shoplifting | 29 | 24,835 | 856.7 MB |
| Train/Stealing | 95 | 44,802 | 1.1 GB |
| Train/Vandalism | 45 | 13,626 | 343.9 MB |
<!-- AUTOGEN:END -->

## 3. Estructura esperada en `data/raw/`

```
data/raw/
├── ucf_crime/
│   ├── Train/<Categoria>/<VideoID>_<frame>.jpg|png     ej. Abuse028_x264_120.png
│   └── Test/<Categoria>/<VideoID>_<frame>.jpg|png
└── Temporal_Anomaly_Annotation_for_Testing_Videos.txt  (opcional, para `vad localize`)
```

Categorías reconocidas: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, NormalVideos,
RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism.

> **Supuesto importante:** el sufijo numérico del nombre (`_120`) es el **número de frame en el video
> original**. Con `fps = 30` (config) eso permite convertir a segundos (`120 / 30 = 4.0 s`).
> Si tu variante numera los frames de otra forma, ajusta `dataset.fps` o la conversión en `localize.py`.

## 4. Variables principales

### Crudo (`data/manifests/raw_manifest.csv.gz`, una fila por frame)

| Variable | Tipo | Descripción |
|---|---|---|
| `split` | str | `Train` / `Test` (split oficial UCF-Crime) |
| `category` | str | Categoría original (14 posibles) |
| `video_id` | str | Identificador del video, ej. `Fighting003_x264` |
| `frame_idx` | int | Nº de frame original (del sufijo del archivo) |
| `rel_path` | str | Ruta relativa a `data/raw/ucf_crime/` |
| `bytes` | int | Tamaño del archivo |
| `sha256` | str | Hash del archivo |
| *(imagen)* | uint8 H×W×3 | Frame RGB; resolución medida en la sección 2 |

### Procesado (`data/processed/ucf_crime_3class/splits/{train,val,test}.jsonl`, una fila por video)

| Variable | Tipo | Descripción |
|---|---|---|
| `video_id` | str | Identificador del video |
| `ucf_category` | str | Categoría original |
| `class_name` | str | Macro-clase: `NormalVideos`, `Fighting`, `Vandalism` |
| `label` | int | 0 = NormalVideos, 1 = Fighting, 2 = Vandalism |
| `split` | str | `train` / `val` / `test` |
| `source_split` | str | Split UCF de origen (`Train`/`Test`) |
| `n_frames` | int | Frames disponibles del video |
| `brightness` | float | Brillo medio (escala de grises, 0–255) en 5 frames muestreados |
| `frame_indices` | list[int] | Nº de frame original de cada archivo |
| `frames` | list[str] | Rutas relativas de los frames, en orden temporal |

### Features (`data/processed/features/<nombre>/<video_id>.npz`, generado por `vad extract`)

| Variable | Forma | Descripción |
|---|---|---|
| `rgb` | [n_frames, 768] float16 | ConvNeXt-Small (ImageNet, congelado) sobre cada frame RGB 224×224 |
| `mot` | [n_frames, 768] float16 | ConvNeXt-Tiny sobre \|frame_t − frame_{t−1}\| entre frames extraídos consecutivos |
| `frame_indices` | [n_frames] int | Nº de frame original, para convertir a segundos |

`index.json` registra los backbones, las dimensiones y el hash del dataset procesado del que salieron las features.

### Anotación temporal (solo Test; necesaria para medir la localización)

`<video>.mp4 <Categoria> <inicio1> <fin1> <inicio2> <fin2>` — en frames; `-1` = sin segundo intervalo.

## 5. Transformaciones aplicadas (`vad preprocess`)

| Paso | Regla |
|---|---|
| Remapeo | Fighting ← Assault, Abuse, Fighting · Vandalism ← Robbery, Arson, Burglary, Explosion, Shoplifting, Stealing, Vandalism · NormalVideos ← NormalVideos. Arrest, RoadAccidents y Shooting **no se usan**. |
| Filtro de longitud | Se excluyen videos con < 8 frames |
| Filtro de calidad | Se excluyen videos con brillo medio < 30 |
| Tope por clase | Solo en Train: NormalVideos 200, Fighting 150, Vandalism 150 (semilla 42) |
| Validación | 20 % estratificado **desde Train** |
| Test | Split oficial Test completo (tras filtros); **no se usa para seleccionar modelos** |
| Control de fuga | Se verifica que ningún `video_id` aparezca en dos splits |

Los videos excluidos y el motivo quedan en `data/processed/ucf_crime_3class/excluded_videos.csv`.

## 6. Control de versiones del dataset

- `data/manifests/raw_dataset_version.json` → hash SHA-256 global del crudo (hash de la lista ordenada
  `ruta + sha256` de cada archivo) + hash por categoría. **Se versiona en git.**
- `data/manifests/processed_dataset_version.json` → hash de los archivos de split + hash del crudo del que
  provienen (linaje). **Se versiona en git.**
- `raw_manifest.csv.gz` → inventario completo por archivo (pesado; no va a git, se regenera con `vad ingest`).
- Cada checkpoint guarda el `processed_dataset_sha256` con el que se entrenó.
- Si `vad ingest` detecta un hash distinto al registrado, lo avisa en el log y guarda la versión anterior.

Para versionar también los datos binarios se puede añadir DVC encima de esta estructura
(`dvc add data/raw/ucf_crime`), sin cambiar el código.

## 7. Consideraciones éticas

Videos de vigilancia reales con personas identificables. Uso exclusivamente académico; no redistribuir frames
ni publicar imágenes de personas sin anonimizar.
