# Detección de Anomalías en Video con Deep Learning (UCF-Crime, 3 clases)

Este proyecto implementa un sistema de **clasificación de videos de vigilancia** para la detección de anomalías utilizando **Deep Learning** sobre el dataset **UCF-Crime**.

El modelo combina información **espacial** (apariencia) y **temporal** (movimiento) mediante una arquitectura *two-stream* basada en **ConvNeXt** y un **Transformer temporal con mecanismo de atención**, clasificando cada video en una de las siguientes tres categorías:

| Clase | Categorías originales de UCF-Crime |
|--------|------------------------------------|
| **NormalVideos** | NormalVideos |
| **Fighting** | Assault, Abuse, Fighting |
| **Vandalism** | Robbery, Arson, Burglary, Explosion, Shoplifting, Stealing, Vandalism |

Las categorías originales fueron agrupadas para reducir el desbalance de clases y simplificar el problema de clasificación, manteniendo eventos con características visuales similares.

---

# Arquitectura

El modelo (`AnomalyDetector3Class` en `train.py`) sigue una arquitectura **Two-Stream** con modelado temporal mediante Transformer.

```
                  Video
                    │
      ┌─────────────┴─────────────┐
      │                           │
 RGB Frames                Frame Difference
      │                           │
ConvNeXt-Small             ConvNeXt-Tiny
      │                           │
Spatial Features          Motion Features
      └─────────────┬─────────────┘
                    │
          Concatenación de Features
                    │
          Temporal Transformer
                    │
           Attention Pooling
                    │
             MLP Classifier
                    │
      ┌─────────────┼─────────────┐
      │             │             │
   Normal       Fighting     Vandalism
```

### Componentes

### Spatial Stream

- Backbone **ConvNeXt-Small** preentrenado (`timm`).
- Procesa los frames RGB para extraer características espaciales.

### Motion Stream

- Backbone **ConvNeXt-Tiny**.
- Recibe como entrada la diferencia absoluta entre frames consecutivos (*frame difference*), utilizada como aproximación del movimiento.

### Temporal Transformer

- `TransformerEncoder`
- 4 capas
- 8 cabezas de atención
- Positional Embeddings aprendibles

Su objetivo es modelar las dependencias temporales entre los **16 segmentos** muestreados de cada video.

### Attention Pooling

En lugar de promediar todos los frames, una capa de atención aprende automáticamente cuáles son los más relevantes para la clasificación final.

### Clasificador

MLP compuesto por:

```
Linear
↓
ReLU
↓
Dropout
↓
Linear
↓
3 clases
```

---

# Estrategia de entrenamiento

El entrenamiento utiliza **Differential Learning Rate**:

| Componente | Learning Rate |
|------------|--------------:|
| Backbones ConvNeXt | 1e-5 |
| Transformer + Attention + Clasificador | 1e-3 |

Además:

- Los backbones permanecen congelados durante las primeras **3 épocas** (*warmup*).
- Posteriormente se descongelan para realizar **fine-tuning end-to-end**.

---

# Estructura del proyecto

```
.
├── filter_dataset_3class.py
├── train.py
├── eval_final.py
├── README.md
└── .gitignore
```

## Archivos principales

| Archivo | Descripción |
|----------|-------------|
| `filter_dataset_3class.py` | Filtra y reorganiza UCF-Crime a 3 clases |
| `train.py` | Dataset, modelo, entrenamiento y evaluación |
| `eval_final.py` | Evaluación final (ROC, AUC y matriz de confusión) |

---

## Directorios generados localmente

Estos directorios **no forman parte del repositorio** y están excluidos mediante `.gitignore`.

```
ucf_crime/
ucf_crime_3class/
checkpoints_3class/
eval_outputs/
```

Descripción:

- **ucf_crime/** → Dataset original con frames extraídos.
- **ucf_crime_3class/** → Dataset filtrado y balanceado.
- **checkpoints_3class/** → Checkpoints del entrenamiento.
- **eval_outputs/** → Resultados de la evaluación.

---

# Requisitos

- Python 3.10 o superior
- PyTorch
- GPU con soporte CUDA (recomendado)

El entrenamiento fue probado en una **NVIDIA RTX 3060 de 12 GB** utilizando precisión mixta (**FP16**).

También es posible entrenar en CPU, aunque con tiempos considerablemente mayores.

---

# Instalación

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

Instalar PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Instalar el resto de dependencias:

```bash
pip install timm einops opencv-python numpy scikit-learn matplotlib tqdm
```

> Ajusta la versión de CUDA según tu instalación de PyTorch.

---

# Dataset

Este proyecto utiliza el dataset público **UCF-Crime**.

https://www.crcv.ucf.edu/projects/real-world/

Se espera que los videos hayan sido previamente convertidos en frames con la siguiente estructura:

```
ucf_crime/

├── Train/
│   ├── Abuse/
│   ├── Assault/
│   ├── Burglary/
│   ├── Explosion/
│   ├── Fighting/
│   ├── NormalVideos/
│   ├── Robbery/
│   ├── Shoplifting/
│   ├── Stealing/
│   └── Vandalism/
│
└── Test/
    └── (misma estructura)
```

Si solo se dispone de los videos originales, los frames pueden extraerse mediante **FFmpeg** u **OpenCV**.

---

# Reproducción de resultados

## 1. Filtrar el dataset

Modo de prueba:

```bash
python filter_dataset_3class.py --dry-run
```

Generar el dataset:

```bash
python filter_dataset_3class.py
```

Este proceso:

- Agrupa los frames por video.
- Elimina videos con menos de 8 frames.
- Descarta videos con brillo promedio inferior a 30.
- Convierte las 10 categorías originales en 3 clases.
- Balancea el número de videos por clase.
- Mantiene el split Train/Test original.
- Genera `metadata.json` y los archivos de partición.

---

## 2. Entrenamiento

```bash
python train.py --mode train
```

Configuración por defecto:

- 50 épocas
- Warmup de 3 épocas
- Evaluación cada 5 épocas
- Guardado automático del mejor modelo

Checkpoints generados:

```
checkpoints_3class/

best_model.pth
last_model.pth
```

Todos los hiperparámetros se encuentran centralizados en el diccionario `CFG` dentro de `train.py`.

---

## 3. Evaluación rápida

```bash
python train.py --mode eval \
    --checkpoint checkpoints_3class/best_model.pth
```

Se muestran:

- Accuracy
- Classification Report
- Macro AUC (One-vs-Rest)

---

## 4. Evaluación completa

```bash
python eval_final.py \
    --checkpoint checkpoints_3class/best_model.pth
```

Se generan los siguientes archivos:

```
eval_outputs/

confusion_matrix.png
roc_curves.png
```

Además se imprimen:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- AUC por clase
- Macro AUC

---

# Métricas de evaluación

El rendimiento del modelo se evalúa mediante:

- Accuracy
- Precision
- Recall
- F1-score
- Macro AUC (One-vs-Rest)
- Matriz de confusión
- Curvas ROC

---

# Limitaciones

- Requiere que los videos hayan sido convertidos previamente a frames.
- La clasificación se realiza a nivel de video y no de segmentos temporales.
- El rendimiento depende de la calidad de los frames y del balance del dataset.

---

# Notas

- El dataset y los checkpoints no se incluyen en el repositorio debido a su tamaño.
- El entrenamiento utiliza un **WeightedRandomSampler** para compensar el desbalance entre clases.
- El entrenamiento en precisión mixta (**FP16**) puede activarse o desactivarse mediante `CFG["use_fp16"]`.

---

# Licencia

Este proyecto se distribuye únicamente con fines académicos y de investigación.

El dataset **UCF-Crime** pertenece a sus autores originales y debe utilizarse respetando sus términos de uso.