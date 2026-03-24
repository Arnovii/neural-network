# Distributed Neural Network Training – ImageNet CNN+MLP

Entrenamiento distribuido de redes neuronales (CNN + MLP) sobre ImageNet usando arquitectura Parameter Server + Workers.

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [¿Por Qué Este Proyecto?](#por-qué-este-proyecto)
- [Arquitectura](#arquitectura)
- [Modos de Datos](#modos-de-datos)
- [Requisitos e Instalación](#requisitos-e-instalación)
- [Uso Rápido](#uso-rápido)
- [Ejemplos Completos](#ejemplos-completos)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Sistema de Persistencia de Modelos](#sistema-de-persistencia-de-modelos)
- [Troubleshooting](#troubleshooting)

---

## Descripción General

Este proyecto implementa un sistema distribuido de entrenamiento de redes neuronales que:

1. **Extrae características** de imágenes ImageNet usando una CNN (convolutional neural network)
2. **Entrena un clasificador MLP** (multilayer perceptron) sobre esas características
3. **Distribuye el aprendizaje** entre múltiples Workers usando un Parameter Server central

El sistema es transparente respecto al origen de los datos:

- **Modo LOCAL**: Lee imágenes de disco (~150 GB requeridos)
- **Modo STREAMING**: Descarga bajo demanda desde HuggingFace (sin descargar el dataset completo)

### ¿Por Qué Este Proyecto?

Entrena modelos CNN+MLP grandes sin necesidad de GPU con memoria masiva. El Parameter Server promedia gradientes de múltiples Workers, reduciendo el costo computacional y permitiendo entrenar modelos más complejos.

```
neural-network/
├── Data/
│   ├── ImageNet/            # Dataset ImageNet (~150 GB)
│   │   ├── train/           # 1.28M imágenes de entrenamiento
│   │   └── val/             # 50k imágenes de validación
│   └── feature_cache/       # Caché de features extraídos y pesos CNN
├── Distributed/             # Núcleo del sistema distribuido
│   ├── parameter_server.py  # Clase ParameterServer (lógica TCP + entrenamiento)
│   ├── worker_node.py       # Clase WorkerNode (extracción CNN + MLP forward/backward)
│   └── protocol.py          # Serialización de mensajes Pickle sobre TCP
├── Model/                   # Lógica central: CNN + MLP
│   ├── cnn_extractor.py     # CNN PyTorch (features cachés, pesos ImageNet/propios)
│   └── mlp.py               # MLP NumPy (pesos distribuidos, gradientes serializables)
├── Utils/
│   ├── feature_scaler.py    # StandardScaler para normalización de features CNN
│   ├── imagenet_loader.py   # Cargador lazy de ImageNet (DataLoader PyTorch)
│   └── results_exporter.py  # Exportación de resultados a JSON
├── Exports/                 # Resultados de entrenamiento (JSON por timestamp)
├── Docker/                  # Contenedores para Workers
│   ├── Dockerfile.worker    # Imagen Docker del Worker
│   ├── run_workers.ps1      # Script PowerShell para lanzar N Workers en Docker
│   └── .dockerignore
├── ps_terminal.py           # Parameter Server (interfaz de terminal)
├── ps_gui.py                # Parameter Server (interfaz gráfica Tkinter)
├── worker.py                # Worker Node
├── pyproject.toml
└── requirements.txt
```

---

## Instalación

**Con pip:**
```bash
pip install -r requirements.txt
```

**Con uv:**
```bash
uv sync
```

Dependencias principales: `numpy`, `torch`, `torchvision`, `matplotlib`.

### Preparación del Dataset: ImageNet

**ImageNet es un dataset grande (~150 GB) y requiere preparación manual:**

1. Descargar ImageNet desde [image-net.org](https://image-net.org/) (requiere registro).
2. Extraer los archivos en la estructura esperada:
   ```
   Data/ImageNet/
   ├── train/
   │   ├── n01440764/  (synset ID)
   │   │   ├── img_001.JPEG
   │   │   └── ...
   │   └── ... (1000 synsets)
   └── val/
       ├── n01440764/
       │   ├── img_001.JPEG
       │   └── ...
       └── ... (1000 synsets)
   ```

**Alternativa:** si tienes el dataset descargado en otra ubicación, especificar con:
```bash
python worker.py --data-dir /ruta/a/ImageNet
```

Los Workers descargan **solo las etiquetas** (~1 MB) en la primera ejecución; las imágenes se cargan lazy bajo demanda desde disco.

---

## Modo Distribuido (varias máquinas)

El Parameter Server (PS) gestiona los pesos globales del MLP. Cada Worker carga ImageNet
localmente, recibe los parámetros actuales y una semilla del PS, reconstruye su propio
chunk de datos localmente (sin transmitir imágenes por red), extrae features con la CNN
(pesos preentrenados en ImageNet o propios), normaliza los features con StandardScaler
y calcula gradientes del MLP. El PS promedia los gradientes de todos los Workers y actualiza
los pesos globales.

```
Worker 0 ──┐
Worker 1 ──┼──► Parameter Server  (ps_terminal.py / ps_gui.py)
Worker N ──┘
```

Los Workers son **persistentes**: no se desconectan entre sesiones de
entrenamiento y pueden participar en varias sesiones sin reiniciarse.

### Ciclo de vida del Parameter Server

```
listen()        → Abre el socket TCP, acepta Workers en un hilo de fondo.
                  Retorna inmediatamente.

[esperar N Workers]

train(...)      → Ejecuta el loop de entrenamiento:
                    Por cada época:
                      1. Genera una semilla aleatoria de época.
                      2. Asigna shards/indices a cada Worker de forma determinista.
                      3. Broadcast: envía params + semilla a cada Worker.
                      4. Cada Worker reconstruye su indices localmente (sin red).
                      5. Espera gradientes de TODOS los Workers (barrera).
                      6. Promedia: ∇θ = (1/N) × Σ ∇θᵢ
                      7. Actualiza: θ ← θ − lr × ∇θ

shutdown()      → Envía STOP a todos los Workers y cierra el socket.
```

### Lanzar el servidor (terminal)

```bash
# Configuración básica: esperar 2 workers, 100 épocas, ImageNet
python ps_terminal.py

# Configuración extendida
python ps_terminal.py --workers 3 --epochs 500 --hidden1 256 --hidden2 128 --lr 0.01 --n-train 1280000
```

| Opción | Descripción | Default |
|---|---|---|
| `--host` | IP de escucha del servidor | `0.0.0.0` |
| `--port` | Puerto TCP | `9999` |
| `--workers` | Número de Workers a esperar antes de entrenar | `2` |
| `--epochs` | Épocas de entrenamiento | `10` |
| `--hidden1` | Neuronas en la primera capa oculta del MLP | `256` |
| `--hidden2` | Neuronas en la segunda capa oculta del MLP | `128` |
| `--lr` | Tasa de aprendizaje | `0.01` |
| `--momentum` | Momentum SGD (0.0 = SGD puro) | `0.9` |
| `--n-train` | Total de ejemplos de entrenamiento | `50000` |
| `--seed` | Semilla aleatoria (reproducibilidad) | ninguna |
| `--cnn-arch` | Arquitectura CNN: `simple` o `resnet18` | `resnet18` |
| `--cnn-device` | Dispositivo PyTorch para CNN: `cpu`, `cuda`, `mps` | `cpu` |
| `--data-dir` | Directorio raiz de ImageNet con `train/` y `val/` | ninguna |
| `--hf-token` | Token HuggingFace para modo stream | `""` |
| `--cnn-pretrain-epochs` | Epocas de pretrain para `simple` (0 = sin pretrain) | `5` |
| `--cnn-pretrain-lr` | Learning rate del pretrain para CNN `simple` | `0.001` |
| `--cnn-pretrain-samples` | Muestras de train para preentrenamiento/fallback | `10000` |

### Lanzar el servidor (GUI)

```bash
python ps_gui.py
```

La GUI expone los mismos parámetros con controles visuales y muestra en tiempo
real las gráficas de precisión y pérdida por época, la tabla de Workers
conectados y un log de eventos.

### Lanzar un Worker

Ejecutar en cada máquina (o terminal) que participará en el entrenamiento.
El Worker recibe su ID del PS automáticamente al conectarse; no hay que
especificarlo manualmente. El Worker carga ImageNet una sola vez, extrae
sus features al iniciar y los cachea para acceso rápido.

```bash
# Worker en la misma máquina que el PS
python worker.py

# Worker en otra máquina
python worker.py --server-host 192.168.1.10

# Modo stream (sin dataset local): token por argumento
python worker.py --hf-token hf_xxxx

# Modo stream (sin dataset local): token por variable de entorno
export HF_TOKEN=hf_xxxx && python worker.py

# Ver todas las opciones
python worker.py --help
```

| Opción | Descripción | Default |
|---|---|---|
| `--server-host` | IP del Parameter Server | `127.0.0.1` |
| `--server-port` | Puerto TCP del Parameter Server | `9999` |
| `--data-dir` | Directorio raiz de ImageNet (`train/` y `val/`) | `Data/ImageNet/` |
| `--hf-token` | Token HuggingFace para modo stream (`HF_TOKEN` tambien aplica) | `""` |
| `--cnn-device` | Dispositivo PyTorch: `cpu`, `cuda`, `mps` | `cpu` |
| `--cache-dir` | Directorio para shards de features | `Data/feature_cache/` |
| `--quiet` | Suprime mensajes de progreso | `False` |

> **Nota:** El Worker debe iniciarse **después** de que el PS esté escuchando.
> El PS bloquea el inicio del entrenamiento hasta que se conecten todos los
> Workers configurados con `--workers`.

### Streaming HuggingFace (sin descarga previa)

```bash
# Configurar token una vez (o pasarlo con --hf-token en cada comando)
export HF_TOKEN=hf_xxxxxxxxxxxx

# Terminal 1: Parameter Server
python ps_terminal.py \
  --workers 1 \
  --epochs 10 \
  --hf-token hf_xxxxxxxxxxxx

# Terminal 2: Worker
python worker.py \
  --hf-token hf_xxxxxxxxxxxx \
  --cnn-device cpu
```

### Ejemplo completo (3 terminales en local)

```bash
# Terminal 1 — Parameter Server (ImageNet, 2 workers, 50 épocas)
python ps_terminal.py --workers 2 --epochs 50

# Terminal 2 — Worker 0
python worker.py

# Terminal 3 — Worker 1
python worker.py
```

---

## Workers con Docker

La carpeta `Docker/` contiene una imagen lista para lanzar Workers en
contenedores, lo que simplifica ejecutar varios Workers en la misma máquina
sin gestionar entornos virtuales por separado.

### Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop) instalado y en ejecución.
- El Parameter Server ya está escuchando en el puerto `9999`.

### Construir la imagen

Ejecutar desde la raíz del proyecto:

```bash
docker build -f Docker/Dockerfile.worker -t nn-worker .
```

> La imagen excluye `Data/` (definido en `.dockerignore`) ya que cada
> contenedor usara dataset local montado o modo stream de ImageNet.

### Lanzar Workers

**Un solo Worker:**
```bash
docker run -d nn-worker python worker.py --server-host host.docker.internal --server-port 9999
```

**N Workers con el script PowerShell:**
```powershell
# Lanza 3 Workers en paralelo
.\Docker\run_workers.ps1 -N 3
```

`host.docker.internal` resuelve automáticamente a la IP de la máquina anfitriona
en Docker Desktop (Windows y macOS), permitiendo que los contenedores se conecten
al PS que corre fuera de Docker.

### Ejemplo completo con Docker (2 terminales)

```bash
# Terminal 1 — Parameter Server (en el host, ImageNet)
python ps_terminal.py --workers 3 --epochs 100

# Terminal 2 — 3 Workers en Docker
.\Docker\run_workers.ps1 -N 3
```

---

## Protocolo de Comunicación

Los mensajes se serializan con **Pickle** y van precedidos de 4 bytes (big-endian)
con la longitud del bloque. Pickle serializa `np.ndarray` de forma binaria nativa,
eliminando conversiones y reduciendo el tamaño en red en un **60-70%** respecto a JSON.

| Mensaje | Dirección | Descripción |
|---|---|---|
| `READY` | Worker → PS | El Worker se conecta y solicita un ID |
| `WORKER_ID` | PS → Worker | PS asigna un ID único al Worker |
| `CNN_WEIGHTS` | PS → Worker | Envio de pesos CNN para extracción consistente de features |
| `CNN_READY` | Worker → PS | Confirmación de que el Worker terminó extracción/carga de features |
| `REQUEST_TEST_FEATURES` | PS → Worker | Solicita features de validación a un Worker específico |
| `TEST_FEATURES` | Worker → PS | Envia features y etiquetas de validación |
| `TRAIN_SAMPLE` | PS → Worker | Solicita muestra de train para preentrenamiento/fallback |
| `TRAIN_SAMPLE_DATA` | Worker → PS | Envia muestra de imágenes y etiquetas de train |
| `TRAIN_START` | PS → Workers | Inicia una sesión; envía `epochs`, `n_train`, `n_workers`, `worker_rank` |
| `PARAMS` | PS → Workers | Pesos globales + semilla de época (el Worker reconstruye su chunk determinísticamente) |
| `GRADIENTS` | Worker → PS | Gradientes calculados sobre el batch asignado |
| `STOP` | PS → Workers | Finaliza la sesión; Workers cierran la conexión |

---

## Arquitectura del Sistema: CNN + MLP Distribuido

La pipeline completa se divide en dos etapas con responsabilidades distintas:

```
┌────────────────────────────────────────────────────────────────┐
│  ImageNet imagen (3 × 224 × 224) — 1.28M imágenes train        │
│         │                                                      │
│         ▼                                                      │
│  ┌──────────────────────┐                                      │
│  │ CNN Extractor        │  ← PyTorch (pesos ImageNet)          │
│  │ (ResNet18/SimpleCNN) │    Idéntica en todos Workers        │
│  │ Modos:              │    Semilla reproducible              │
│  │ • resnet18: frozen  │    (same features everywhere)        │
│  │ • simple: trainable │                                      │
│  └──────────┬──────────┘                                      │
│             │ feature vector  (512,)                          │
│             ▼                                                  │
│  ┌──────────────────────┐                                      │
│  │ Feature Scaler       │  ← StandardScaler (μ=0, σ=1)        │
│  │ (Normalization)      │    Calculado sobre train features   │
│  │ Solo en resnet18     │                                      │
│  └──────────┬──────────┘                                      │
│             │                                                  │
│             ▼                                                  │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ MLP (NumPy) — Distributed Training                        │ │
│  │                                                          │ │
│  │  Input:  512-dims (features)                            │ │
│  │  Output: 1000-dims (logits para 1000 clases ImageNet)   │ │
│  │                                                          │ │
│  │  ┌─────────────────────────────────────────────────┐   │ │
│  │  │ Worker i                                        │   │ │
│  │  │ ────────────────────────────────────────────    │   │ │
│  │  │ Batch i = n_train / n_workers  (estratificado) │   │ │
│  │  │ ▼                                               │   │ │
│  │  │ logits = MLP.forward(X_batch_i)                │   │ │
│  │  │ loss = softmax_cross_entropy(logits_i, Y_i)    │   │ │
│  │  │ grads_i = backward(loss)  [solo MLP!]          │   │ │
│  │  │ ▼                                               │   │ │
│  │  │ Envía gradientes al Parameter Server            │   │ │
│  │  └─────────────────────────────────────────────────┘   │ │
│  │  + [Worker 2, Worker 3, ...]                           │ │
│  │                                                          │ │
│  │  ┌──────────────────────────────────────────────────┐   │ │
│  │  │ Parameter Server (Agregación)                    │   │ │
│  │  │ ────────────────────────────────────────────     │   │ │
│  │  │ params_global = params - lr * mean(all_grads)    │   │ │
│  │  │ Broadcast params_global to all Workers ↺         │   │ │
│  │  └──────────────────────────────────────────────────┘   │ │
│  │                                                          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                              │
│  ▼                                                          │
│  [Predicción en Test Set — validación accuracy]             │
└────────────────────────────────────────────────────────────────┘
```

### Componentes Clave

#### **CNN Extractor** (`Model/cnn_extractor.py`)
- **Propósito**: Extraer features de imágenes ImageNet
- **Modos**:
  - **resnet18**: Pesos ImageNet preentrenados, congelados (requires_grad=False)
  - **simple**: Arquitectura CNN simple, puede ser entrenada junto con MLP
- **Salida**: Vector de 512 dimensiones (penúltima capa ReLU)
- **Inicialización**: Semilla reproducible para synchronizar features entre Workers

#### **Feature Scaler** (`Utils/feature_scaler.py`)
- **Propósito**: Normalizar features a media=0, desv=1
- **Solo en modo resnet18**: Mejora estabilidad del MLP
- **Parámetros**: Calculados sobre todo el training set features (no batch-wise)

#### **MLP** (`Model/mlp.py`)
- **Framework**: NumPy puro (serializable con Pickle)
- **Arquitectura**: 512 → [capas ocultas] → 1000
- **Loss**: Softmax Cross-Entropy
- **Optimizador**: SGD vanilla (distribución de gradientes via Parameter Server)

---

## Modos de Entrenamiento

### Modo "simple" (CNN + MLP Conjuntamente)

```bash
python ps_terminal.py --mode simple --epochs 10
```

**Comportamiento:**
1. CNN es **entrenable** durante distribución
2. NO hay pre-extracción de features
3. Raw images → CNN forward (con_grad) → features → MLP forward/backward
4. Gradientes del CNN NO se sincronizan (solo MLP)

**Caso de uso**: Datasets pequeños, CNN fine-tuning local

---

### Modo "resnet18" (CNN Congelada + MLP Distribuido)

```bash
python ps_terminal.py --mode resnet18 --epochs 10
```

**Comportamiento:**
1. CNN es **congelada** (requires_grad=False)
2. Pre-extrae features a shards (50k imágenes por archivo .npy)
3. FeatureScaler normaliza features
4. features → MLP forward/backward (distribuido via PS)
5. Solo MLP gradientes se sincronizan

**Caso de uso**: Entrenar rápido sobre features precomputadas

---

## CLI - Ejemplos de Uso

### Parameter Server - Terminal

**Uso básico (resnet18, full dataset):**
```bash
python ps_terminal.py --workers 3 --epochs 100 --mode resnet18
```

**Con ImageNet local:**
```bash
python ps_terminal.py --workers 3 --epochs 100 --data-source local --imagenet-dir /path/to/ImageNet
```

**Con streaming (HuggingFace):**
```bash
python ps_terminal.py --workers 3 --epochs 100 --data-source stream --hf-token your_token
```

**Limitar dataset (n_train):**
```bash
python ps_terminal.py --workers 3 --epochs 100 --n-train 100000
```

**Guardar CNN model después del entrenamiento:**
```bash
python ps_terminal.py --workers 3 --epochs 100 --save-cnn
```

**Usar CNN preentrenado:**
```bash
python ps_terminal.py --workers 3 --epochs 100 --cnn-hash abc123def456
```

**Listar modelos CNN guardados:**
```bash
python ps_terminal.py --cnn-list
```

### Worker Node

```bash
python worker.py --ps-host localhost --ps-port 9090
```

---

## Sistema de Persistencia de Modelos CNN

Los modelos CNN entrenados se guardan automáticamente en `Data/cnn_models/`.

**Estructura de archivos:**
```
Data/cnn_models/
├── cnn_abc123def456.pt       # Pesos del modelo (PyTorch)
└── cnn_abc123def456.json     # Metadatos (epochs, accuracy, etc.)
```

**Metadatos JSON:**
```json
{
  "hash": "abc123def456",
  "n_train": 1281167,
  "epochs": 100,
  "final_accuracy": 0.92,
  "final_loss": 0.295,
  "training_time_seconds": 3600.5,
  "timestamp": "2025-03-15 14:30:45"
}
```

**Comandos de gestión:**
```bash
# Listar todos los modelos guardados
python ps_terminal.py --cnn-list

# Usar modelo específico (por hash)
python ps_terminal.py --workers 3 --epochs 10 --cnn-hash abc123def456
```

---

## Troubleshooting

### Error: "ImageNet not found"
**Causa:** Dataset no descargado o estructura incorrecta
**Solución:**
```bash
# Asegurate de que la estructura es:
# Data/ImageNet/train/{synset_id}/{image.JPEG}
# Data/ImageNet/val/{synset_id}/{image.JPEG}

# O especifica ruta alternativa:
python ps_terminal.py --imagenet-dir /ruta/a/ImageNet
```

### Error: "Connection refused" (Worker → PS)
**Causa:** Parameter Server no está corriendo o puerto incorrecto
**Solución:**
```bash
# Terminal 1: inicia PS en puerto explícito
python ps_terminal.py --port 9090

# Terminal 2: conecta Worker al puerto correcto
python worker.py --ps-port 9090
```

### Error: "HuggingFace token not found"
**Causa:** Modo streaming pero token no proporcionado
**Solución:**
```bash
# Opción 1: pasar token como argumento
python ps_terminal.py --data-source stream --hf-token tu_token

# Opción 2: exportar variable de entorno
export HF_TOKEN=tu_token
python ps_terminal.py --data-source stream
```

### Entrenamiento muy lento
**Causa**: Posible: no hay aceleración hardware, dataset demasiado grande, red lenta
**Soluciones:**
- Reduce n_train: `--n-train 100000`
- Usa resnet18 (features precomputadas) en lugar de simple
- Reduce número de epochs
- Verifica velocidad de red entre PS y Workers

### Modo simple: CNN no se está entrenando bien
**Causa:** CNN parameters se inicializan sin semilla reproducible
**Solución:**
- Asegúrate que todos los Workers usan la MISMA semilla (automático en resnet18)
- En modo simple, CNN se entrena localmente, no distribuido

---

## Preguntas Frecuentes (FAQ)

**P: ¿Puedo usar GPU?**
R: El proyecto usa CPU para propósitos demostraticos. Para GPU: reemplaza NumPy MLP con PyTorch GPU, sincroniza gradientes vía NCCL.

**P: ¿Funciona con otros datasets?**
R: Sí. Reemplaza `imagenet_loader.py` con un DataLoader de tu dataset. Asegúrate que el CNN siga outputeando 512-dims.

**P: ¿Cuantos Workers puedo tener?**
R: Teóricamente ilimitado. Cada Worker debe tener acceso a ImageNet (local o streaming). PS agrega gradientes de todos.

**P: ¿Se sincronizan los pesos del CNN en modo simple?**
R: NO. Solo MLP se sincroniza vía Parameter Server. CNN se entrena localmente en cada Worker (extensible en futuro).

**P: ¿Qué batch size usa cada Worker?**
R: Determinado automáticamente: batch_per_worker = n_train / n_workers.

**P: ¿Puedo reanudar entrenamiento desde checkpoint?**
R: Actualmente cada pass de TRAIN_START es independiente. Mejora futura: guardar params_global entre sesiones.

---

## Reproducibilidad

Para reproducir exactamente los mismos resultados:

1. **Semilla**: Todos los Workers usan la misma `seed` (proporcionada por PS en TRAIN_START)
2. **Índices**: Determinísticos — cada Worker calcula su chunk basado en `rank`
3. **Features**: Idénticas entre Workers (mismo CNN + initialization)
4. **Orden de datos**: Estratificado por clase (no aleatorio)

**Ejemplo reproducible:**
```bash
python ps_terminal.py --workers 3 --epochs 5 --n-train 50000 --seed 42
```

Ejecutándolo 2 veces debería dar exactamente los mismos accuracy/loss valores.

---

## Créditos y Referencias

- **ImageNet**: [image-net.org](https://image-net.org/)
- **ResNet18**: He et al., "Deep Residual Learning for Image Recognition" (2015)
- **Distributed SGD**: Dean et al., "Large Scale Distributed Deep Networks" (Google, 2012)
- **Parameter Server**: Li et al., "Scaling Distributed Machine Learning with the Parameter Server" (CMU, 2014)

---

## Licencia

Este proyecto se proporciona con propósitos educativos y de demostración.
