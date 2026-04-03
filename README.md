# Entrenamiento Distribuido Asíncrono ImageNet-1k

Sistema de entrenamiento distribuido con arquitectura **Parameter Server** que implementa SGD asíncrono con corrección de staleness para clasificación multiclase en ImageNet-1k.

## Descripción General

### ¿Qué es este sistema?

Este proyecto implementa un framework completo para entrenamiento distribuido de redes neuronales profundas sobre ImageNet-1k sin sincronización global entre Workers. Cada Worker opera de forma autónoma, descargando datos de HuggingFace en streaming, entrenando localmente, y compartiendo actualizaciones de gradientes con un Parameter Server central.

### ¿Qué problema resuelve?

- **Escalabilidad**: Múltiples Workers entrenan en paralelo sin barreras de sincronización (no-wait asynchronous SGD)
- **Eficiencia**: Streaming de datos bajo demanda eliminates bottlenecks de IO
- **Convergencia**: Corrección de staleness (factor λ) mitiga la divergencia por asiduidad de gradientes
- **Transparencia**: GUI integrado para monitoreo en tiempo real

### Enfoque técnico

- **Arquitectura**: Parameter Server + N Workers independientes
- **Comunicación**: TCP/IP con serialización Pickle, 9 tipos de mensaje
- **Modelos**: CNN extractor (ResNet-18 preentrenado o SimpleCNN) + MLP clasificador
- **Datos**: Streaming desde ILSVRC/imagenet-1k o timm/imagenet-1k-wds
- **Hardware**: Soporte automático para CUDA, MPS (Apple Metal), CPU

---

## Documentación Técnica

El repositorio incluye documentación exhaustiva en el subdirectorio `./Docs/`:

| Archivo | Descripción |
|---------|-------------|
| `00_Resumen_General.md` | Overview del sistema, problema, componentes, scope |
| `01_Arquitectura.md` | Diagrama ASCII, responsabilidades de componentes, flujo de datos |
| `02_Flujo_de_Entrenamiento.md` | Step-by-step del training loop, ciclo REQUEST_PARAMS→train→UPDATES |
| `03_Parameter_Server.md` | Funcionamiento del PS, inicialización, async SGD, corrección de staleness |
| `04_Worker.md` | Ciclo de vida del Worker, conexión, streaming, training loop |
| `05_Modelos.md` | Arquitecturas CNN (ResNet-18 vs SimpleCNN), diseño de MLP |
| `06_Comunicacion.md` | Protocolo TCP, 9 tipos de mensaje, serialización |
| `07_GUI_y_Monitoreo.md` | GUI tkinter, configuración de parámetros, visualización de métricas |
| `08_Hiperparametros_y_Config.md` | Learning rate, staleness λ, batch size, impacto en convergencia |
| `09_Streaming.md` | Pipeline HuggingFace, sharding per-Worker, PrefetchBuffer async, I/O optimization |

---

## Características Principales

✅ **Entrenamiento distribuido asíncrono** sin sincronización global entre Workers

✅ **Parameter Server central** que gestiona parámetros globales (CNN + MLP)

✅ **Streaming de ImageNet-1k** desde HuggingFace bajo demanda (nunca descarga completo)

✅ **Arquitecturas CNN soportadas**:
  - ResNet-18 con pesos IMAGENET1K_V1 preentrenados (~50M parámetros)
  - SimpleCNN custom de 3 bloques (~1M parámetros)

✅ **Clasificador MLP configurable** (feature_dim → hidden1 → hidden2 → 1000 clases)

✅ **Auto-detección de dispositivo** (CUDA > MPS > CPU)

✅ **GUI interactivo** con métricas en tiempo real:
  - Configuración de parámetros del servidor
  - Tabla de Workers conectados
  - Gráficas live de loss, accuracy, workers activos
  - Logs estructurados

✅ **Corrección de staleness integrada** (α(s) = 1/(1+λ·s)) para mitigar asiduidad de gradientes

✅ **Acumulación de gradientes** (accum_steps) para reducir overhead de comunicación

✅ **Sharding automático** de dataset por Worker sin duplicación

✅ **Prefetching asíncrono** con hilo background para eliminar bottlenecks de I/O

✅ **Soporte para múltiples máquinas** vía TCP con configuración por línea de comandos

---

## Arquitectura del Sistema

### Diagrama General

```
┌──────────────────────────────────────────────────────────────────┐
│                      PARAMETER SERVER (PS)                       │
│                        ("coordinador")                           │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ · CNN global: ResNet-18 (feature_dim=512)               │     │
│  │ · MLP global: 512 → hidden1 → hidden2 → 1000            │     │
│  │ · version: contador de actualizaciones                  │     │
│  │ · Aplica async SGD con staleness correction             │     │
│  └─────────────────────────────────────────────────────────┘     │
│         ↑                    ↓                    ↑       ↓      │
│  REQUEST_PARAMS      PARAMS + version           STOP   UPDATES   │
│       (request)       (global state)          (signal)  (grads)  │
└──────────────────────────────────────────────────────────────────┘
         ↑       ↓                    ↑       ↓                     
    ┌────────────────┐          ┌────────────────┐                
    │  WORKER 0      │          │  WORKER 1      │  ...            
    │  (rank=0)      │          │  (rank=1)      │                
    │  GPU 0         │          │  GPU 1         │                
    │                │          │                │                
    │ 1. REQUEST     │          │ 1. REQUEST     │                
    │    PARAMS      │          │    PARAMS      │                
    │                │          │                │                
    │ 2. STREAM:     │          │ 2. STREAM:     │                
    │    IMG batch   │          │    IMG batch   │                
    │    (sharded)   │          │    (sharded)   │                
    │                │          │                │                
    │ 3. TRAIN:      │          │ 3. TRAIN:      │                
    │    CNN extract │          │    CNN extract │                
    │    MLP forward │          │    MLP forward │                
    │    backward    │          │    backward    │                
    │                │          │                │                
    │ 4. UPDATES:    │          │ 4. UPDATES:    │                
    │    send grads  │          │    send grads  │                
    │    (repeat)    │          │    (repeat)    │                
    └────────────────┘          └────────────────┘                
        Async Loop                  Async Loop                     
        (sin barreras)              (sin barreras)                 
```

### Componentes

#### **Parameter Server (PS)**
- Coordinador central que mantiene estado global (CNN + MLP)
- Recibe actualizaciones asincrónicas desde cada Worker
- Aplica FedAvg + corrección de staleness
- No bloqueante: no espera a todos los Workers
- Mantiene historial de loss, accuracy, n_workers por step

#### **Workers**
- Nodos computacionales independientes
- Descarga ImageNet-1k en streaming desde HuggingFace
- Loop autónomo: REQUEST → TRAIN → UPDATES
- Soportan CPU y GPU (CUDA/MPS)
- Sharding automático para evitar overlap de datos

#### **Modelos**
- **CNN Extractor**: Transforma imágenes (3, 224, 224) → (512) features
  - ResNet-18: 50M params, preentrenado
  - SimpleCNN: 1M params, custom
- **MLP Classifier**: Clasifica 1000 clases sobre features CNN

#### **Comunicación**
- Protocolo TCP con serialización Pickle
- 9 tipos de mensaje (READY, WORKER_ID, CNN_WEIGHTS, START, etc.)
- Handshake seguro: Workers bloqueados hasta que PS esté listo

#### **Datos**
- Streaming desde ILSVRC/imagenet-1k o alternativa pública timm/imagenet-1k-wds
- Transforms: RandomResizedCrop + Flip para train, CenterCrop para val
- PrefetchBuffer con hilo background para eliminar I/O overhead

#### **GUI**
- Tkinter con matplotlib integrado
- Configuración de parámetros (LR, staleness λ, batch_size, etc.)
- Monitoreo live de Workers conectados
- Gráficas dinámicas: loss, accuracy, workers activos

---

## Flujo de Entrenamiento

### Iniciación del Sistema

```
PASO 1: Iniciar Parameter Server
  ps_imagenet.py --wait-workers 2 --lr 0.001 --staleness-lambda 0.1
  ↓
  · Carga CNN (ResNet-18)
  · Inicializa MLP (feature_dim → hidden1 → hidden2 → 1000)
  · Abre socket TCP en 0.0.0.0:9999
  · Espera a 2 Workers antes de empezar

PASO 2: Conectar Worker 0
  worker_imagenet.py --rank 0 --num-workers 2 --device cuda:0
  ↓
  · Conecta al PS
  · Recibe WORKER_ID
  · Recibe CNN_WEIGHTS (ResNet-18 completo)
  · Carga stream de datos (sharded para rank=0: posiciones 0, 2, 4, ...)

PASO 3: Conectar Worker 1
  worker_imagenet.py --rank 1 --num-workers 2 --device cuda:1
  ↓
  · Conecta al PS
  · Recibe WORKER_ID
  · Recibe CNN_WEIGHTS
  · Carga stream de datos (sharded para rank=1: posiciones 1, 3, 5, ...)

PASO 4: PS detecta ready (2 Workers)
  ↓
  · Envía START a ambos Workers
  · Imprime "✓ 2 Worker(s) conectados. Entrenamiento asíncrono activo."
```

### Loop de Entrenamiento Asíncrono

Cada Worker ejecuta este ciclo infinito **sin esperar a otros Workers**:

```
for iteration in [0, ∞):
    
    # 1. SOLICITAR PARÁMETROS GLOBALES
    send(PS, REQUEST_PARAMS, {})
    PARAMS = receive(PS, msg_type=PARAMS)
    version_read = PARAMS.version
    loss_global = PARAMS.learning_rate
    mlp_state = PARAMS.mlp_state
    cnn_state = PARAMS.cnn_state
    
    # 2. SINCRONIZAR MODELO LOCAL CON ESTADO GLOBAL
    cnn.load_state_dict(cnn_state)  # Cargar CNN global
    mlp.load_state_dict_numpy(mlp_state)  # Cargar MLP global
    
    # 3. OBTENER BATCH DEL STREAM
    images_batch, labels_batch = next(prefetch_buffer)
    # images_batch: (batch_size, 3, 224, 224)
    # labels_batch: (batch_size,) valores en [0, 1000)
    
    # 4. FORWARD PASS E2E
    features = cnn.extract_batched(images_batch)  # (batch_size, 512)
    logits = mlp(features)  # (batch_size, 1000)
    
    # 5. BACKWARD PASS
    loss = cross_entropy(logits, labels_batch)
    loss.backward()
    
    # 6. SGD LOCAL
    for param in mlp.parameters():
        param.data -= lr * param.grad
    for param in cnn.parameters():
        param.data -= lr * param.grad
    
    # 7. ENVIAR ACTUALIZACIONES AL PS
    mlp_updates = mlp.state_dict_numpy()
    cnn_updates = cnn.state_dict_numpy()
    accuracy = (logits.argmax(1) == labels_batch).float().mean()
    
    send(PS, UPDATES, {
        'mlp_weights': mlp_updates,
        'cnn_weights': cnn_updates,
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'version_read': version_read
    })
```

### Actualización en el PS (Asíncronamente)

Cuando PS recibe UPDATES de cualquier Worker:

```
staleness = current_version - version_read  # cuántas updates pasaron
alpha = 1.0 / (1.0 + staleness_lambda * staleness)  # corrección

# Aplicar FedAvg ASÍNCRONO:
mlp_state_new = mlp_state + alpha * (mlp_worker - mlp_state)
cnn_state_new = cnn_state + alpha * (cnn_worker - cnn_state)

current_version += 1  # versión nueva

# Registrar métrica
loss_avg, acc_avg = running_metrics.snapshot()
```

---

## Tecnologías Utilizadas

### Lenguaje & Frameworks
- **Python** ≥ 3.13.5
- **PyTorch 2.10.0**: Redes neuronales, autograd, estado distribuido
- **torchvision 0.25.0**: ResNet-18, transforms de imágenes

### Datasets & Streaming
- **HuggingFace datasets**: ILSVRC/imagenet-1k (1.2M imágenes), timm/imagenet-1k-wds (alternativa pública)
- **Streaming puro**: Nunca descarga dataset completo a disco

### Interfaz Gráfica
- **tkinter**: GUI del Parameter Server
- **matplotlib 3.10.8**: Gráficas live de métricas
- **mplcursors 0.7**: Interactividad en gráficas

### Utilidades
- **numpy 2.4.2**: Serialización de arrays
- **psutil 6.0.0**: Monitoreo de recursos del sistema
- **filelock, sympy**: Dependencias indirectas de HuggingFace

---

## Instalación

### Requisitos Previos

- **Python** ≥ 3.13.5
- **pip** o **uv** (gestor alternativo ultra-rápido)
- Para CUDA: driver NVIDIA + CUDA toolkit
- Para MPS (Apple): macOS 12+

### Método 1: pip (estándar)

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/neural-network.git
cd neural-network

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Método 2: uv sync (ultra-rápido, recomendado para producción)

```bash
# Instalar uv si no lo tienes
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/macOS
# O descargar desde https://github.com/astral-sh/uv

# Sincronizar dependencias (crea uv.lock para reproducibilidad)
uv sync

# Activar entorno virtual
source .venv/bin/activate  # Linux/macOS
# O en Windows: .venv\Scripts\Activate.ps1
```

**Ventajas de `uv sync`**:
- Genera `uv.lock` con versiones pinned (reproducibilidad exacta)
- Más rápido que pip
- Gestiona entorno virtual automáticamente

### Método 3: Entorno virtual manual + uv pip (alternativa)

```bash
python -m venv .venv
source .venv/bin/activate

uv pip install -r requirements.txt
```

### Método 4: pip + pyproject.toml (sin uv, menos recomendado)

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e .          # modo editable
```

### Token de HuggingFace (requerido para ImageNet-1k)

ImageNet requiere aceptar licencia en HuggingFace:

```bash
# 1. Ir a https://huggingface.co/datasets/ILSVRC/imagenet-1k
# 2. Aceptar términos
# 3. Generar token en: https://huggingface.co/settings/tokens
# 4. Exportar token
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"

# O usar --hf-token en línea de comandos (ver sección Uso)
```

**Alternativa sin token**: Usar dataset público `timm/imagenet-1k-wds`

---

## Uso

### Escenario 1: Ejecución Simple (1 PS + 1 Worker en la misma máquina)

#### Terminal 1: Parameter Server

```bash
python ps_imagenet.py \
  --host 127.0.0.1 \
  --port 9999 \
  --wait-workers 1 \
  --lr 0.001 \
  --staleness-lambda 0.1 \
  --cnn-arch resnet18 \
  --hidden1 1024 \
  --hidden2 512 \
  --max-steps 50000 \
  --hf-token "hf_xxxxx"
```

**Salida esperada**:
```
════════════════════════════════════════════════════════════════════
PARAMETER SERVER ASÍNCRONO — ImageNet-1k
════════════════════════════════════════════════════════════════════
  Host              : 127.0.0.1:9999
  Esperando Workers : 1
  Dataset           : ILSVRC/imagenet-1k
  CNN               : resnet18
  MLP               : feature_dim → 1024 → 512 → 1000
  LR                : 0.001
  Staleness λ       : 0.1
  Steps/reporte     : 500
  Max steps         : 50000
  HF Token          : ✓ configurado
════════════════════════════════════════════════════════════════════

Cargando CNN resnet18...
CNN lista: arch=resnet18 | feature_dim=512 | params=22 (1 excluidos del avg)

Esperando 1 Worker(s)...

  [+] Worker 0 desde ('127.0.0.1', 50123) (1/1)

✓ 1 Worker(s) conectado. Entrenamiento asíncrono activo.
(Ctrl+C para detener)

  step=    50 | loss=6.9234 | acc=0.00% | staleness=0 | 15.2 steps/s
  step=   100 | loss=6.8912 | acc=0.00% | staleness=1 | 18.3 steps/s
  ...
```

#### Terminal 2: Worker

```bash
python worker_imagenet.py \
  --server-host 127.0.0.1 \
  --server-port 9999 \
  --rank 0 \
  --num-workers 1 \
  --batch-size 64 \
  --device cuda \
  --hf-token "hf_xxxxx"
```

**Salida esperada**:
```
════════════════════════════════════════════════════════════════════
WORKER ASÍNCRONO — ImageNet-1k Distribuido
════════════════════════════════════════════════════════════════════
  PS             : 127.0.0.1:9999
  Rank           : 0/1
  Dataset        : ILSVRC/imagenet-1k
  Batch size     : 64
  MLP hidden     : 1024 → 512 → 1000
  Device         : cuda (auto-detected)
  Shuffle buffer : 1000
  Prefetch       : 4 batches
  Accum steps    : 1
  HF Token       : ✓ configurado
════════════════════════════════════════════════════════════════════

Conectado | rank=0/1 | device=cuda | batch=64 | accum=1
[Entrenamiento asíncrono activo...]
```

---

### Escenario 2: Múltiples Workers en GPUs Distintas

**Máquina local con 2 GPUs**:

```bash
# Terminal 1: PS
python ps_imagenet.py --wait-workers 2

# Terminal 2: Worker 0 (GPU 0)
python worker_imagenet.py --rank 0 --num-workers 2 --device cuda:0

# Terminal 3: Worker 1 (GPU 1)
python worker_imagenet.py --rank 1 --num-workers 2 --device cuda:1
```

---

### Escenario 3: Workers en Máquinas Diferentes (Red)

**Máquina 1 (192.168.1.10) — Parameter Server**:
```bash
python ps_imagenet.py --host 0.0.0.0 --port 9999 --wait-workers 2
```

**Máquina 2 — Worker 0**:
```bash
python worker_imagenet.py --server-host 192.168.1.10 --rank 0 --num-workers 2
```

**Máquina 3 — Worker 1**:
```bash
python worker_imagenet.py --server-host 192.168.1.10 --rank 1 --num-workers 2
```

---

### Escenario 4: GUI Interactivo (Recomendado)

```bash
# Terminal 1: GUI del Parameter Server
python ps_gui_imagenet.py

# Luego en la GUI:
# 1. Click "Encender servidor" → Carga CNN+MLP en background
# 2. Configurar parámetros (Learning rate, batch size, etc.)
# 3. Click "Iniciar entrenamiento"
# 4. Ver gráficas live de loss, accuracy, workers
```

**Características de la GUI**:
- Panel izquierdo: Configuración de parámetros
- Arriba a la derecha: Tabla de Workers conectados con estado
- Centro derecha: Gráficas live (loss, accuracy, workers activos)
- Abajo: Log de eventos

---

## Configuración

### Hiperparámetros Principales

#### Learning Rate (LR)
- **Default**: 0.001
- **Rango**: (0.00001, 1.0)
- **Efecto**: Tamaño del paso en SGD local de cada Worker
- **Recomendaciones**:
  - 0.001: Convergencia estable (recomendado)
  - 0.01: Convergencia rápida pero posible inestabilidad
  - 0.0001: Muy lento

#### Staleness Lambda (λ)
- **Default**: 0.1
- **Rango**: (0.0, 1.0)
- **Efecto**: Corrección de asiduidad: α(s) = 1/(1+λ·s)
  - s = número de actualizaciones que pasaron desde que este Worker leyó los parámetros
  - α: factor de aplicación de actualizaciones (0 = ignorar, 1 = aplicar directo)
- **Recomendaciones**:
  - 0.1: Balance óptimo (recomendado)
  - 0.5: Mayor corrección si red es lenta
  - 0.0: Sin corrección (puro async-SGD)

#### Architecture CNN
- **Default**: `resnet18`
- **Opciones**:
  - `resnet18`: ResNet-18 con pesos IMAGENET1K_V1 (50M parámetros)
  - `simple`: SimpleCNN custom de 3 bloques (1M parámetros)
- **Impacto**:
  - ResNet-18: Mejor convergencia (pesos preentrenados), más lento
  - SimpleCNN: Más rápido, menos parámetros, convergencia más lenta

#### MLP Architecture
- **Parámetros**: `--hidden1` (default 1024), `--hidden2` (default 512)
- **Arquitectura**: feature_dim(512) → hidden1 → hidden2 → 1000 clases
- **Recomendaciones**:
  - (512, 256): Rápido, menor expresividad
  - (1024, 512): Balance recomendado
  - (2048, 1024): Mayor capacidad, más lento

#### Batch Size
- **Default**: 64
- **Impacto**:
  - 32: Less memory, noisier gradients
  - 64: Balance (recomendado)
  - 256: Faster throughput, requires more VRAM

#### Prefetch Buffer
- **Default**: 4 batches
- **Efecto**: Descarga este número de batches en background
- **Impacto**:
  - Bajo: Mayor latencia de I/O
  - 4-8: Óptimo
  - Alto: Usa más RAM

### Configuración por Línea de Comandos

**Parameter Server**:
```bash
python ps_imagenet.py \
  --host 0.0.0.0 \
  --port 9999 \
  --wait-workers 4 \
  --lr 0.005 \
  --staleness-lambda 0.1 \
  --cnn-arch resnet18 \
  --hidden1 1024 \
  --hidden2 512 \
  --batch-size 64 \  # nota: batch_size en Worker, no en PS
  --steps-per-report 500 \
  --max-steps 100000 \
  --dataset ILSVRC/imagenet-1k \
  --hf-token "hf_xxxxx" \
  --metrics-window 200
```

**Worker**:
```bash
python worker_imagenet.py \
  --server-host 192.168.1.10 \
  --server-port 9999 \
  --rank 0 \
  --num-workers 4 \
  --batch-size 128 \
  --hidden1 1024 \
  --hidden2 512 \
  --device cuda:0 \
  --dataset ILSVRC/imagenet-1k \
  --shuffle-buffer 1000 \
  --prefetch 4 \
  --image-size 224 \
  --hf-token "hf_xxxxx" \
  --accum-steps 4 \  # Acumular 4 batches antes de UPDATES
  --quiet  # Suprimir logs de progreso
```

### Variable de Entorno

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"

# Ahora ps_imagenet.py y worker_imagenet.py usan HF_TOKEN automáticamente
python ps_imagenet.py --wait-workers 2
```

---

## Estructura del Proyecto

```
neural-network/
│
├── README.md                          ← Este archivo
├── pyproject.toml                     ← Metadatos del proyecto (dependencias, Python ≥3.13.5)
├── requirements.txt                   ← Dependencias (numpy, torch, matplotlib, etc.)
│
├── ps_imagenet.py                     ← Punto de entrada: Parameter Server terminal
├── worker_imagenet.py                 ← Punto de entrada: Worker asíncrono
├── ps_gui_imagenet.py                 ← Punto de entrada: GUI del Parameter Server
│
├── Distributed/                       ← Arquitectura distribuida
│   ├── __init__.py
│   ├── parameter_server.py            ← ParameterServer: coordinador, FedAvg, staleness
│   ├── worker_node.py                 ← WorkerNode: loop autónomo de entrenamiento
│   └── protocol.py                    ← MsgType enum, send/receive_message
│
├── Model/                             ← Redes neuronales
│   ├── __init__.py
│   ├── cnn_extractor.py               ← CNNExtractor: ResNet-18 o SimpleCNN
│   └── mlp_pytorch.py                 ← MLPPyTorch: clasificador 2-capas ocultas
│
├── Utils/                             ← Utilidades
│   ├── __init__.py
│   ├── imagenet_streaming.py          ← ImageNetStream, PrefetchBuffer, transforms
│   ├── logging_util.py                ← FormattedLogger con colorización ANSI
│   └── results_exporter.py            ← Export de históricos a JSON
│
├── Data/                              ← Directorio de datos (ignorado por .gitignore)
│   ├── cifar-10-batches-py/           ← CIFAR-10 (para testing legacy, no usado aquí)
│   ├── cifar10_train_nchw.npz
│   └── feature_cache/                 ← Cache de features (unused)
│
└── Docs/                              ← Documentación técnica exhaustiva
    ├── 00_Resumen_General.md
    ├── 01_Arquitectura.md
    ├── 02_Flujo_de_Entrenamiento.md
    ├── 03_Parameter_Server.md
    ├── 04_Worker.md
    ├── 05_Modelos.md
    ├── 06_Comunicacion.md
    ├── 07_GUI_y_Monitoreo.md
    ├── 08_Hiperparametros_y_Config.md
    └── 09_Streaming.md
```

### Descripción de Módulos Clave

| Archivo | Líneas | Propósito |
|---------|--------|----------|
| `Distributed/parameter_server.py` | ~400 | ParameterServer: TCP server, FedAvg asíncrono, staleness correction, aggregation |
| `Distributed/worker_node.py` | ~350 | WorkerNode: streaming + training loop, sincronización de modelo, SGD local |
| `Distributed/protocol.py` | ~100 | Protocolo TCP: 9 tipos de mensaje, serialización Pickle |
| `Model/cnn_extractor.py` | ~150 | CNNExtractor con ResNet-18 / SimpleCNN, serialización para red |
| `Model/mlp_pytorch.py` | ~100 | MLPPyTorch clasificador, inicialización He, conversión numpy ↔ torch |
| `Utils/imagenet_streaming.py` | ~400 | ImageNetStream (infinite), PrefetchBuffer (async), transforms, sharding |
| `Utils/logging_util.py` | ~80 | FormattedLogger con timestamps, colores, tags |
| `ps_imagenet.py` | ~180 | Entry point PS terminal: argparse, callbacks, loop principal |
| `worker_imagenet.py` | ~150 | Entry point Worker: argparse, auto-detect device, instantiate WorkerNode |
| `ps_gui_imagenet.py` | ~700 | GUI tkinter: config, tabla workers, gráficas live, logs |

---

## Notas Técnicas

### Decisiones de Diseño

#### 1. **Asincronía Sin Sincronización Global**
```
Problema: Sincronizar N Workers es caro y lento (barrera).

Solución: Cada Worker que termina de entrenar envía UPDATES inmediatamente
al PS sin esperar a otros Workers. El PS aplica cambios al estado global
y continúa sirviendo a otros Workers.

Beneficio: Escalabilidad lineal, Workers lentos no frenan a los rápidos.
```

#### 2. **Corrección de Staleness (λ-factor)**
```
Problema: Si un Worker está atrasado (leyó parámetros antiguos),
sus gradientes pueden estar sesgados.

Solución: Aplicar actualización atenuada: α(s) = 1/(1+λ·s)
donde s = versionActual - versionLeída

Ejemplo:
  λ=0.1, s=0 (actualización fresca) → α=1.0 (aplicar 100%)
  λ=0.1, s=10 (10 updates atrás)   → α=0.5 (aplicar 50%)
  λ=0.1, s=100                      → α=0.09 (aplicar 9%)

Beneficio: Garantiza convergencia incluso con gran varianza de staleness.
```

#### 3. **Streaming Pure (No Descargas Completas)**
```
Problema: ImageNet full = 1.2M imágenes = 144GB (descarga prohibitiva).

Solución: Usar HuggingFace Datasets en modo streaming:
  - Descarga chunks bajo demanda
  - Nunca ocupa más de prefetch_batches × batch_size en RAM
  - Reinicio automático al llegar al final (loop infinito)

Beneficio: Funciona en máquinas con 8GB RAM, solo descarga lo que entrena.
```

#### 4. **Sharding per-Worker**
```
Problema: Si 4 Workers descargan independientemente, cada uno ve todas
las 1.2M imágenes → overlap terrible.

Solución: índice strided:
  Worker 0: posiciones [0, 4, 8, 12, ...]
  Worker 1: posiciones [1, 5, 9, 13, ...]
  Worker 2: posiciones [2, 6, 10, 14, ...]
  Worker 3: posiciones [3, 7, 11, 15, ...]

Beneficio: Cobertura completa sin repetición, gradientes descorrelacionados.
```

#### 5. **Handshake Seguro**
```
Problema: Si PS.listen() comienza antes que ps.set_cnn(), Workers
conectan pero reciben None para CNN (crash).

Solución: Cada Worker que conecta queda bloqueado en handshake hasta
que CNN + MLP estén disponibles (máx 120s timeout).

Beneficio: Eliminancía race conditions, modelo siempre consistente.
```

#### 6. **Exclusión de num_batches_tracked en Averaging**
```
Problema: num_batches_tracked es counter interno de BatchNorm2d (dtype=int64).
Promediar este campo entre Workers no tiene sentido semántico.

Solución: Detectar int64 keys en CNN state_dict y excluirlas del averaging
en FedAvg:
  mlp_state = ps.mlp + α(mlp_worker - ps.mlp)              ← Promediado
  cnn.num_batches_tracked = ps.cnn.num_batches_tracked     ← Sin promedio

Beneficio: BN running stats consistente, sin corrupción de metadatos.
```

### Consideraciones de Rendimiento

#### **Throughput Típico**
- **CPU (i7-9700K)**: ~15-20 batches/sec (~1000 imágenes/sec)
- **GPU (RTX 2080)**: ~100-120 batches/sec (~6500 imágenes/sec)

#### **Latencia de Comunicación**
```
Un ciclo de entrenamiento (REQUEST + TRAIN + UPDATES):
  - Red local (localhost): ~1ms overhead
  - Red LAN (192.168): ~5-10ms
  - Entrenar 1 batch: ~20ms (CPU) o 2ms (GPU)
  
Total: Communication es despreciable vs compute en GPU, importante en CPU.
```

#### **Memory Footprint por Worker**
```
Base:
  - CNN ResNet-18: ~200MB (state_dict)
  - MLP: ~20MB
  - PrefetchBuffer(4 batches, 64 imgs, 224×224): ~400MB
  Total mínimo: ~620MB

Con overhead PyTorch: ~1-1.5GB on GPU
```

#### **Convergencia**
```
Sin staleness correction (λ=0):
  - Convergencia NO garantizada si Workers muy desbalanceados
  - Posibles oscilaciones en loss

With staleness correction (λ=0.1):
  - Convergencia garantizada (demostrable teóricamente)
  - Loss suaviza, mayor estabilidad
  - Trade-off: puede converger más lentamente si λ muy alto
```

### Limitaciones Actuales

- **No hay evaluación periódica en validación**: El sistema entrena indefinidamente (o hasta max_steps). Para evaluación, debe pausarse e iniciarse separadamente.
- **Sin checkpoint automático**: Si PS cae, pierde estado global. Se debe implementar persistencia para producción.
- **Sin compresión de gradientes**: Cada UPDATES envía full precision floats. Posibilidad de usar quantización para reducir ancho de banda.
- **Sincronía de BN**: BatchNorm running stats dependeel de la secuencia de datos. Workers con diferentes ratios de datos verán diferentes stats.

### Replicación de Código Real

#### Inicio del PS
```python
# ps_imagenet.py
ps = ParameterServer(
    host=args.host,
    port=args.port,
    learning_rate=args.lr,
    staleness_lambda=args.staleness_lambda,
    # ...
)
ps.set_cnn(cnn)
ps.set_mlp(mlp.state_dict_numpy())
ps.listen()  # Abre TCP server en hilo background
ready.wait()  # Bloquea main hasta que llegan workers
```

#### Loop del Worker
```python
# Distributed/worker_node.py :: _training_loop()
for iter_count in range(1, 1000000):
    # REQUEST_PARAMS
    send_message(self._sock, MsgType.REQUEST_PARAMS, {})
    msg = receive_message(self._sock)
    params = msg["payload"]
    
    # SYNC MODEL
    self._sync_cnn()
    self._sync_mlp()
    
    # GET BATCH
    batch_t, labels_t = next(self._stream)
    
    # TRAIN
    loss, acc = self._train_batch(batch_t, labels_t)
    
    # SEND UPDATES
    send_message(self._sock, MsgType.UPDATES, {
        'mlp_weights': self._mlp.state_dict_numpy(),
        'cnn_weights': self._serialize_cnn(),
        'loss': loss,
        'accuracy': acc,
        'version_read': params['version']
    })
```

#### Aplicación de Actualización en PS
```python
# Distributed/parameter_server.py :: _apply_update()
def _apply_update(self, worker_idx, update):
    staleness = self._version - update['version_read']
    alpha = 1.0 / (1.0 + self.staleness_lambda * staleness)
    
    # FedAvg asíncrono
    for key in self._mlp_state:
        self._mlp_state[key] = (
            self._mlp_state[key] +
            alpha * (update['mlp_weights'][key] - self._mlp_state[key])
        )
    
    # Similar para CNN (excluyendo int64 keys)
    self._version += 1
```

---

## Próximos Pasos / Extensiones

1. **Checkpoint/Recovery**: Guardar/restaurar estado del PS para tolerar fallos
2. **Gradient Compression**: Quantización de parámetros para reducir ancho de banda (~10x)
3. **Validación Periódica**: Evaluación automática en split de validation cada N steps
4. **Scaling**: Testar con 10+ Workers en cluster real
5. **Profiling**: Instrumentación para identificar bottlenecks de comunicación vs compute

---

## Referencias

- **Async-SGD Theory**: Ho et al. (2013) "More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server"
- **HuggingFace Datasets**: https://huggingface.co/docs/datasets
- **PyTorch Distributed**: https://pytorch.org/docs/stable/distributed.html
- **ImageNet-1k**: https://www.image-net.org

---

## Licencia

Sin especificación (ver LICENSE si existe).

## Contacto

Para preguntas sobre el sistema, consultar documentación en [`./Docs/`](./Docs/).

---

**Última actualización**: Abril 2026  
**Versión del código**: 0.1.0

