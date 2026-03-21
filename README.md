# Entrenamiento Distribuido ImageNet con CNN + MLP

Proyecto de aprendizaje automático distribuido que implementa el **Algoritmo de Diego** (promediado de gradientes por época) sobre **ImageNet**, combinando una **CNN como extractor de features** (PyTorch, pesos preentrenados o propios) y un **MLP como clasificador** (NumPy, pesos distribuidos). Parameter Server y Workers están conectados por TCP usando **Pickle** para la serialización optimizada de mensajes (sin conversiones de arrays).

---

## Estructura del Proyecto

```
neural-network/
├── Data/
│   └── ImageNet/            # Dataset ImageNet (~150 GB)
│       ├── train/           # 1.28M imágenes de entrenamiento
│       └── val/             # 50k imágenes de validación
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
| `--n-train` | Total de ejemplos de entrenamiento (ImageNet: 1280000) | `1280000` |
| `--seed` | Semilla aleatoria (reproducibilidad) | ninguna |
| `--cnn-arch` | Arquitectura CNN: `simple` o `resnet18` | `resnet18` |
| `--cnn-pretrained` | Usar pesos ImageNet preentrenados | activo por defecto |
| `--data-dir` | Directorio donde está ImageNet | `Data/ImageNet/` |

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

# Especificar arquitectura CNN simple (sin pesos ImageNet preentrenados)
python worker.py --cnn-arch simple

# ResNet18 con ImageNet preentrenado (défault)
python worker.py --cnn-arch resnet18 --cnn-pretrained

# Ver todas las opciones
python worker.py --help
```

| Opción | Descripción | Default |
|---|---|---|
| `--server-host` | IP del Parameter Server | `127.0.0.1` |
| `--server-port` | Puerto TCP del Parameter Server | `9999` |
| `--data-dir` | Directorio donde está ImageNet | `Data/ImageNet/` |
| `--hidden1` | Neuronas capa oculta 1 del MLP | `256` |
| `--hidden2` | Neuronas capa oculta 2 del MLP | `128` |
| `--cnn-arch` | Arquitectura CNN: `simple` o `resnet18` | `resnet18` |
| `--cnn-pretrained` | Usar pesos ImageNet preentrenados | True |
| `--cnn-device` | Dispositivo PyTorch: `cpu`, `cuda`, `mps` | `cpu` |
| `--cnn-seed` | Semilla para inicialización CNN | `42` |
| `--quiet` | Suprime mensajes de progreso | False |

> **Nota:** El Worker debe iniciarse **después** de que el PS esté escuchando.
> El PS bloquea el inicio del entrenamiento hasta que se conecten todos los
> Workers configurados con `--workers`.

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
> contenedor descarga CIFAR-10 automáticamente en su primer arranque.

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
| `TRAIN_START` | PS → Workers | Inicia una sesión; envía `epochs`, `n_train`, `n_workers`, `worker_rank` |
| `PARAMS` | PS → Workers | Pesos globales + semilla de época (el Worker reconstruye su chunk determinísticamente) |
| `GRADIENTS` | Worker → PS | Gradientes calculados sobre el batch asignado |
| `STOP` | PS → Workers | Finaliza la sesión; Workers cierran la conexión |

---

## Arquitectura del Sistema: CNN + MLP Distribuido

La pipeline completa se divide en dos etapas con responsabilidades distintas:

```
┌──────────────────────────────────────────────────────────────┐
│  ImageNet imagen  (3 × 224 × 224) — 1.28M imágenes train     │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────┐                                         │
│  │  CNN Extractor  │  ← PyTorch (pesos preentrenados ImageNet)
│  │  (convolucional)│    o arquitectura propia ("simple")     │
│  │   ResNet18      │    Idéntica en todos los Workers        │
│  │   o SimpleCNN   │    Inicialización: semilla reproducible │
│  └────────┬────────┘                                         │
│           │  feature vector  (512,)                          │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │ FeatureScaler   │  ← StandardScaler (normalización)       │
│  │  (BatchNorm)    │    Media/std calculadas offline         │
│  └────────┬────────┘    sobre features de train              │
│           │  normalized features (512,)                      │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │   MLP (NumPy)   │  ← Pesos DISTRIBUIDOS (Algoritmo Diego) │
│  │  (clasificador) │    PS: promedia gradientes, actualiza   │
│  │  2 capas hidden │    Workers: forw/backward sobre chunks  │
│  └────────┬────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  1000 clases ImageNet  (animal, vehículo, objeto, planta, …) │
└──────────────────────────────────────────────────────────────┘
```

### Por qué CNN preentrenada + MLP distribuido

1. **CNN preentrenada:** Pesos fijos en pesos ImageNet (torchvision).
   - Beneficio práctico real: transfer learning (extracción de features genéricas).
   - Reproducibilidad: todos los Workers usan exactamente los mismos pesos CNN.
   - Eficiencia: no hay que entrenar la CNN desde cero (ahorra semanas de cómputo).

2. **MLP distribuido:** Es donde los Workers difieren en sus datos.
   - Cada Worker recibe los pesos globales + una semilla de época.
   - Reconstruye su chunk de índices localmente de forma determinística (sin transmitir índices por red).
   - Calcula gradientes sobre sus features extraídas.
   - El PS promedia: `∇θ = (1/N) × Σᵢ ∇θᵢ(Bᵢ)`

3. **FeatureScaler:** Normalización StandardScaler para estabilizar el MLP.
   - Calcula media/std offline sobre un subset de features de train.
   - Las estadísticas se guardan en caché y se reutilizan en cada sesión.

### Componentes

#### CNN Extractor (Model/cnn_extractor.py)

Dos arquitecturas disponibles:

- **`resnet18`** (défault): ResNet-18 de torchvision.
  - Con pesos **ImageNet preentrenados** (descarga automática la 1ª vez).
  - Salida: 512 features.
  - Caché: pesos en `Data/feature_cache/` para cargas rápidas.

- **`simple`**: CNN diseñada desde cero.
  - 3 bloques: Conv → BatchNorm → ReLU → MaxPool
  - Salida: 512 features
  - Sin preentrenamiento (inicialización aleatoria)
  - Útil para experimentos pedagógicos

Todas las extracciones de features se **cachean automáticamente** en disco
(con hash de pesos CNN) para evitar recomputar features en cada sesión.

#### Feature Scaler (Utils/feature_scaler.py)

Normalización per-dimensión de los features CNN:

```
Para cada dimensión d:
    x_d_norm = (x_d - media_d) / std_d
```

- **Beneficio:** Mejora convergencia del MLP (gradientes más uniformes).
- **Cálculo:** Se realiza offline sobre un shard de 50k features de train.
- **Caché:** Se guarda junto con los features cachés.

#### MLP (Model/mlp.py)

Red en NumPy puro de dos capas ocultas:

```
Entrada (512 features normalizados)
    │
    ▼  W1 (512 × hidden1)
Oculta 1 (hidden1 neuronas, ReLU)
    │
    ▼  W2 (hidden1 × hidden2)
Oculta 2 (hidden2 neuronas, ReLU)
    │
    ▼  W3 (hidden2 × 1000)
Salida (1000 clases, softmax)
```

- **Forward:** `Z1 = W1·Xᵀ + b1` → `A1 = ReLU(Z1)` → `Z2 = W2·A1 + b2` → `A2 = ReLU(Z2)` → `Z3 = W3·A2 + b3` → `softmax(Z3)`
- **Backward:** retropropagación estándar.
- **Pérdida:** entropía cruzada promediada por muestra.
- **Inicialización:** Xavier para W1, W2, W3; ceros para sesgos.
- **Optimizador:** SGD con momentum (défault: 0.9)

El MLP usa **NumPy puro** porque:
- Los gradientes se serializan con Pickle.
- NumPy arrays son ~3x más pequeños que tensores PyTorch serializados.
- Implementación transparente: se ve exactamente qué se envía por la red.

#### ImageNet Loader (Utils/imagenet_loader.py)

Cargador lazy de ImageNet que nunca carga el dataset completo en RAM:

- **DataLoader PyTorch:** Carga imágenes bajo demanda desde disco durante shards.
- **Lazy labels:** Solo carga las etiquetas (1 MB) en la primera ejecución.
- **Normalización:** Transformaciones estándar de ImageNet (Resize, CenterCrop, Normalize).

---
