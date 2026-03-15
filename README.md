# Entrenamiento Distribuido CIFAR-10 con CNN + MLP

Proyecto de aprendizaje automático distribuido que implementa el **Algoritmo de Diego** (promediado de gradientes por época) sobre CIFAR-10, combinando una **CNN como extractor de features** (PyTorch, pesos fijos) y un **MLP como clasificador** (NumPy, pesos distribuidos). Parameter Server y Workers están conectados por TCP usando Pickle para la serialización de mensajes.

---

## Estructura del Proyecto

```
neural-network/
├── Data/
│   └── CIFAR-10/            # Dataset CIFAR-10 (se descarga automáticamente)
├── Distributed/             # Núcleo del sistema distribuido
│   ├── parameter_server.py  # Clase ParameterServer (lógica TCP + entrenamiento)
│   ├── worker_node.py       # Clase WorkerNode (extracción CNN + MLP forward/backward)
│   └── protocol.py          # Serialización de mensajes Pickle sobre TCP
├── Model/                   # Lógica central: CNN + MLP
│   ├── cnn_extractor.py     # CNN PyTorch (features fijos, pesos iniciales cachés)
│   └── mlp.py               # MLP NumPy (pesos distribuidos, gradientes serializables)
├── Utils/
│   ├── cifar_loader.py      # Carga CIFAR-10 en formato NCHW normalizado
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
CIFAR-10 se descarga automáticamente en `Data/CIFAR-10/` al primer uso.

---

## Modo Distribuido (varias máquinas)

El Parameter Server (PS) gestiona los pesos globales del MLP. Cada Worker carga CIFAR-10
localmente, recibe los parámetros actuales y una semilla del PS, reconstruye su propio
chunk de datos localmente (sin transmitir índices por red), extrae features con la CNN
(pesos fijos, idénticos en todos los Workers) y calcula gradientes del MLP. El PS
promedia los gradientes de todos los Workers y actualiza los pesos globales.

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
                      2. Broadcast: envía params + semilla a cada Worker.
                      3. Cada Worker reconstruye su chunk localmente (sin red).
                      4. Espera gradientes de TODOS los Workers (barrera).
                      5. Promedia: ∇θ = (1/N) × Σ ∇θᵢ
                      6. Actualiza: θ ← θ − lr × ∇θ

shutdown()      → Envía STOP a todos los Workers y cierra el socket.
```

### Lanzar el servidor (terminal)

```bash
# Configuración básica: esperar 2 workers, 100 épocas
python ps_terminal.py

# Configuración extendida
python ps_terminal.py --workers 3 --epochs 500 --hidden1 256 --hidden2 128 --lr 0.01 --n-train 60000
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
especificarlo manualmente. El Worker carga CIFAR-10 una sola vez y extrae
sus features al iniciar.

```bash
# Worker en la misma máquina que el PS
python worker.py

# Worker en otra máquina
python worker.py --server-host 192.168.1.10

# Especificar arquitectura CNN y pesos ImageNet
python worker.py --cnn-arch resnet18 --cnn-pretrained

# Ver todas las opciones
python worker.py --help
```

| Opción | Descripción | Default |
|---|---|---|
| `--server-host` | IP del Parameter Server | `127.0.0.1` |
| `--server-port` | Puerto TCP del Parameter Server | `9999` |
| `--data-dir` | Directorio donde está CIFAR-10 | `Data/` |
| `--hidden1` | Neuronas capa oculta 1 del MLP | `256` |
| `--hidden2` | Neuronas capa oculta 2 del MLP | `128` |
| `--cnn-arch` | Arquitectura CNN: `simple` o `resnet18` | `simple` |
| `--cnn-pretrained` | Usar pesos ImageNet (solo resnet18) | — |
| `--cnn-device` | Dispositivo PyTorch: `cpu`, `cuda`, `mps` | `cpu` |
| `--cnn-seed` | Semilla para inicialización CNN | `42` |
| `--quiet` | Suprime mensajes de progreso | — |

> **Nota:** El Worker debe iniciarse **después** de que el PS esté escuchando.
> El PS bloquea el inicio del entrenamiento hasta que se conecten todos los
> Workers configurados con `--workers`.

### Ejemplo completo (3 terminales en local)

```bash
# Terminal 1 — Parameter Server (CIFAR-10, 2 workers, 50 épocas)
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
> contenedor descarga MNIST automáticamente en su primer arranque.

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
# Terminal 1 — Parameter Server (en el host, CIFAR-10)
python ps_terminal.py --workers 3 --epochs 100

# Terminal 2 — 3 Workers en Docker
.\Docker\run_workers.ps1 -N 3
```

---

## Protocolo de Comunicación

Los mensajes se serializan con **Pickle** y van precedidos de 4 bytes (big-endian)
con la longitud del bloque. Pickle serializa `np.ndarray` de forma binaria nativa,
lo que elimina conversiones y reduce el tamaño en red respecto a JSON.

| Mensaje | Dirección | Descripción |
|---|---|---|
| `READY` | Worker → PS | El Worker se conecta y solicita un ID |
| `WORKER_ID` | PS → Worker | PS asigna un ID único al Worker |
| `TRAIN_START` | PS → Workers | Inicia una sesión; envía `epochs`, `n_train`, `n_workers`, `worker_rank` |
| `PARAMS` | PS → Workers | Pesos globales + semilla de época (el Worker reconstruye su chunk localmente) |
| `GRADIENTS` | Worker → PS | Gradientes calculados sobre el batch asignado |
| `STOP` | PS → Workers | Finaliza la sesión; Workers cierran la conexión |

---

## Arquitectura del Sistema: CNN + MLP Distribuido

La pipeline completa se divide en dos etapas con responsabilidades distintas:

```
┌──────────────────────────────────────────────────────────────┐
│  CIFAR-10 imagen  (3 × 32 × 32) — 50 000 imágenes train      │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────┐                                         │
│  │  CNN Extractor  │  ← PyTorch (pesos FIJOS tras entrenar)  │
│  │  (convolucional)│    Idéntica en todos los Workers        │
│  └────────┬────────┘    Inicialización: semilla reproducible │
│           │  feature vector  (512,)                          │
│           ▼                                                  │
│  ┌─────────────────┐                                         │
│  │   MLP (NumPy)   │  ← Pesos DISTRIBUIDOS (Algoritmo Diego) │
│  │  (clasificador) │    PS: promedia gradientes, actualiza   │
│  └────────┬────────┘    Workers: forw/backward sobre chunks  │
│           │                                                  │
│           ▼                                                  │
│   10 clases CIFAR-10  (avión, auto, pájaro, gato, …)         │
└──────────────────────────────────────────────────────────────┘
```

### Por qué CNN congelada + MLP distribuido

1. **CNN congelada:** Los pesos CNN no cambian durante el entrenamiento distribuido.
   - Beneficio pedagógico: los Workers solo transmiten gradientes del MLP (mucho más pequeño).
   - Práctica industrial real: transfer learning con features fijos.
   - Reproducibilidad: todos los Workers usan exactamente los mismos pesos CNN.

2. **MLP distribuido:** Es donde los Workers difieren en sus datos.
   - Cada Worker recibe los pesos globales + una semilla de época.
   - Reconstruye su chunk localmente (sin transmitir índices por red).
   - Calcula gradientes sobre sus features extraídas.
   - El PS promedia: `∇θ = (1/N) × Σᵢ ∇θᵢ(Bᵢ)`

### Componentes

#### CNN Extractor (Model/cnn_extractor.py)

Dos arquitecturas disponibles:

- **`simple`** (défault): CNN diseñada desde cero.
  - 3 bloques: Conv → BatchNorm → ReLU → MaxPool
  - Salida: 512 features
  - Preentrenamiento local (10 épocas, solo la primera vez)
  - Cacheado: pesos en `Data/feature_cache/` para cargas rápidas

- **`resnet18`**: ResNet-18 de torchvision.
  - Con `--cnn-pretrained`: pesos ImageNet (descarga automática)
  - Sin `--cnn-pretrained`: pesos aleatorios con semilla
  - Salida: 512 features
  - No necesita preentrenamiento

#### MLP (Model/mlp.py)

Red en NumPy puro de dos capas ocultas:

```
Entrada (512 features)
    │
    ▼  W1 (512 × hidden1)
Oculta 1 (hidden1 neuronas, ReLU)
    │
    ▼  W2 (hidden1 × hidden2)
Oculta 2 (hidden2 neuronas, ReLU)
    │
    ▼  W3 (hidden2 × 10)
Salida (10 clases, softmax)
```

- **Forward:** `Z1 = W1·Xᵀ + b1` → `A1 = ReLU(Z1)` → `Z2 = W2·A1 + b2` → `A2 = ReLU(Z2)` → `Z3 = W3·A2 + b3` → `softmax(Z3)`
- **Backward:** retropropagación estándar.
- **Pérdida:** entropía cruzada promediada por muestra.
- **Inicialización:** Xavier para W1, W2, W3; ceros para sesgos.
- **Optimizador:** SGD con momentum opcional (défault: 0.9)

El MLP usa **NumPy puro** porque:
- Los gradientes se serializan con Pickle.
- NumPy arrays son ~3x más pequeños que tensores PyTorch serializados.
- Implementación transparente: se ve exactamente qué se envía por la red.
