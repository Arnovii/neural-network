# Red Neuronal con Algoritmo de Diego para MNIST

Proyecto de redes neuronales que implementa el **Algoritmo de Diego** (promediado de gradientes por época) sobre el dataset MNIST. Incluye dos modos de ejecución independientes:

- **Local** — experimentos en una sola máquina con paralelismo de hilos (`main.py`).
- **Distribuido** — entrenamiento real sobre varias máquinas con Parameter Server y Workers conectados por TCP (`ps_terminal.py` / `ps_gui.py` + `worker.py`).

---

## Estructura del Proyecto

```
neural-network/
├── Analytics/               # Motor de experimentos, estadísticas y gráficas
│   ├── experiment_runner.py
│   ├── statistics_engine.py
│   └── chart_generator.py
├── Data/
│   └── MNIST/raw/           # Dataset MNIST (se descarga automáticamente)
├── Distributed/             # Núcleo del sistema distribuido
│   ├── parameter_server.py  # Clase ParameterServer (lógica TCP + entrenamiento)
│   ├── worker_node.py       # Clase WorkerNode (forward + backward + gradientes)
│   └── protocol.py          # Serialización de mensajes JSON sobre TCP
├── Networks/
│   └── nn_diego.py          # Red neuronal (álgebra lineal pura, sin frameworks)
├── Parallel/
│   ├── worker.py            # Worker local (hilo)
│   └── core_validator.py
├── Results/                 # JSONs de experimentos anteriores
├── Utils/
│   ├── math_utils.py        # Xavier, softmax, sigmoid, etc.
│   ├── mnist_loader.py      # Carga y descarga de MNIST
│   └── data_partitioner.py
├── main.py                  # Modo local: experimentos con paralelismo de hilos
├── ps_terminal.py           # Modo distribuido: Parameter Server (terminal)
├── ps_gui.py                # Modo distribuido: Parameter Server (GUI Tkinter)
├── worker.py                # Modo distribuido: Worker Node
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

Dependencias principales: `numpy`, `matplotlib`, `mplcursors`, `psutil`.  
MNIST se descarga automáticamente en `Data/MNIST/raw/` al primer uso.

---

## Modo 1 — Local (una sola máquina)

Ejecuta el Algoritmo de Diego usando hilos locales como "workers" paralelos.

### Terminal

```bash
# Experimento básico: 2 particiones, 5 épocas, 10 repeticiones
python main.py --partitions 2 --epochs 5 --experiments 10

# Configuración extendida
python main.py --partitions 3 --epochs 10 --experiments 5 \
               --hidden-neurons 50 --learning-rate 0.5 --n-train 8000

# Ver todas las opciones
python main.py --help
```

| Opción | Descripción | Default |
|---|---|---|
| `-p`, `--partitions` | Número de particiones (workers locales) | 2 |
| `-e`, `--epochs` | Épocas de entrenamiento | 10 |
| `-x`, `--experiments` | Repeticiones del experimento | 5 |
| `--hidden-neurons` | Neuronas en la capa oculta | 30 |
| `--learning-rate` | Tasa de aprendizaje | 0.1 |
| `--n-train` | Ejemplos de entrenamiento a usar | 10 000 |

### Interfaz Gráfica (GUI)

```bash
python main.py --interactive
```

---

## Modo 2 — Distribuido (varias máquinas)

El Parameter Server (PS) gestiona los pesos globales. Cada Worker carga MNIST
localmente, recibe índices del PS, calcula gradientes sobre su batch y los
devuelve. El PS promedia los gradientes y actualiza los pesos.

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
                      1. Divide índices 0..n_train en N chunks disjuntos.
                      2. Broadcast: envía params + índices a cada Worker.
                      3. Espera gradientes de TODOS los Workers (barrera).
                      4. Promedia: ∇θ = (1/N) × Σ ∇θᵢ
                      5. Actualiza: θ ← θ − lr × ∇θ

shutdown()      → Envía STOP a todos los Workers y cierra el socket.
```

### Lanzar el servidor (terminal)

```bash
# Configuración básica: esperar 2 workers, 10 épocas
python ps_terminal.py

# Configuración extendida
python ps_terminal.py --workers 3 --epochs 20 --hidden 64 --lr 0.05 --n-train 60000
```

| Opción | Descripción | Default |
|---|---|---|
| `--host` | IP de escucha del servidor | `0.0.0.0` |
| `--port` | Puerto TCP | `9999` |
| `--workers` | Número de Workers a esperar antes de entrenar | `2` |
| `--epochs` | Épocas de entrenamiento | `10` |
| `--hidden` | Neuronas en la capa oculta | `30` |
| `--lr` | Tasa de aprendizaje | `0.1` |
| `--n-train` | Total de ejemplos de entrenamiento | `10 000` |
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
especificarlo manualmente.

```bash
# Worker en la misma máquina que el PS
python worker.py

# Worker en otra máquina
python worker.py --server-host 192.168.1.10

# Ver todas las opciones
python worker.py --help
```

| Opción | Descripción | Default |
|---|---|---|
| `--server-host` | IP del Parameter Server | `127.0.0.1` |
| `--server-port` | Puerto TCP del Parameter Server | `9999` |
| `--data-dir` | Directorio donde está MNIST | `Data/` |
| `--quiet` | Suprime mensajes de progreso | — |

> **Nota:** El Worker debe iniciarse **después** de que el PS esté escuchando.
> El PS bloquea el inicio del entrenamiento hasta que se conecten todos los
> Workers configurados con `--workers`.

### Ejemplo completo (3 terminales en local)

```bash
# Terminal 1 — Parameter Server
python ps_terminal.py --workers 2 --epochs 15

# Terminal 2 — Worker 0
python worker.py

# Terminal 3 — Worker 1
python worker.py
```

---

## Protocolo de Comunicación

Los mensajes se transmiten como JSON precedidos de 4 bytes (big-endian) con la
longitud del mensaje. Los arrays NumPy se codifican como objetos JSON con
`__ndarray__: true`.

| Mensaje | Dirección | Descripción |
|---|---|---|
| `READY` | Worker → PS | El Worker se conecta y solicita un ID |
| `WORKER_ID` | PS → Worker | PS asigna un ID único al Worker |
| `TRAIN_START` | PS → Workers | Inicia una sesión de entrenamiento |
| `PARAMS` | PS → Workers | Pesos globales + índices del batch para la época |
| `GRADIENTS` | Worker → PS | Gradientes calculados sobre el batch asignado |
| `STOP` | PS → Workers | Finaliza la sesión; Workers cierran la conexión |

---

## Arquitectura de la Red Neuronal

Red totalmente conectada de dos capas:

```
Entrada (784) → Oculta (hidden, σ) → Salida (10, softmax)
```

- **Forward:** `Z1 = W1·Xᵀ + b1` → `A1 = σ(Z1)` → `Z2 = W2·A1 + b2` → `A2 = softmax(Z2)`
- **Backward:** gradientes estándar por retropropagación.
- **Inicialización:** Xavier para W1 y W2; ceros para b1 y b2.
- **Implementación:** álgebra lineal pura con NumPy (sin PyTorch ni TensorFlow).