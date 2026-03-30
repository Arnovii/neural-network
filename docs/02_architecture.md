# 02. Arquitectura del Sistema

## Componentes de Alto Nivel

El sistema se compone de **cuatro capas funcionales** que interactúan de manera precisa:

```
┌─────────────────────────────────────────────────────────────┐
│  CAPA DE APLICACIÓN                                         │
│  ├─ ps_terminal.py (CLI)  / ps_gui.py (GUI)                 │
│  └─ worker.py (Worker launcher)                             │
├─────────────────────────────────────────────────────────────┤
│  CAPA DE COORDINACIÓN DISTRIBUIDA                           │
│  ├─ Parameter Server (distribuida/parameter_server.py)      │
│  ├─ Worker Node (distribuida/worker_node.py)                │
│  └─ Protocol (distribuida/protocol.py)                      │
├─────────────────────────────────────────────────────────────┤
│  CAPA DE MODELOS                                            │
│  ├─ CNN Extractor (Model/cnn_extractor.py)                  │
│  └─ MLP Classifier (Model/mlp.py)                           │
├─────────────────────────────────────────────────────────────┤
│  CAPA DE DATOS E INFRAESTRUCTURA                            │
│  ├─ Data Loading (Utils/cifar_loader.py)                    │
│  ├─ Caching (Model/cnn_extractor.py + Data/feature_cache)   │
│  └─ Results Export (Utils/results_exporter.py)              │
└─────────────────────────────────────────────────────────────┘
```

---

## Capa de Aplicación

### ps_terminal.py

Punto de entrada **sin interfaz gráfica** para entrenar distribuido.

**Responsabilidades**:
- Parsear argumentos CLI (host, port, workers, epochs, lr, etc.)
- Inicializar CNN (SimpleCNN o ResNet18)
- Crear ParameterServer
- Esperar hasta que se conecten N Workers
- Lanzar sesión de entrenamiento
- Imprimir progreso por época
- Exportar resultados

**No tiene UI**, solo texto. Útil para:
- Experimentos automatizados
- Ejecución en cluster
- Reproducibilidad

**Flujo típico**:
```
$ python ps_terminal.py --epochs 10 --workers 3
[PS] Inicializando CNN (simple)...
[PS] Listening en 0.0.0.0:9999
[PS] Esperando 3 Workers...
(espera 2-5 min mientras se connectan workers)
[PS] READY. Iniciando entrenamiento.
Epoch 1: [████████░░] 87% loss=0.234 accuracy=94.3%
...
```

### ps_gui.py

Punto de entrada **con interfaz gráfica Tkinter** (GUI).

**Responsabilidades**:
- Crear UI en Tkinter con dos paneles (control + visualización)
- Permitir seleccionar modo (PRECOMPUTED vs END-TO-END)
- Permitir seleccionar arquitectura CNN (simple vs resnet18)
- Mostrar conexiones de Workers en tiempo real
- Mostrar gráficos de loss y accuracy (Matplotlib embebido)
- Controlar inicio/parada del servidor
- Exportar resultados a JSON

**Ventajas vs terminal**:
- Visual, fácil para presentaciones
- Ajustar parámetros antes de entrenar
- Ver curvas de convergencia en vivo

**Nota**: ps_gui.py usa threads porque Tkinter no es thread-safe. Toda comunicación con ParameterServer se hace via un Queue para evitar race conditions.

### worker.py

Punto de entrada que inicializa un Worker Node.

**Responsabilidades**:
- Parsear argumentos CLI (server-host, server-port, data-dir, etc.)
- Cargar CIFAR-10 completo en RAM
- Crear WorkerNode
- Conectar al ParameterServer
- Entrar en bucle persistente de recibir instrucciones

**Particularidad**: Se inicia con `--server-host <IP_DEL_PS>`. El resto (ID, CNN weights, etc.) se recibe del PS durante la conexión.

---

## Capa de Coordinación Distribuida

### ParameterServer (parameter_server.py)

Núcleo del sistema distribuido. Mantiene el estado global del entrenamiento.

**Responsabilidades principales**:

#### 1. Aceptación de Conexiones (Hilo de Background)
```
listen() → _accept_thread
    ├─ Abre socket TCP en host:port
    ├─ Accept indefinidamente
    ├─ Asigna IDs secuenciales (0, 1, 2, ...)
    └─ Guarda socket para cada Worker
```

**Invariante**: El PS NUNCA desconecta Workers. Permanecen en el `_worker_sockets` dict incluso entre sesiones de entrenamiento.

#### 2. Sincronización de CNN (Barrera CNN_READY)
```
TRAIN_START → PS envía CNN_WEIGHTS a todos
    ↓
Cada Worker extrae features, envía CNN_READY
    ↓
PS cuenta CNN_READY. Cuando count == n_workers, desbloquea
    ↓
Procede a enviar PARAMS para época 1
```

**Crítico**: Si un Worker no envía CNN_READY, el PS espera indefinidamente (timeout no implementado).

#### 3. Coordinación de Épocas

Por cada época:
```
PS → All Workers: PARAMS (pesos MLP + semilla aleatoria)
Workers (en paralelo, no sincronizado):
    ├─ Extraen features con seed
    ├─ Forward/backward MLP
    └─ Envían GRADIENTS

PS recibe gradientes de Worker 0, 1, 2, ... (en el orden que lleguen)
    ├─ Guarda cada uno
    ├─ Cuando recibe de TODOS los workers o timeout:
    │  ├─ Promedia: grad_avg[k] = sum(grad[i][k] para i=0..n-1) / n
    │  ├─ Actualiza: w[k] -= lr * grad_avg[k]
    │  ├─ Evalúa en test si disponible
    │  └─ Callback on_epoch_end()
    └─ Repite con siguiente época
```

#### 4. Estado de Entrenamiento
```
_training_mode: "precomputed" o "end_to_end"
    ├─ PRECOMPUTED: CNN congelada, features cacheados
    └─ END-TO-END: CNN entrenable, features por época

_cnn: CNNExtractor (compartida, pesos distribuidos a Workers)
_active_training_workers: List[int] - IDs que participan en sesión actual
_epoch_gradients: Dict[worker_id, Dict[param_name, ndarray]] - buffer por época
_epoch_metrics: Dict[worker_id, (loss, accuracy)]
```

#### 5. Callbacks para Notificación Asincrona
- `on_worker_connected(worker_id, addr)`: Worker nuevo se registró
- `on_worker_disconnected(worker_id)`: Worker perdió conexión
- `on_gradients_received(worker_id, epoch, loss, accuracy)`: Gradientes recibidos
- `on_epoch_end(epoch, n_epochs, train_acc, train_loss, test_acc, test_loss)`: Época completada
- `on_worker_joined_late(worker_id, addr)`: Worker intentó conectarse durante entrenamiento

**ps_gui.py consume estos callbacks para actualizar la UI**.

---

### WorkerNode (worker_node.py)

Executor local de cálculos. Sincroniza con ParameterServer.

**Responsabilidades principales**:

#### 1. Conexión Inicial (Handshake)
```
Worker:  → READY (payload vacío)
PS:      → WORKER_ID (e.g., {"worker_id": 2})

Resultado: Worker conoce su ID, PS lo registra
```

#### 2. Ciclo Persistente de Espera
```
while True:
    msg = receive_message(socket)
    
    if msg.type == READY:
        # Solo PS la envía al arrancar, no Workers
        
    if msg.type == CNN_WEIGHTS:
        _handle_cnn_weights(msg)
        # Load CNN, extract features, send CNN_READY
        
    if msg.type == REQUEST_TEST_FEATURES:
        _handle_request_test_features()
        # Enviar features test al PS
        
    if msg.type == TRAIN_SAMPLE:
        _handle_train_sample(msg)
        # Enviar muestra de train raw (para preentrenamiento CNN)
        
    if msg.type == TRAIN_START:
        _run_training_session(epochs, n_train, n_workers, rank)
        # Entrar en bucle de entrenamiento
        
    if msg.type == STOP:
        break
```

#### 3. Sesión de Entrenamiento
```
For each epoch in range(n_epochs):
    msg = receive_message()  # Esperar PARAMS + seed
    
    # Reconstruir índices (distribuidos, determinista)
    indices = _reconstruct_indices(n_train, n_workers, rank, seed)
    X_batch = X_features[indices]
    Y_batch = Y_raw[indices]
    
    # Forward + Backward (NumPy)
    grads, loss, accuracy = mlp.forward_and_gradients(params, X_batch, Y_batch)
    
    # Enviar respuesta
    send_message(GRADIENTS, {"worker_id": my_id, "epoch": e, 
                             "gradients": grads, "loss": loss, "accuracy": acc})
```

**Crítica**: El bucle es **bloqueante**. Si un Worker se queda en `receive_message()` esperando PARAMS, no puede procesar nada más.

#### 4. Partición Determinista de Datos
```
# En el PS: random.seed(epoch_seed); shuffle(range(n_train))
shuffled = np.arange(n_train)
rng = RandomState(epoch_seed)
rng.shuffle(shuffled)

# Dividir round-robin
my_indices = []
for i in range(n_train):
    if i % n_workers == my_rank:
        my_indices.append(shuffled[i])
```

**Invariante**: Con el mismo seed, el Worker 0 siempre obtiene el mismo batch cada época.

---

### Protocol (protocol.py)

Define cómo se comunican PS y Workers.

**Formato de Mensaje**:
```
┌─────────────┬──────────────────────┐
│  4 bytes    │  N bytes             │
│  len(data)  │  pickle.dumps(dict)  │
│  (big-end)  │                      │
└─────────────┴──────────────────────┘

Ejemplo (WORKER_ID message):
  big-endian(4): 0x00 0x00 0x00 0x18  (24 bytes)
  pickle: {"type": "WORKER_ID", "payload": {"worker_id": 2}}
```

**Tipos de Mensaje**:

| Tipo | Sentido | Descripción |
|------|---------|---|
| READY | Worker→PS | Worker declara que está listo |
| WORKER_ID | PS→Worker | Asignación de ID |
| CNN_WEIGHTS | PS→Worker | Pesos CNN para usar |
| CNN_READY | Worker→PS | Confirmó extracción de features |
| REQUEST_TEST_FEATURES | PS→Worker | Me envías features de test |
| TEST_FEATURES | Worker→PS | Respuesta con features test |
| TRAIN_SAMPLE | PS→Worker | Envíame muestra de train |
| TRAIN_SAMPLE_DATA | Worker→PS | Respuesta con muestra |
| TRAIN_START | PS→Worker | Inicia sesión de entrenamiento |
| PARAMS | PS→Worker | Pesos MLP nuevos + seed |
| GRADIENTS | Worker→PS | Gradientes calculados |
| STOP | PS→Worker | Finalizando servidor |

**Decisión de Diseño**: Pickle permite serializar `np.ndarray` sin perder tipos. JSON requeriría convertir a lista (3-5x más grande en red y más lento en parsing).

---

## Capa de Modelos

### CNN Extractor (Model/cnn_extractor.py)

**Opciones de Arquitectura**:

#### SimpleCNN (diseñado desde cero)
```
Conv(3→64, 3×3) → BN → ReLU → MaxPool(2×2)     → 64@16×16
Conv(64→128, 3×3) → BN → ReLU → MaxPool(2×2)  → 128@8×8
Conv(128→256, 3×3) → BN → ReLU → MaxPool(2×2) → 256@4×4
AdaptiveAvgPool() → 256@1×1
FC(256→512) → 512
```

- **Ventajas**: Ligera, rápida en CPU
- **Preentrenamiento**: Necesita datos para entrenar (o usa pesos aleatorios)
- **Feature dim**: 512

#### ResNet18 (torchvision)
```
ImageNet → 18 capas → 512 features
```

- **Ventajas**: Pretrained en ImageNet (pesos transferibles)
- **Desventajas**: Más lenta (18 capas vs 3)
- **Feature dim**: 512

**Responsabilidades de CNNExtractor**:
1. Cargar arquitectura (SimpleCNN o ResNet)
2. Manejar inicialización de pesos (seed o pretrained)
3. Extraer features en batch (forward, no backward)
4. Cachear features con MD5 hash
5. Serializar/deserializar state_dict para transmisión

**Métodos públicos**:
- `extract(X_batch)`: Forward pass, devuelve features (N, 512)
- `extract_with_cache(X, Y, split, device)`: Extract + cachea automáticamente
- `set_trainable(bool)`: Congela/descongela CNN (PRECOMPUTED vs E2E)

---

### MLP Classifier (Model/mlp.py)

**Arquitectura**:
```
Entrada (512)
  ↓ W1 (512×256) + b1 (256,)
  ↓ ReLU
Hidden1 (256)
  ↓ W2 (256×128) + b2 (128,)
  ↓ ReLU
Hidden2 (128)
  ↓ W3 (128×10) + b3 (10,)
  ↓ Softmax
Salida (10)
```

**Funciones públicas**:

#### init_params(...)
- He initialization: $w \sim \mathcal{N}(0, \sqrt{2/fan\_in})$
- Rationale: ReLU satura si W es muy grande; He compensa que mitad de neuronas ≈ 0
- Output: Dict[W1, b1, W2, b2, W3, b3] (todos float32)

#### forward_and_gradients(params, X, Y)
- Forward pass: compute Z1, A1, Z2, A2, A3
- Backward pass: compute δ3, δ2, δ1 (regla cadena)
- Output: (gradients_dict, mean_loss, accuracy_pct)

#### evaluate(params, X, Y)
- Solo forward pass (sin gradientes)
- Output: (accuracy_pct, mean_loss)

#### apply_gradients(params, gradients, lr)
- In-place update: params[k] -= lr * gradients[k]
- Usado por PS después de promediar

**Por qué NumPy y no PyTorch**:
1. Transparencia: cada línea de backward es legible
2. Serialización: PyTorch tensors + graph = 10x más grande que np.ndarray
3. Velocidad: NumPy con BLAS es suficiente para MLP

---

## Capa de Datos e Infraestructura

### Data Loading (Utils/cifar_loader.py)

**Formato Output**:
- X: (N, 3, 32, 32) float32 NCHW normalizado
- Y: (N,) int32 en [0, 9]

**Normalización**:
- μ=[0.4914, 0.4822, 0.4465] (RGB)
- σ=[0.2470, 0.2435, 0.2616]
- Applied per-channel: `X = (X - μ) / σ`

**Cacheado en .npz**: La primera vez carga de torchvision, luego .npz en Data/

### Caching System

**Ubicación**: `Data/feature_cache/`

**Cache key**:
```
{arch}_{weights_md5_first_8_chars}_{split}_X.npy
{arch}_{weights_md5_first_8_chars}_{split}_Y.npy

Ejemplo:
simple_a3f2c8d1_train_X.npy  ← 6667×512 array = ~34 MB
simple_a3f2c8d1_train_Y.npy  ← 6667 array = ~26 KB
```

**Invalidación automática**: Si CNN weights cambian (hash distinto), automáticamente extrae features nuevos (no reutiliza viejo caché).

---

## Flujo de Datos: Estado y Transiciones

### Estado Inicial
- PS escucha, Workers desconectados
- Ningún entrenamiento activo

### Fase 1: Conexión Workers
- Workers envían READY
- PS asigna WORKER_ID
- Workers quedan en `waitinig`

### Fase 2: Sincronización CNN
- PS envía CNN_WEIGHTS
- Workers extraen features (first epoch ~60s, después caché)
- Workers envían CNN_READY
- PS espera barrera (todo N Workers)

### Fase 3: Entrenamiento
- Epochs 1 a N
- Each epoch: PS→PARAMS, Workers→GRADIENTS, PS average+update

### Fase 4: Shutdown
- PS envía STOP
- Workers cierran socket
- PS cierra TCP server

---

## Threading Model

### PS Threads
- **Main thread**: ps_terminal.py / ps_gui.py, llama a ps.train()
- **Accept thread**: Accept conexiones Worker (bloqueante en socket.accept())
- **Hilo GUI** (solo ps_gui.py): Tkinter event loop

**Sincronización**: 
- Mutex `_lock` protege `_worker_sockets`, `_next_id`
- Events para barreras (`_cnn_ready_event`)

### Worker Thread
- **Main thread**: worker.py, bucle bloqueante en `receive_message()`
- **No hay threads adicionales**

### ps_gui.py Threading
- **Main (Tkinter)**: UI loop
- **Comunicación PS**: Via queue para evitar race conditions con Tkinter

---

## Cadena de Responsabilidades

```
ps_terminal.py
    ↓
ParameterServer.train()
    ├─ Inicializar CNN
    ├─ Enviar CNN_WEIGHTS a Workers
    ├─ Esperarrera CNN_READY
    ├─ For each epoch:
    │   ├─ Enviar PARAMS a todos
    │   ├─ Recibir GRADIENTS (bloqueante)
    │   ├─ Promediar
    │   ├─ Actualizar pesos
    │   └─ on_epoch_end()

worker.py
    ↓
WorkerNode.run()
    ├─ _connect() → send READY, receive WORKER_ID
    ├─ _main_loop() → bucle bloqueante recibiendo mensajes
    │   ├─ CNN_WEIGHTS → extract features
    │   ├─ TRAIN_START → _run_training_session()
    │   │   ├─ Receive PARAMS
    │   │   ├─ forward_and_gradients()
    │   │   └─ Send GRADIENTS
    │   └─ STOP → exit
    └─ _disconnect()
```

---

## Puntos de Sincronización Crítica

1. **CNN_READY Barrier**: Ningún PARAMS se envía hasta que TODOS los Workers confirmen extracción
2. **GRADIENTS Collection**: Cada epoch, PS espera recibir UN GRADIENTS de cada Worker antes de actualizar
3. **Seed Sequencing**: Semillas distintas por época evitan patterns repetitivos

---

## Invariantes de Correctness

- ✓ Todos los Workers usan la misma CNN (distribuida por PS)
- ✓ Todos los Workers convergen al mismo modelo global (mismos pesos después de cada update)
- ✓ El entrenamiento es determinista (dado seed fijo)
- ✓ Partición de datos es reproducible round-robin estratificada
- ✓ Gradient averaging = Batch SGD sobre dataset completo

