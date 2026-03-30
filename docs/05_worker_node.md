# 05. Worker Node: Internals y Caching

## Ciclo de Vida Completo

Un Worker tiene tres fases biofísicas:

### Fase 1: Inicialización (Constructor)

```python
worker = WorkerNode(
    server_host="192.168.1.10",
    server_port=9999,
    X_train=(50000, 3, 32, 32),  # CIFAR-10 imágenes
    Y_train=(50000,),
    X_test=(10000, 3, 32, 32),   # datos test
    Y_test=(10000,),
    cnn_device="cuda",            # PyTorch device
    cnn_seed=42,
    hidden1=256, hidden2=128,
    training_mode="precomputed",  # o "end_to_end"
)
```

**Estado después del constructor**:
- `_sock = None` (no conectado)
- `_cnn = CNNExtractor(arch="simple", seed=42)` (pesos aleatorios, serán sobrescritos)
- `_X_raw = X_train` (datos raw en memoria, 50000×3×32×32)
- `_X_features = empty array` (rellenado después de CNN_WEIGHTS)
- `_class_indices = [where(Y==0), where(Y==1), ..., where(Y==9)]` (precalculado)

**Memoria utilizada**: ~600 MB (50000 imágenes × 3×32×32×float32 ≈ 600 MB)

### Fase 2: Conexión al PS (worker.run())

```python
worker.run()  # Punto de entrada
    ├─ worker._connect()
    │   ├─ socket.socket(AF_INET, SOCK_STREAM)
    │   ├─ socket.connect(server_host, server_port)  # TCP connect
    │   ├─ send_message(MsgType.READY, {})          # "I'm ready"
    │   └─ msg = receive_message()                   # Wait for WORKER_ID
    │       └─ worker.worker_id = msg["worker_id"]  # e.g., 2
    │
    ├─ worker._log(f"Connected as Worker {worker_id}")
    └─ worker._main_loop()
```

**Después de _connect**:
- `worker_id` asignado (e.g., 2)
- Socket configurado, conectado al PS
- Listo para recibir mensajes

### Fase 3: Bucle Principal (wait-for-messages)

```python
def _main_loop(self):
    while True:
        msg = receive_message(self._sock)  # BLOQUEANTE
        
        if msg["type"] == MsgType.STOP:
            break
        
        elif msg["type"] == MsgType.CNN_WEIGHTS:
            self._handle_cnn_weights(msg["payload"])
        
        elif msg["type"] == MsgType.REQUEST_TEST_FEATURES:
            self._handle_request_test_features()
        
        elif msg["type"] == MsgType.TRAIN_SAMPLE:
            self._handle_train_sample(msg["payload"])
        
        elif msg["type"] == MsgType.TRAIN_START:
            self._run_training_session(...)
```

**Punto crítico**: El bucle es **bloqueante**. Si el Worker se queda en `receive_message()`, puede esperar indefinidamente. Si el PS envía un sleep(1000), el Worker se duerme 1 segundo esperandorespuesta.

---

## Manejo de CNN_WEIGHTS

Cuando el PS envía `CNN_WEIGHTS`, el Worker:

1. **Deserializa** los pesos (torch.load from bytes)
2. **Carga en la CNN local** (state_dict.load)
3. **Extrae features** de todo el dataset de train (con caché)
4. **Confirma al PS** con CNN_READY

```python
def _handle_cnn_weights(self, payload: dict) -> None:
    """
    PS envió los pesos de CNN. El Worker los carga, extrae features,
    y confirma con CNN_READY.
    """
    arch = payload["arch"]
    weights_bytes = payload["weights_bytes"]
    
    # 1. Deserializar
    weights_dict = torch.load(BytesIO(weights_bytes))
    
    # 2. Cargar en CNN
    self._cnn = CNNExtractor(arch=arch, device=self._cnn.device)
    self._cnn.load_state_dict(weights_dict)
    self._cnn.set_trainable(training_mode == "end_to_end")
    
    # 3. Extraer features del COMPLETO dataset
    X_features, _ = self._load_features_with_cache(
        self._X_raw,
        self._Y_raw,
        arch=arch,
        batch_size=self._optimal_batch_size(),
        split="train",  # guardar con key="train"
    )
    
    # 4. Guardar para uso en training
    self._X_features = X_features  # (50000, 512)
    
    # 5. Confirmar al PS
    send_message(self._sock, MsgType.CNN_READY, {"worker_id": self.worker_id})
```

### Cálculo del Batch Size Óptimo

El Worker adapta el batch size según arquitectura CNN y dispositivo:

```python
def _optimal_batch_size(self) -> int:
    """
    Calcula batch size adaptativo para extracción CNN.
    
    Trade-off: grande = fast pero memory-intensive, 
               pequeño = lento pero safe.
    """
    device_type = str(self._cnn.device).split(":")[0]  # cuda/cpu/mps
    arch = self._cnn.arch  # simple vs resnet18
    n_cpus = os.cpu_count()
    
    if arch == "resnet18":
        # ResNet es pesada (18 capas)
        if device_type == "cpu":
            return 64  # muy conservador
        elif device_type == "cuda":
            return 256  # GPU es más forgiving
        else:
            return 128
    else:
        # SimpleCNN es más ligera
        if device_type == "cpu":
            return 512
        elif device_type == "cuda":
            return 2048  # pueder subir más
        else:
            return 1024
```

**Razón**: CNN forward pass es O(batch × depth × H × W). ResNet18 es ~18x más profunda que SimpleCNN → necesita batch 4x más pequeño.

---

## Sistema de Caché Inteligente

### Función: _load_features_with_cache()

```python
def _load_features_with_cache(
    self,
    X: np.ndarray,        # (n_samples, 3, 32, 32)
    Y: np.ndarray,        # (n_samples,)
    arch: str,            # "simple" o "resnet18"
    batch_size: int,
    split: str,           # "train" o "test"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae features con caché automático e inteligente.
    
    Lógica:
    1. Calcular hash MD5 de pesos CNN actuales
    2. Construir cache key: {arch}_{hash8}_{split}_{X|Y}.npy
    3. ¿Existe en disco? → cargar (< 0.5s)
    4. ¿No? → CNN forward (primer uso o pesos cambiaron)
    5. Guardar al disco para siguiente acceso
    """
```

### Flujo Detallado

```
Cache Query:
├─ weight_hash = md5(cnn.state_dict())[:8]  # e.g., "a3f2c8d1"
├─ cache_key = f"simple_a3f2c8d1_train"
├─ cache_path_X = f"Data/feature_cache/{cache_key}_X.npy"
├─ cache_path_Y = f"Data/feature_cache/{cache_key}_Y.npy"
│
├─ ¿Existen los archivos?
│  ├─ SÍ (PRECOMPUTED con CNN congelada):
│  │  ├─ X_feat = np.load(cache_path_X)
│  │  ├─ Y = np.load(cache_path_Y)
│  │  └─ Tiempo: 0.1-0.3s (disk I/O)
│  │
│  └─ NO (primera ejecución o CNN cambió):
│     ├─ Para each batch en X:
│     │  └─ X_batch_feat = cnn.extract(X_batch)
│     ├─ Concatenar: X_features = [X_batch_feat1, X_batch_feat2, ...]
│     ├─ np.save(cache_path_X, X_features)
│     ├─ np.save(cache_path_Y, Y)
│     └─ Tiempo: 3-30s (CNN forward, depende de arch)
│
└─ Devolver X_features
```

### Caché Key: Hash y Invalidación

**Requerimiento**: Si pesos CNN cambian, debe recalcualarse features. Hash automáticamente invalida.

**Ejemplo**: 
- Época 0 (PRECOMPUTED): CNN pesos "A3F2..." → features cacheados "simple_a3f2_train_X.npy"
- Época 1-N: CNN congelada → hash "A3F2..." (constante) → reutiliza caché
- Vs END-TO-END: CNN pesos cambian cada época → hash distinto → siempre re-extrae

**Ventaja sobre manualmente verificar**: 
- Dev no necesita trackear "¿cambié los pesos?"
- Sistema lo detecta automáticamente

---

## Reconstrucción Determinista de Indices

Cuando el Worker recibe PARAMS, el payload incluye una semilla. Con esa semilla, **reconstruye sus índices de forma determinista**:

```python
def _reconstruct_indices(
    self,
    n_train: int,        # 50000
    n_workers: int,      # 3
    my_rank: int,        # 0, 1, or 2
    seed: int,           # e.g., 45
) -> np.ndarray:
    """
    Reconstruye los índices que este Worker debe procesar en esta época.
    
    Todos los Workers con el MISMO seed y rank obtienen EXACTAMENTE
    los mismos índices (determinismo). Esta es la clave para que el
    entrenamiento sea reproducible sin transmitir indices por red.
    """
    rng = np.random.RandomState(seed)
    
    # 1. Permutar indices de manera determinista
    shuffled = np.arange(n_train)
    rng.shuffle(shuffled)  # [3124, 18734, 202, ...] (determinista dado seed)
    
    # 2. Dividir round-robin: Worker i toma elementos donde idx % n_workers == i
    my_indices = shuffled[my_rank::n_workers]
    
    # Resultado:
    # Worker 0: [shuffled[0], shuffled[3], shuffled[6], ...]  (16667 elementos)
    # Worker 1: [shuffled[1], shuffled[4], shuffled[7], ...]
    # Worker 2: [shuffled[2], shuffled[5], shuffled[8], ...]
    
    return my_indices
```

**Ejemplo concreto**:
```
n_train = 50000, n_workers = 3, seed = 45
shuffled = [3124, 18734, 202, 999, 25000, 1, 14111, ...]  (50000 elementos)

Worker 0 (rank=0): [3124, 999, 14111, ...]     (0, 3, 6, 9, ...)
Worker 1 (rank=1): [18734, 25000, ...]         (1, 4, 7, 10, ...)
Worker 2 (rank=2): [202, 1, ...]               (2, 5, 8, 11, ...)
```

**Invariante**: Ejecutar dos veces con seed=45 → Worker 0 obtiene identicos indices → identicas muestras → identicos gradientes (hasta float precision).

---

## Manejo de Estratificación (Clases Balanceadas)

Al construir, el Worker precalcula índices por clase:

```python
# En constructor
self._class_indices = [
    np.where(Y_train == digit)[0]
    for digit in range(10)
]

# Result:
# _class_indices[0] = [3, 15, 27, 100, ...]  (todos los índices Y==0)
# _class_indices[1] = [4, 18, 32, 102, ...]  (todos los índices Y==1)
# ...
# _class_indices[9] = [8, 100, 255, ...]
```

**Uso potencial**: Si necesitases estratificación (garantizar que cada batch tiene todas las clases → evitar 1-sample batches con Y=[0,0,0,0,...]), podrías usar:

```python
batch_per_class = n_batch // n_classes  # e.g., 100 // 10 = 10
indices_batch = []
for class_id in range(10):
    sample_indices = rng.choice(
        self._class_indices[class_id],
        size=batch_per_class,
        replace=True
    )
    indices_batch.extend(sample_indices)

rng.shuffle(indices_batch)  # (opcional, para no tener clase-grupos)
```

**Realmente implementado**: Actualmente el código USA round-robin simple (no estratificado). Pero la estructura está lista.

---

## Mini-batching en End-To-End

En END-TO-END, el Worker extrae features en **mini-batches** (no completo) para no saturar memoria:

```python
def _load_features_with_cache(...):
    X_features_list = []
    
    # Dividir en mini-batches para CNN
    for batch_start in range(0, len(X), batch_size):
        batch_end = min(batch_start + batch_size, len(X))
        X_batch = X[batch_start:batch_end]
        
        # CNN forward sobre mini-batch
        X_batch_features = cnn.extract(X_batch)
        X_features_list.append(X_batch_features)
    
    # Concatenar todos
    X_features = np.concatenate(X_features_list, axis=0)
    
    # Guardar caché
    np.save(cache_path_X, X_features)
```

**Por qué**: Forward 50000 imágenes de golpe en GPU = 50000×256×32×32×4 bytes ≈ 51 GB (no cabe). En mini-batches: 2048×256×32×32×4 ≈ 2 GB (manejable).

---

## Sincronización de training_mode

El TRAIN_START message incluye `training_mode`. El Worker lo sincroniza:

```python
def _main_loop(self):
    ...
    elif msg["type"] == MsgType.TRAIN_START:
        payload = msg["payload"]
        
        # Sincronizar mode desde PS
        if "training_mode" in payload:
            new_mode = payload["training_mode"]
            if new_mode != self.training_mode:
                print(f"Sync mode: {self.training_mode} → {new_mode}")
                self.training_mode = new_mode
        
        # Ahora sí, entrenar con el mode correc
        self._run_training_session(...)
```

**Razón**: 

Después de una sesión PRECOMPUTED, pueden cambiar a END-TO-END sin reiniciar Workers. 

El PS envía `{"training_mode": "end_to_end"}` en TRAIN_START del siguiente entrenamiento.

---

## Manejo de Errores y Timeouts

**No implementado** (debería estarlo):

```python
# IDEAL (no en código actual):
def _main_loop(self):
    timeout_seconds = 300  # 5 min
    
    while True:
        msg = receive_message(self._sock, timeout=timeout_seconds)
        
        if msg is None:
            # Timeout: PS no envió nada en 5 min
            print("Timeout esperando PS. Desconectando.")
            break
        
        # Procesar msg
```

**Actual**: Sin timeout. Si PS falla, Worker se queda esperando indefinidamente.

---

## Debugging y Logging

El Worker usa un logger con colores:

```python
_logger.worker(f"Conectado como Worker {worker_id}")
_logger.worker(f"PRECOMPUTED mode, features cacheados: {self._X_features.shape}")
_logger.worker(f"Epoch {epoch}: loss={loss:.4f} accuracy={accuracy:.2f}%")
```

**Salida típica**:
```
[WORKER 2] Connected to PS as ID=2
[WORKER 2] CNN features extracted: (16667, 512)
[WORKER 2] Mode synchronized: precomputed
[WORKER 2] Epoch 0: loss=0.328 accuracy=91.2%
[WORKER 2] Sending GRADIENTS...
[WORKER 2] Waiting for next epoch...
```

