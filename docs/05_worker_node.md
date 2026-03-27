# 5. WORKER NODE — ARQUITECTURA Y CICLO DE VIDA

## Ciclo de vida del Worker

```
┌────────────────────────────────────────────────────────────────┐
│               WORKER NODE LIFECYCLE                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ 1. CONSTRUCCIÓN                                                │
│    Worker(server_host, port, X_train, Y_train, ...)            │
│    └─ Cargar 50K imágenes en RAM                               │
│    └─ Inicializar CNN placeholder                              │
│    └─ Pre-calcular índices por clase                           │
│                                                                │
│ 2. CONEXIÓN                                                    │
│    worker.run()                                                │
│    └─ Conectar TCP al PS                                       │
│    └─ Enviar READY, recibir WORKER_ID                          │
│                                                                │
│ 3. CARGA DE CNN                                                │
│    ◄─ CNN_WEIGHTS de PS                                        │
│    └─ Cargar pesos                                             │
│    └─ Congelar (precomputed) O habilitar (E2E)                 │
│    └─ Extraer features (precomputed) O nada (E2E)              │
│    └─ Enviar CNN_READY                                         │
│                                                                │
│ 4. SINCRONIZACIÓN                                              │
│    └─ Esperar TRAIN_START                                      │
│                                                                │
│ 5. LOOP DE ENTRENAMIENTO                                       │
│    ├─ Recibir PARAMS + seed                                    │
│    ├─ Reconstruir índices                                      │
│    ├─ Calcular gradientes (forward + backward)                 │
│    ├─ Enviar GRADIENTS                                         │
│    └─ Repetir (N épocas)                                       │
│                                                                │
│ 6. FIN DE SESIÓN                                               │
│    ├─ Volver a paso 4 (esperar siguiente TRAIN_START)          │
│    O                                                           │
│    └─ Recibir STOP → desconectar                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Estructura de datos del Worker

```python
class WorkerNode:
    # Configuración
    server_host: str              # IP del PS (ej. "192.168.1.100")
    server_port: int              # Puerto TCP (ej. 9999)
    worker_id: int                # Asignado por PS (0, 1, 2, …)
    training_mode: str            # "precomputed" o "end_to_end"
    
    # Datos de entrenamiento
    _X_raw: np.ndarray            # (50000, 3, 32, 32) — imágenes raw
    _Y_raw: np.ndarray            # (50000,) — etiquetas
    _X_features: np.ndarray       # (50000, 512) o empty — features cacheados
    _Y_train: np.ndarray          # Copia de _Y_raw
    
    # Datos de prueba (solo si proporcionados)
    _X_test: np.ndarray           # (10000, 3, 32, 32) — imagenes test
    _Y_test: np.ndarray           # (10000,) — etiquetas test
    
    # Modelo CNN
    _cnn: CNNExtractor            # Extractor de features
    
    # Parámetros MLP (recibidos del PS)
    _mlp_params: Dict            # {W1, b1, W2, b2, W3, b3}
    
    # Índices por clase (precalculados)
    _class_indices: List[np.ndarray]
                                  # [0] = índices donde Y==0
                                  # [1] = índices donde Y==1
                                  # …
                                  # [9] = índices donde Y==9
    
    # Socket TCP
    _sock: socket.socket          # Conexión al PS
```

---

## Reconstrucción de índices (stratified round-robin)

### **Propósito**

Cada Worker obtiene el mismo conjunto de índices si usa el mismo seed, pero distribuidos de forma balanceada.

### **Algoritmo**

```python
def _reconstruct_indices(self, seed, n_train, n_workers, worker_rank):
    """
    Retorna ~25% de los índices (12500 imgs de las 50000 totales).
    
    Invariante: cada Worker obtiene imgs de TODAS las clases.
    Estrategia: round-robin dentro de cada clase.
    
    Ejemplo (2 workers, 5000 imgs por clase):
    Clase 0: [0, 1, 2, 3, 4, …, 4999]
    
    Mezcla con seed:
    Clase 0 mezclada: [42, 1001, 3, 4502, …]
    
    Round-robin por rank:
    Worker 0 obtiene: [42, 3, 1, …]      (posiciones 0, 2, 4, …)
    Worker 1 obtiene: [1001, 4502, …]   (posiciones 1, 3, 5, …)
    
    Resultado:
    Worker 0: [42, 3, 1, …] + [otros indices clase 1] + … → 12500 total
    Worker 1: [1001, 4502, …] + [otros indices clase 1] + … → 12500 total
    
    Garantías:
    - Ambos workers obtienen datos de clase 0, 1, …, 9
    - Balanceo perfecto (50-50)
    - Determinístico (mismo seed → mismos índices)
    """
    rng = np.random.RandomState(seed)
    n_per_class = n_train // 10  # 5000 por clase
    
    all_indices = []
    for class_label in range(10):
        class_indices = self._class_indices[class_label]
        shuffled = rng.permutation(class_indices)
        
        # Distribuir en round-robin
        my_indices = shuffled[worker_rank::n_workers]
        all_indices.extend(my_indices)
    
    return np.array(all_indices)
```

### **Ejemplo numérico**

```
Total: 50000 imágenes
Clases: 10
Por clase: 5000 imágenes

Worker 0 (rank=0), Worker 1 (rank=1), n_workers=2

Clase 0:
  Índices: [0, 1, 2, 3, 4, …, 4999]
  Seed=42 mezcla: [3241, 102, 4501, 40, …]
  Worker 0 obtiene (posiciones 0, 2, 4, …): [3241, 4501, …] ← 2500
  Worker 1 obtiene (posiciones 1, 3, 5, …): [102, 40, …] ← 2500

Clase 1:
  Índices: [5000, 5001, …, 9999]
  Seed=42 mezcla: [7401, 5123, 9211, …]
  Worker 0 obtiene: [7401, 9211, …] ← 2500
  Worker 1 obtiene: [5123, …] ← 2500

… (repite para clases 2-9)

Total por worker: 50000 / 2 = 25000
Total de ambos: 50000
Cobertura: ambos workers ven todas las clases
```

---

## Méthodos principales

### **run()**
```python
def run(self) -> None:
    """
    Punto de entrada. Conecta, carga CNN, entra en loop persistente.
    """
    self._connect()          # READY → WORKER_ID
    self._main_loop()        # Espera mensajes indefinidamente
    self._disconnect()       # Cierra conexión
```

### **_connect()**
```python
def _connect(self) -> None:
    """
    Envía READY, recibe WORKER_ID, guarda ID y socket.
    """
    self._sock = socket.socket()
    self._sock.connect((self.server_host, self.server_port))
    
    # Handshake
    send_message(self._sock, MsgType.READY, {})
    msg = receive_message(self._sock)
    self.worker_id = msg["payload"]["worker_id"]
```

### **_main_loop()**
```python
def _main_loop(self) -> None:
    """
    Loop persistente: espera mensajes del PS indefinidamente.
    
    Maneja:
    - CNN_WEIGHTS: cargar CNN, extraer features (precomp) o habilitar (E2E)
    - TRAIN_START: inicia sesión de N épocas
    - STOP: terminar
    """
    while True:
        msg = receive_message(self._sock)
        
        if msg["type"] == MsgType.STOP:
            break
        elif msg["type"] == MsgType.CNN_WEIGHTS:
            self._handle_cnn_weights(msg["payload"])
        elif msg["type"] == MsgType.TRAIN_START:
            self._run_training_session(msg["payload"])
```

### **_handle_cnn_weights()**
```python
def _handle_cnn_weights(self, payload) -> None:
    """
    Procesa CNN_WEIGHTS del PS:
    1. Carga pesos CNN
    2. Congelica (precomputed) O habilita (E2E)
    3. Extrae features (precomputed) — con caché
    4. Envía CNN_READY
    
    PRECOMPUTED:
    ├─ set_trainable(False)
    ├─ Extract + cache features = [CACHE HIT] ~0.5s o [CACHE MISS] ~30-60s
    └─ CNN_READY
    
    END-TO-END:
    ├─ set_trainable(True)
    ├─ NO extraer features
    └─ CNN_READY
    """
    arch = payload["arch"]
    weights_bytes = payload["weights_bytes"]
    
    # Reconstruir si arch cambió
    if self._cnn.arch != arch:
        self._cnn = CNNExtractor(arch=arch, device=self._cnn.device)
    
    # Cargar pesos
    self._cnn.load_weights_from_bytes(weights_bytes)
    
    if self.training_mode == "precomputed":
        # Rama PRECOMPUTED
        self._cnn.set_trainable(False)
        # Extraer con caché inteligente
        X_feat, Y = self._load_features_with_cache(
            self._X_raw, self._Y_raw,
            arch, batch_size=2048
        )
        self._X_features = X_feat
    else:
        # Rama END-TO-END
        self._cnn.set_trainable(True)
        self._X_features = np.empty((0,))  # Placeholder
    
    # Confirmar
    send_message(self._sock, MsgType.CNN_READY, {"worker_id": self.worker_id})
```

### **_run_training_session()**
```python
def _run_training_session(self, payload) -> None:
    """
    Ejecuta una sesión de N épocas.
    Por cada época: recibe PARAMS, calcula gradientes, envía GRADIENTS.
    """
    epochs = payload["epochs"]
    n_train = payload["n_train"]
    n_workers = payload["n_workers"]
    worker_rank = payload["worker_rank"]
    
    for _ in range(epochs):
        msg = receive_message(self._sock)
        if msg["type"] == MsgType.PARAMS:
            self._handle_params(msg["payload"], n_train, n_workers, worker_rank)
```

### **_handle_params()**
```python
def _handle_params(self, payload, n_train, n_workers, worker_rank) -> None:
    """
    Procesa PARAMS de una época:
    1. Reconstruir índices con seed
    2. Cargar datos (indexar features o X_raw)
    3. Forward CNN + MLP
    4. Backward MLP (+ CNN si E2E)
    5. Enviar GRADIENTS
    """
    epoch = payload["epoch"]
    mlp_params = payload["params"]
    seed = payload["seed"]
    cnn_params = payload.get("cnn_params")  # None en precomputed
    
    # Reconstruir índices
    indices = self._reconstruct_indices(seed, n_train, n_workers, worker_rank)
    
    if self.training_mode == "precomputed":
        # RAMA PRECOMPUTED: features ya cacheados
        X_batch = self._X_features[indices]
        Y_batch = self.Y_train[indices]
        gradients, loss, acc = forward_and_gradients(mlp_params, X_batch, Y_batch)
        
        send_message(self._sock, MsgType.GRADIENTS, {
            "gradients": gradients,
            "cnn_gradients": None
        })
    else:
        # RAMA END-TO-END: calcular features on-the-fly
        X_raw_batch = self._X_raw[indices]
        Y_batch = self.Y_train[indices]
        
        # Cargar CNN con pesos actualizados
        self._cnn.load_weights_from_bytes(cnn_params)
        
        # Forward CNN + MLP + backward
        X_feat_batch = self._cnn.forward(X_raw_batch)
        gradients, loss, acc = forward_and_gradients(mlp_params, X_feat_batch, Y_batch)
        cnn_gradients = self._cnn.backward(loss)
        
        send_message(self._sock, MsgType.GRADIENTS, {
            "gradients": gradients,
            "cnn_gradients": cnn_gradients
        })
```

---

## Optimizaciones principales

### **1. Batch size adaptativo**

```python
def _optimal_batch_size(self) -> int:
    """
    Calcula batch size dinámicamente según:
    - Arquitectura CNN (simple vs resnet18)
    - Dispositivo (CPU vs GPU)
    - CPUs disponibles
    
    Objetivo: No congelar el proceso, aprovechar paralelismo.
    
    Heurística:
    - Simple + CPU: 512 (ligero, muchas muestras)
    - Simple + GPU: 2048 (más parallelismo)
    - ResNet18 + CPU: 64 (pesado, pocas muestras)
    - ResNet18 + GPU: 256 (moderado)
    """
    device_type = str(self._cnn.device).split(":")[0]
    arch = self._cnn.arch
    
    if arch == "resnet18":
        return 64 if device_type == "cpu" else 256
    else:  # simple
        return 512 if device_type == "cpu" else 2048
```

### **2. Caché de features con hash**

```python
def _load_features_with_cache(self, X, Y, arch, batch_size, split="train"):
    """
    Extrae features con caché MD5:
    
    1. Calcula hash de pesos CNN actuales
    2. Mira en Data/feature_cache/{arch}_{hash}_{split}_X.npy
    3. Si existe y shape es correcto: CACHE HIT (~0.5s)
    4. Si no existe: CACHE MISS → extrae (~30-60s) y guarda
    """
    weights_hash = self._cnn._weights_hash()  # MD5 de W
    cache_key = f"{arch}_{weights_hash}_{split}"
    cache_X_path = f"Data/feature_cache/{cache_key}_X.npy"
    
    if os.path.exists(cache_X_path):
        X_feat = np.load(cache_X_path)
        if X_feat.shape == (len(X), 512):  # Validar
            return X_feat, Y
    
    # CACHE MISS: extraer
    X_feat = self._cnn.extract_batched(X, batch_size)
    np.save(cache_X_path, X_feat)
    return X_feat, Y
```

### **3. Indices pre-calculados por clase**

```python
# En init: pre-calcular una sola vez
self._class_indices = [
    np.where(Y_train == digit)[0] for digit in range(10)
]

# En cada época: reutilizarlos
indices = self._reconstruct_indices(seed)  # O(10 * len(clase) / 10) = O(n_train)
```

---

## 🔌 Manejo de mensajes del Worker

```
MENSAJE             RESPUESTA                    ACCIÓN
────────────────────────────────────────────────────────────────
CNN_WEIGHTS    ──►  CNN_READY         Cargar CNN, extraer/habilitar
TRAIN_START    ──►  (inicia loop)     Comienza PARAMS/GRADIENTS
PARAMS         ──►  GRADIENTS         Calcula gradientes
REQUEST_TEST   ──►  TEST_FEATURES     Envía features test
STOP           ──►  (desconexión)     Cierra conexión
```

---

## Puntos clave del Worker

1. **Persistente**: No se desconecta entre épocas/sesiones
2. **Autónomo**: Reconstruye índices localmente — no confía en lista del PS
3. **Agnóstico a red**: Carga datos locales, solo envía gradientes (60 KB o 50 MB)
4. **Simétrico**: Todos los Workers ejecutan exactamente el mismo código
5. **Determinístico**: Mismo seed → mismos índices (reproducible)

---

**Documento**: `docs/05_worker_node.md`  
**Última actualización**: 2026-03-27  
**Nivel**: Intermedio → Avanzado
