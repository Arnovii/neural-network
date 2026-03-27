# 6. PARAMETER SERVER — ARQUITECTURA Y SINCRONIZACIÓN

## 🎯 Rol central

El Parameter Server es el orquestador centralizado. **Nunca ve datos de entrenamiento locales**, pero coordina:
- Sincronización entre Workers
- Distribución de modelos (CNN + MLP)
- Promediación de gradientes
- Evaluación en datos de prueba

---

## 🔄 Ciclo de vida del PS

```
┌────────────────────────────────────────────────────────────────┐
│                PS LIFECYCLE                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ 1. CONSTRUCCIÓN                                                │
│    PS(host="0.0.0.0", port=9999)                              │
│    └─ Inicializar estructuras (dicts de workers)              │
│    └─ Preparar sockets TCP                                    │
│                                                                │
│ 2. ESCUCHA (LISTENING STATE)                                  │
│    ps.listen()                                                 │
│    └─ Abre servidor TCP                                       │
│    └─ Hilo de fondo acepta Workers indefinidamente           │
│    └─ Workers se conectan con READY                          │
│    └─ PS asigna IDs (0, 1, 2, …)                            │
│    └─ Estado: LISTENING (espera ordenes)                      │
│                                                                │
│ 3. CONFIGURACIÓN DE CNN                                        │
│    ps.set_cnn(cnn)                                             │
│    └─ Cargar/crear CNN preentrenada                           │
│    └─ Será distribuida a Workers en siguiente sesión        │
│                                                                │
│ 4. SESIÓN DE ENTRENAMIENTO (TRAINING STATE)                   │
│    ps.train(epochs=10, training_mode="precomputed", …)        │
│    ├─ [SINCRONIZACIÓN CNN]                                    │
│    │  ├─ Enviar CNN_WEIGHTS a todos los Workers             │
│    │  └─ Esperar barrera CNN_READY                           │
│    │                                                          │
│    ├─ LOOP DE ÉPOCAS (N épocas)                              │
│    │  ├─ Generar seed aleatorio                              │
│    │  ├─ Enviar PARAMS (con seed) a cada Worker             │
│    │  ├─ Esperar GRADIENTS de TODOS los Workers            │
│    │  ├─ Promediar: ∇̄ = (1/N) * Σ ∇                        │
│    │  ├─ Actualizar: W ← W − lr * ∇̄                        │
│    │  ├─ Evaluar en test (si hay datos)                     │
│    │  └─ Callback: on_epoch_end()                            │
│    │                                                          │
│    └─ [FIN DE SESIÓN]                                        │
│       └─ Retorna a LISTENING (listo para siguiente sesión)   │
│                                                                │
│ 5. APAGADO (SHUTDOWN)                                          │
│    ps.shutdown()                                               │
│    └─ Envía STOP a todos los Workers                         │
│    └─ Cierra conexiones TCP                                  │
│    └─ Detiene hilo de aceptación                             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 💾 Estructuras de datos del PS

```python
class ParameterServer:
    # Configuración
    host: str = "0.0.0.0"        # Escucha en todas las IPs
    port: int = 9999             # Puerto TCP
    training_mode: str           # "precomputed" o "end_to_end"
    
    # Modelo
    _cnn: CNNExtractor           # CNN centralizada
    _mlp_params: Dict            # Parámetros MLP {W1, b1, …}
    
    # Workers conectados
    _worker_sockets: Dict[int, socket]  # worker_id → socket TCP
    _worker_addrs: Dict[int, str]       # worker_id → "IP:port"
    _next_id: int = 0                   # Contador para asignar IDs
    
    # Workers en sesión actual
    _active_training_workers: List[int]  # None si no hay sesión
    
    # Métricas de época actual
    _epoch_gradients: Dict[int, Dict]    # worker_id → gradients
    _epoch_metrics: Dict[int, Tuple]     # worker_id → (loss, acc)
    
    # Test data (opcional)
    _X_test_features: np.ndarray         # (10000, 512) o None
    _Y_test_from_worker: np.ndarray      # (10000,) o None
    
    # Sincronización
    _cnn_ready_event: threading.Event    # Barrera CNN_READY
    _cnn_ready_count: int                # Contador de CNN_READY recibidos
    
    # Threading
    _lock: threading.Lock               # Mutex para _worker_sockets, etc.
    _server_sock: socket               # Socket servidor TCP
    _accept_thread: threading.Thread   # Hilo de aceptación
    _shutdown_flag: threading.Event    # Flag para parar
```

---

## 🎯 Métodos principales

### **listen()**
```python
def listen(self) -> None:
    """
    Abre servidor TCP en background.
    Acepta Workers indefinidamente hasta shutdown().
    
    Retorna inmediatamente — las conexiones se procesan en hilo aparte.
    """
    self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._server_sock.bind((self.host, self.port))
    self._server_sock.listen(32)  # Queue de 32 conexiones
    
    # Inicia hilo de aceptación
    self._accept_thread = threading.Thread(target=self._accept_loop)
    self._accept_thread.start()
    
    print(f"Escuchando en {self.host}:{self.port}")
```

### **_accept_loop()** (en hilo de fondo)
```python
def _accept_loop(self) -> None:
    """
    Bucle infinito que acepta conexiones TCP.
    Por cada nueva conexión, ejecuta _handshake en hilo aparte.
    """
    while not self._shutdown_flag.is_set():
        try:
            conn, addr = self._server_sock.accept()
        except socket.timeout:
            continue  # Timeout corto para verificar shutdown_flag
        except:
            break
        
        # Handshake en hilo aparte (no bloquea accept())
        threading.Thread(
            target=self._handshake,
            args=(conn, addr),
            daemon=True
        ).start()
```

### **_handshake()** (en hilo aparte)
```python
def _handshake(self, conn, addr) -> None:
    """
    Realiza handshake con un Worker que acaba de conectarse.
    
    1. Lee READY
    2. Asigna ID único
    3. Envía WORKER_ID
    4. Registra en _worker_sockets
    5. Llama callback on_worker_connected
    """
    msg = receive_message(conn)
    if msg["type"] != MsgType.READY:
        conn.close()
        return
    
    # Asignar ID atomicamente
    with self._lock:
        worker_id = self._next_id
        self._next_id += 1
        self._worker_sockets[worker_id] = conn
        self._worker_addrs[worker_id] = f"{addr[0]}:{addr[1]}"
    
    # Enviar WORKER_ID
    send_message(conn, MsgType.WORKER_ID, {"worker_id": worker_id})
    
    # Callback
    if self.on_worker_connected:
        self.on_worker_connected(worker_id, f"{addr[0]}:{addr[1]}")
```

### **train()**
```python
def train(self, epochs, training_mode, test_data=None, learning_rate=0.01) -> Dict:
    """
    Ejecuta una sesión de entrenamiento.
    
    :param epochs: Número de épocas
    :param training_mode: "precomputed" o "end_to_end"
    :param test_data: (X_test, Y_test) o None
    :param learning_rate: Tasa de aprendizaje
    :return: { "accuracy_history": […], "loss_history": […], … }
    """
    # Limpiar estado anterior
    self._reset_training_state(training_mode)
    
    # Obtener Workers conectados
    with self._lock:
        active_workers = sorted(self._worker_sockets.keys())
    
    if not active_workers:
        raise RuntimeError("No hay Workers conectados")
    
    self._active_training_workers = active_workers
    n_workers = len(active_workers)
    
    # Send CNN_WEIGHTS a todos
    self._broadcast_cnn_weights(active_workers)
    
    # Esperar CNN_READY de todos
    self._cnn_ready_event.wait()  # Barrera de sincronización
    
    # [E2E ONLY] Solicitar TEST_FEATURES si es necesario
    if training_mode == "end_to_end" and test_data is not None:
        X_test, Y_test = test_data
        self._request_test_features(active_workers[0])  # Worker 0
    
    # LOOP DE ÉPOCAS
    history = {"train_acc": [], "train_loss": [], "test_acc": [], "test_loss": []}
    
    for epoch in range(epochs):
        # Generar seed de época
        seed = np.random.randint(0, 2**31 - 1)
        
        # Enviar PARAMS a cada Worker
        self._broadcast_params(
            active_workers,
            epoch=epoch,
            seed=seed,
            learning_rate=learning_rate
        )
        
        # Esperar GRADIENTS de todos
        gradients_dict = self._collect_gradients(active_workers)
        
        # Promediar gradientes
        avg_gradients = self._average_gradients(gradients_dict)
        
        # Actualizar parámetros
        self._update_parameters(avg_gradients, learning_rate)
        
        # Evaluar
        train_acc, train_loss = self._evaluate_train(active_workers)
        test_acc, test_loss = self._evaluate_test(test_data) if test_data else (None, None)
        
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        
        # Callback
        if self.on_epoch_end:
            self.on_epoch_end(epoch, epochs, train_acc, train_loss, test_acc, test_loss)
    
    # Retorna a LISTENING
    self._active_training_workers = None
    return history
```

---

## 🔄 Distribución de CNN

### **_broadcast_cnn_weights()**

```python
def _broadcast_cnn_weights(self, worker_ids) -> None:
    """
    Envía CNN_WEIGHTS a todos los Workers y espera CNN_READY.
    """
    weights_bytes = self._cnn._get_weights_bytes()
    
    for wid in worker_ids:
        with self._lock:
            sock = self._worker_sockets.get(wid)
        
        if sock is None:
            continue
        
        send_message(sock, MsgType.CNN_WEIGHTS, {
            "arch": self._cnn.arch,
            "weights_bytes": weights_bytes
        })
    
    # Esperar CNN_READY de todos
    self._cnn_ready_event.clear()
    self._cnn_ready_count = 0
    
    while self._cnn_ready_count < len(worker_ids):
        # Recibir CNN_READY en hilos aparte (listeners activos)
        time.sleep(0.1)
    
    print(f"[PS] Todos los {len(worker_ids)} Workers listos (CNN_READY)")
```

---

## 📊 Sincronización de parámetros

### **_broadcast_params()**

```python
def _broadcast_params(self, worker_ids, epoch, seed, learning_rate) -> None:
    """
    Envía PARAMS a cada Worker.
    
    PRECOMPUTED:
    ├─ epoch, params (MLP), seed
    └─ cnn_params: None
    
    END-TO-END:
    ├─ epoch, params (MLP), seed
    └─ cnn_params: CNN weights serialized
    """
    payload = {
        "epoch": epoch,
        "params": self._mlp_params,
        "seed": seed,
        "training_mode": self.training_mode,
        # Otros parámetros de sesión
        "n_train": 50000,
        "n_workers": len(worker_ids),
    }
    
    if self.training_mode == "end_to_end":
        payload["cnn_params"] = self._cnn._get_weights_bytes()
    
    # Enviar a cada worker con su rank
    for rank, wid in enumerate(worker_ids):
        with self._lock:
            sock = self._worker_sockets.get(wid)
        
        if sock is None:
            continue
        
        payload_wid = payload.copy()
        payload_wid["worker_rank"] = rank
        
        send_message(sock, MsgType.PARAMS, payload_wid)
```

---

## 📥 Recolección y promediación de gradientes

### **_collect_gradients()**

```python
def _collect_gradients(self, worker_ids) -> Dict[int, Dict]:
    """
    Espera GRADIENTS de todos los Workers.
    Usa listeners activos (threads que escuchan a cada socket).
    """
    gradients_dict = {}
    
    for wid in worker_ids:
        with self._lock:
            sock = self._worker_sockets.get(wid)
        
        if sock is None:
            raise RuntimeError(f"Worker {wid} desconectado durante entrenamiento")
        
        # Recibir GRADIENTS
        msg = receive_message(sock)
        if msg["type"] != MsgType.GRADIENTS:
            raise RuntimeError(f"Worker {wid} envió {msg['type']}, esperaba GRADIENTS")
        
        gradients_dict[wid] = msg["payload"]
    
    return gradients_dict
```

### **_average_gradients()**

```python
def _average_gradients(self, gradients_dict) -> Dict:
    """
    Promedia los gradientes de todos los Workers.
    
    Algebra:
    θ_avg = (1/N) * Σ ∇L(θ_i)  para i en workers
    
    Importante: es un promedio simple, no ponderado.
    """
    if not gradients_dict:
        raise ValueError("No hay gradientes para promediar")
    
    all_wids = sorted(gradients_dict.keys())
    n = len(all_wids)
    
    # Stack y promedia cada parámetro
    avg_grads = {}
    
    # Para MLP
    mlp_grads_list = [gradients_dict[wid]["gradients"] for wid in all_wids]
    
    for param_name in mlp_grads_list[0].keys():
        stacked = np.array([g[param_name] for g in mlp_grads_list])
        avg_grads[param_name] = np.mean(stacked, axis=0)
    
    # Para CNN (si E2E)
    if self.training_mode == "end_to_end":
        cnn_grads_list = [
            gradients_dict[wid].get("cnn_gradients", {})
            for wid in all_wids
        ]
        avg_grads["cnn"] = {}
        
        if cnn_grads_list[0]:  # Si hay
            for layer_name in cnn_grads_list[0].keys():
                stacked = np.array([g.get(layer_name) for g in cnn_grads_list])
                avg_grads["cnn"][layer_name] = np.mean(stacked, axis=0)
    
    return avg_grads
```

### **_update_parameters()**

```python
def _update_parameters(self, avg_gradients, learning_rate) -> None:
    """
    Aplica SGD: W ← W − lr * ∇̄
    """
    # MLP
    for param_name in self._mlp_params.keys():
        self._mlp_params[param_name] -= learning_rate * avg_gradients[param_name]
    
    # CNN (si E2E)
    if self.training_mode == "end_to_end" and self._cnn is not None:
        for layer_name, grad in avg_gradients.get("cnn", {}).items():
            # Actualizar parámetros CNN via PyTorch
            self._cnn._apply_gradient(layer_name, grad, learning_rate)
```

---

## 📈 Evaluación

### **_evaluate_test()**

```python
def _evaluate_test(self, test_data) -> Tuple[float, float]:
    """
    Evalúa en datos de prueba.
    
    PRECOMPUTED:
    ├─ X_test (raw) → CNN (local) → features
    ├─ features → MLP → logits
    └─ Calcular accuracy
    
    END-TO-END:
    ├─ Worker 0 envió X_test_features (con su GPU)
    ├─ X_test_features → MLP → logits
    └─ Calcular accuracy
    """
    if test_data is None:
        return None, None
    
    X_test, Y_test = test_data
    
    if self.training_mode == "precomputed":
        # Extraer features con CNN del PS (CPU)
        X_test_feat = self._cnn.extract_batched(X_test, batch_size=2048)
    else:
        # Usar features recibidos del Worker
        X_test_feat = self._X_test_features
        if X_test_feat is None:
            return None, None
    
    # Forward MLP
    logits = mlp_forward(self._mlp_params, X_test_feat)
    
    # Accuracy
    predictions = np.argmax(logits, axis=1)
    accuracy = np.mean(predictions == Y_test) * 100
    
    # Loss
    loss = cross_entropy_loss(logits, Y_test)
    
    return accuracy, loss
```

---

## 🧵 Manejo de conexiones en múltiples hilos

### **Por qué threading**

- `listen()` abre servidor TCP en hilo
- `_accept_loop()` acepta conexiones indefinidamente
- Por cada Worker que se conecta, `_handshake()` se ejecuta en hilo aparte
- Durante `train()`, hay múltiples hilos listeners esperando GRADIENTS de cada Worker

### **Sincronización con mutex**

```python
self._lock = threading.Lock()

# Sin lock: race condition
# with self._lock:
#     self._worker_sockets[wid] = sock  # Seguro

# Sin lock:
# if wid in self._worker_sockets:          # Otro thread podría borrar aquí
#     sock = self._worker_sockets[wid]     # Crash
```

---

## 🚨 Manejo de fallos

| Escenario | Efecto | Solución |
|-----------|--------|----------|
| Worker se desconecta en setup | CNN_READY timeout | Implement timeout, retry |
| Worker envía gradientes tarde | Epoch bloqueado | Timeout + failover |
| PS se cae durante training | Workers esperan indefinidamente | Workers timeout + reconnect |
| Red interrumpida | Conexión pierde datos | TCP handles partial) + retransmit |

---

## 📝 Callbacks del PS

```python
ps = ParameterServer(
    on_worker_connected=lambda wid, addr: print(f"Worker {wid} connected"),
    on_worker_joined_late=lambda wid, addr: print(f"Worker {wid} arrived late"),
    on_epoch_end=lambda epoch, total, train_acc, train_loss, test_acc, test_loss:
        print(f"Epoch {epoch}: {train_acc:.2f}%"),
    on_gradients_received=lambda wid, epoch, loss, acc:
        print(f"Worker {wid} gradients ready"),
)
```

---

**Documento**: `docs/06_parameter_server.md`  
**Última actualización**: 2026-03-27  
**Nivel**: Intermedio → Avanzado
