# Parameter Server: Núcleo del Sistema Distribuido

## Rol del Parameter Server

El Parameter Server (PS) es el **corazón centralizador** del sistema distribuido. Su responsabilidad es:

1. **Mantener sincronizados** los parámetros globales del modelo (CNN + MLP)
2. **Aceptar Workers** sin límite en tiempo de conexión
3. **Servir parámetros** a demanda sin bloqueos
4. **Aplicar gradientes** asincronicamente con corrección de staleness
5. **Rastrear métricas** de entrenamiento en tiempo real
6. **No participar** en el cálculo (solo almacenamiento + actualización)

---

## Inicialización de Modelos

### Carga de CNN

```python
# En ps_imagenet.py / ps_gui_imagenet.py
cnn = CNNExtractor(
    arch="resnet18",        # Opciones: "resnet18" o "simple"
    pretrained=True,        # Si True: ImageNet1K_V1 weights
    device="cpu",           # PS siempre en CPU (optimización)
    seed=42
)
ps.set_cnn(cnn)
```

**Bajo el capó**:
```python
# En ps.set_cnn()
def set_cnn(self, cnn: CNNExtractor) -> None:
    self._cnn = cnn
    base = getattr(cnn._model, "model", cnn._model)
    with self._params_lock:
        self._cnn_state = {
            name: tensor.cpu().numpy().copy()
            for name, tensor in base.state_dict().items()
        }
    _log.ps(f"CNN lista: arch={cnn.arch}, feature_dim={cnn.feature_dim}")
```

**Tamaño de CNN state_dict**:
- ResNet-18: ~44 MB (11M parámetros × 4 bytes float32)
- Simple CNN: ~6 MB (1.5M parámetros)

### Carga de MLP

```python
# En ps_imagenet.py / ps_gui_imagenet.py
mlp = MLPPyTorch(
    feature_dim=512,        # Proveniente de CNN
    hidden1=1024,           # Configurable
    hidden2=512,            # Configurable
    n_classes=1000
)
ps.set_mlp(mlp.state_dict_numpy())
```

**Tamaño de MLP state_dict** (feature_dim=512, h1=1024, h2=512):
- fc1.weight: (1024, 512) = 524KB
- fc1.bias: (1024,) = 4KB
- fc2.weight: (512, 1024) = 2MB
- fc2.bias: (512,) = 2KB
- fc3.weight: (1000, 512) = 2MB
- fc3.bias: (1000,) = 4KB
- **Total**: ~4.5 MB (2.05M parámetros)

**Inicialización: Kaiming Uniform (He)**
```python
# En MLPPyTorch._init_weights()
for layer in (fc1, fc2, fc3):
    nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
    nn.init.zeros_(layer.bias)
```

Garantiza que los logits iniciales tengan varianza ≈ 1, produciendo:
- Loss inicial ≈ log(1000) ≈ 6.9 (no 0, no NaN)
- Accuracy inicial ≈ 0.1% (aleatorio)

---

## Estructura Interna del PS

```
Parameter Server
├─ _mlp_state        Dict[str, np.ndarray]
│  ├─ "fc1.weight"   (1024, 512)
│  ├─ "fc1.bias"     (1024,)
│  ├─ "fc2.weight"   (512, 1024)
│  ├─ "fc2.bias"     (512,)
│  ├─ "fc3.weight"   (1000, 512)
│  └─ "fc3.bias"     (1000,)
│
├─ _cnn_state        Dict[str, np.ndarray]  (ResNet-18 state_dict)
│
├─ _version          int  (contador monotónico)
│
├─ _sockets          Dict[int, socket.socket]
│  └─ {wid: socket}
│
├─ _metrics          RunningMetrics
│  └─ ventana deslizante de (loss, acc, staleness)
│
├─ _history          Dict[str, List]
│  ├─ "steps"        [0, 1, 2, ...]
│  ├─ "losses"       [8.3, 8.2, 8.1, ...]
│  ├─ "accuracies"   [0.0, 0.0, 1.5, ...]
│  ├─ "n_workers"    [1, 1, 2, ...]
│  └─ "timestamps"   [t0, t1, t2, ...]
│
├─ _params_lock      threading.Lock  (protege _mlp_state, _cnn_state, _version)
├─ _workers_lock     threading.Lock  (protege _sockets, _addrs)
├─ _history_lock     threading.Lock  (protege _history)
│
└─ _accept_thread    threading.Thread  (corre _accept_loop)
   └─ (por cada Worker hay otro hilo de _serve_worker)
```

---

## Manejo de Conexiones de Workers

### Handshake por Worker

Cuando un Worker conecta, el PS ejecuta `_handle_new_connection(conn, addr)`:

```
PASOS:
1. Recibir READY
   ├─ Si socket error → cerrar
   └─ Si msg ≠ READY → cerrar

2. Asignar WORKER_ID
   ├─ wid = self._next_id++ (garantiza IDs únicos)
   ├─ Guardar en diccionario _sockets[wid] = conn
   └─ send(WORKER_ID, {"worker_id": wid})

3. Distribuir CNN
   ├─ if self._cnn is None → no enviar (falla de inicialización)
   ├─ send(CNN_WEIGHTS, {
   │    "arch": "resnet18",
   │    "weights_bytes": self._cnn._get_weights_bytes()  # ≈44MB
   │  })
   ├─ Esperar CNN_ACK (bloqueo en recv, timeout implícito)
   └─ if no CNN_ACK → desconectar Worker

4. Enviar START
   ├─ send(START, {})  # Señal de inicio del training loop
   └─ if error → desconectar

5. Entrar en _serve_worker()
   └─ Loop indefinido: recv → procesar → responder
```

**Tiempo de handshake**: ~1-5 segundos (depende de tamaño CNN + latencia red)

### Loop de Servio Asincrónico

```python
def _serve_worker(wid, conn):
    try:
        while not self._shutdown.is_set():
            try:
                msg = receive_message(conn)  # Bloqueante
            except Exception:
                break
            
            mtype = msg["type"]
            
            if mtype == MsgType.REQUEST_PARAMS:
                # Copiar estado actual (thread-safe con lock)
                with self._params_lock:
                    mlp_copy = copy(self._mlp_state)
                    cnn_copy = copy(self._cnn_state)
                    version = self._version
                
                send_message(conn, MsgType.PARAMS, {
                    "mlp_state": mlp_copy,
                    "cnn_state": cnn_copy,
                    "version": version,
                    "lr": self.learning_rate
                })
            
            elif mtype == MsgType.UPDATES:
                # Aplicar gradientes con corrección staleness
                self._apply_update(wid, msg["payload"])
    
    finally:
        # Limpieza
        self._remove_worker(wid)
```

**Características**:
- Un hilo por Worker (no hay contención en UPDATE)
- Non-blocking respecto a otros Workers
- `_params_lock` protege copias (duración ≈ 1 µs)

---

## Actualización Asincrónica: Async-SGD

### Fórmula de Actualización

```
θ_new = θ + α(s) · (θ_worker - θ)

donde:
  θ        = parámetros actuales en PS
  θ_worker = parámetros en Worker (después de entrenar)
  s        = staleness = version - version_read
  α(s)     = 1 / (1 + λ · s)
  λ        = staleness_lambda (default 0.1)
```

### Implementación en Python

```python
def _apply_update(self, wid, payload):
    loss = payload.get("loss", 0.0)
    acc = payload.get("accuracy", 0.0)
    version_read = payload.get("version_read", 0)
    mlp_weights = payload.get("mlp_weights")
    cnn_weights = payload.get("cnn_weights")
    
    with self._params_lock:
        # Calcular staleness y factor alpha
        staleness = max(0, self._version - version_read)
        alpha = 1.0 / (1.0 + self.staleness_lambda * staleness)
        
        # Actualizar MLP
        if mlp_weights:
            for key in self._mlp_state:
                if key in mlp_weights:
                    delta = mlp_weights[key] - self._mlp_state[key]
                    self._mlp_state[key] += alpha * delta
        
        # Actualizar CNN (análogo)
        if cnn_weights:
            for key in self._cnn_state:
                if key in cnn_weights:
                    delta = cnn_weights[key] - self._cnn_state[key]
                    self._cnn_state[key] += alpha * delta
        
        # Incrementar versión después de actualizar
        self._version += 1
        
        # Registrar métricas
        self._metrics.add(loss, acc, staleness)
        self._record_history(loss, acc)
        
        # Callbacks
        if self.on_step:
            self.on_step(self._version, loss, acc, staleness)
        if self._version % self.steps_per_report == 0 and self.on_report:
            avg_loss, avg_acc = self._metrics.average()
            self.on_report(self._version, avg_loss, avg_acc)
```

### Exclusión de Variables de Tracking

**Problema**: BatchNorm tiene `num_batches_tracked` que no debe promediarse

```python
_NO_AVG_KEYS = {"num_batches_tracked"}  # No promediar

if key not in _NO_AVG_KEYS:
    # Actualizar normalmente
else:
    # Mantener valor del PS (no mezclediar contador)
```

---

## Thread Safety (Seguridad en concurrencia)

### Locks Utilizados

| Lock | Protege | Duración | Contención |
|---|---|---|---|
| `_params_lock` | _mlp_state, _cnn_state, _version | ~1 µs | Alta (copias rápidas) |
| `_workers_lock` | _sockets, _addrs, _next_id | ~0.1 µs | Baja (conexiones raras) |
| `_history_lock` | _history | ~0.1 µs | Baja (GUI cada 100ms) |

### Deadlock Prevention

```
CORRECTO (siempre mismo orden):
├─ Adquirir _params_lock
├─ Realizar operación
└─ Liberar

INCORRECTO (potencial deadlock):
├─ Thread 1: acquire params_lock → acquire workers_lock
└─ Thread 2: acquire workers_lock → acquire params_lock (⚠️ DEADLOCK)
```

En este proyecto: **Nunca se adquieren múltiples locks simultáneamente**.

---

## Monitoreo de Métricas

### Estructura RunningMetrics

```python
class RunningMetrics:
    def __init__(self, window=200):
        self.window = window  # Últimas 200 métricas
        self._losses = deque(maxlen=window)
        self._accuracies = deque(maxlen=window)
        self._staleness = deque(maxlen=window)
    
    def add(self, loss, acc, staleness):
        self._losses.append(loss)
        self._accuracies.append(acc)
        self._staleness.append(staleness)
    
    def average(self):
        return np.mean(self._losses), np.mean(self._accuracies)
```

**Razón de ventana deslizante**: Metrics recientes + historia (no recompiar todo)

### Historial para GUI

```python
{
    "steps": [0, 1, 2, ...],
    "losses": [8.37, 8.26, 8.15, ...],
    "accuracies": [0.0, 0.0, 1.56, ...],
    "n_workers": [1, 1, 1, 2, 2, ...],
    "timestamps": [t0, t1, t2, ...]
}
```

Usado por `ps_gui_imagenet.py` para plotear gráficas.

---

## Graceful Shutdown

```python
def stop(self):
    _log.ps("Deteniendo servidor...")
    self._shutdown.set()  # Signal a todos los _serve_worker threads
    
    with self._workers_lock:
        wids = list(self._sockets.keys())
    
    # Desconectar todos los Workers
    for wid in wids:
        try:
            send_message(self._sockets[wid], MsgType.STOP, {})
        except:
            pass  # Si falla, OK (connection already dead)
        self._remove_worker(wid)
    
    # Cerrar server socket
    if self._server_sock:
        try:
            self._server_sock.close()
        except:
            pass
    
    if self._accept_thread:
        self._accept_thread.join(timeout=3)
    
    _log.ps("Servidor detenido.")
```

---

## Limitaciones y Consideraciones

### Cuello de Botella: PS Centralizado

Con N Workers:
- Si cada Worker envía UPDATE cada 100ms
- Y cada UPDATE es 4.5 MB + overhead
- Throughput necesario = N × 4.5 MB / 0.1s = N × 45 MB/s

Con 10 Gbps Ethernet: Máximo ~10 Workers sin saturación

### Escalabilidad Futura

Para más Workers:
- **Replicación de PS** (master-replica con sincronización)
- **Sharding de parámetros** (parámetro A en PS1, parámetro B en PS2)
- **Gradiente Compression** (reducir tamaño msgs)
- **Comunicación directa Worker-Worker** (gossip protocol)

