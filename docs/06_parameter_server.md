# 06. Parameter Server: Orquestación Distribuida

## Fases de Ciclo de Vida

### Fase 1: Creación e Inicialización

```python
ps = ParameterServer(
    host="0.0.0.0",         # Escucha en todas las IPs
    port=9999,
    on_worker_connected=callback_worker_connected,
    on_gradients_received=callback_grads,
    on_epoch_end=callback_epoch,
    training_mode="precomputed",
)

ps.set_cnn(cnn_model)  # CNN a distribuir a Workers
```

**Estado inicially**:
- `_server_sock = None` (no escuchando aún)
- `_worker_sockets = {}` (vacío)
- `_cnn = cnn_model` (preentrenada o inicializada)
- `_training_mode = "precomputed"`

### Fase 2: Listen (Aceptar Workers)

```python
ps.listen()  # Lanza hilo de fondo
    ├─ Create TCP socket, bind(host:port), listen()
    ├─ Lanza _accept_thread (bloqueante)
    │   └─ while True:
    │       ├─ sock, addr = accept()  # Bloquea hasta que Worker conecte
    │       ├─ Ejecutar handshake:
    │       │   ├─ msg = receive_message(sock)  # recibe READY
    │       │   ├─ next_id = _next_id++
    │       │   ├─ send_message(sock, WORKER_ID, {"worker_id": next_id})
    │       │   ├─ _worker_sockets[next_id] = sock
    │       │   └─ on_worker_connected(next_id, addr)
    │       └─ Vuelve a aceptar
    └─ Devuelve inmediatamente (thread en background)
```

**Invariante**: `listen()` NO bloquea al que llamó. Retorna inmediatamente. El hilo de aceptación continúa indefinidamente.

### Fase 3: Espera Workers (Sincronización Manual)

La aplicación (ps_terminal.py o ps_gui.py) espera manualmente:

```python
# En ps_terminal.py
n_workers_expected = 3
print(f"Waiting for {n_workers_expected} workers...")

while len(ps._worker_sockets) < n_workers_expected:
    time.sleep(0.5)
    print(f"  Connected: {len(ps._worker_sockets)}/{n_workers_expected}")

print("All workers connected. Starting training.")
```

**Alternativa** (no implementada): Barrera automática con threading.Event.

### Fase 4: Entrenamiento

```python
history = ps.train(
    epochs=10,
    learning_rate=0.01,
    X_test=X_test,
    Y_test=Y_test,
)
```

Es la función principal. Devuelve un diccionario con historial de pérdidas/precisiones.

---

## Flujo train() en Detalle

### Sintaxis

```python
def train(
    self,
    epochs: int,
    learning_rate: float = 0.01,
    X_test: Optional[np.ndarray] = None,
    Y_test: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> dict:
    """
    Sesión de entrenamiento ONE.
    
    Puede ser llamado múltiples veces sin reiniciar el PS.
    """
```

### Inicialización de CNN y Features de Test

```python
# Paso 1: Reset training state
self._reset_training_state(self.training_mode)

# Paso 2: Cargar features de test (solo si hay datos test)
if X_test is not None and Y_test is not None:
    # Extractar features con la CNN del PS
    X_test_features, _ = self._cnn.extract_with_cache(
        X_test, Y_test,
        split="test",
    )
    self._X_test_features = X_test_features    # Guardar para evaluate
    self._Y_test_from_worker = Y_test
else:
    self._X_test_features = None
    self._Y_test_from_worker = None
```

### Envío de CNN_WEIGHTS a Workers

```python
# Paso 3: Distribuir CNN a Workers
cnn_weights_bytes = self._cnn._get_weights_bytes()

for worker_id in self._worker_sockets:
    send_message(self._worker_sockets[worker_id], MsgType.CNN_WEIGHTS, {
        "arch": self._cnn.arch,
        "weights_bytes": cnn_weights_bytes,
    })
```

**Nota**: Envía a TODOS los Workers en el `_worker_sockets` dict. Si un Worker se conectó más tarde, también lo recibe (late-joiner).

### Barrera CNN_READY

```python
# Paso 4: Esperar a que todos extraigan features
self._cnn_ready_event.clear()
self._cnn_ready_count = 0

print(f"[PS] Waiting for CNN_READY from all workers...")

# Bloquea hasta CNN_READY_EVENT se activa (visto en _handle_cnn_ready callback)
self._cnn_ready_event.wait()  # Bloqueante

print(f"[PS] All workers ready with features.")
```

**Mecanismo**: Mientras el PS espera, el hilo de lectura (que recibe mensajes de cada Worker) ejecuta callbacks. Cuando llega CNN_READY #N del N-ésimo Worker, se activa el evento.

**Risk**: Si un Worker nunca envía CNN_READY, el PS espera para siempre (deadlock).

### Loop Principal: Época a Época

```python
# Paso 5: Entrenar por N épocas
params = mlp.init_params(FEATURE_DIM, hidden1, hidden2, NUM_CLASSES, seed=42)

for epoch in range(epochs):
    # 5a. Crear semilla para esta época
    epoch_seed = 42 + epoch
    
    # 5b. Enviar PARAMS a todos los Workers
    for worker_id in self._worker_sockets:
        send_message(self._worker_sockets[worker_id], MsgType.PARAMS, {
            "epoch": epoch,
            "params": params,
            "seed": epoch_seed,
            "training_mode": self.training_mode,
        })
    
    # 5c. Recibir GRADIENTS de todos (bloqueante)
    self._epoch_gradients.clear()
    self._epoch_metrics.clear()
    
    for _ in range(len(self._worker_sockets)):
        msg = receive_message()  # Bloqueante hasta recibir UN mensaje
        
        if msg["type"] == MsgType.GRADIENTS:
            worker_id = msg["payload"]["worker_id"]
            gradients = msg["payload"]["gradients"]
            loss = msg["payload"]["loss"]
            accuracy = msg["payload"]["accuracy"]
            
            self._epoch_gradients[worker_id] = gradients
            self._epoch_metrics[worker_id] = (loss, accuracy)
            
            # Callback
            if self.on_gradients_received:
                self.on_gradients_received(worker_id, epoch, loss, accuracy)
    
    # 5d. Promediar gradientes
    averaged_grads = self._average_gradients(self._epoch_gradients)
    
    # 5e. Actualizar pesos
    mlp.apply_gradients(params, averaged_grads, learning_rate)
    
    # 5f. Evaluar en test (si disponible)
    if self._X_test_features is not None:
        test_acc, test_loss = mlp.evaluate(params, self._X_test_features, self._Y_test_from_worker)
    else:
        test_acc, test_loss = None, None
    
    # 5g. Calcular métricas de train (promedio de workers)
    train_accs = [acc for loss, acc in self._epoch_metrics.values()]
    train_losses = [loss for loss, acc in self._epoch_metrics.values()]
    train_acc = np.mean(train_accs)
    train_loss = np.mean(train_losses)
    
    # 5h. Callback y logging
    if self.on_epoch_end:
        self.on_epoch_end(epoch, epochs, train_acc, train_loss, test_acc, test_loss)
    
    if verbose:
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Acc={train_acc:.2f}% Loss={train_loss:.4f} | "
              f"Test Acc={test_acc:.2f}% Loss={test_loss:.4f}")
```

### Shutdown

```python
# Paso 6: Enviar STOP a todos cuando termines
for worker_id in self._worker_sockets:
    send_message(self._worker_sockets[worker_id], MsgType.STOP, None)

# Opcionalmente, cierra sockets
```

---

## Threading Model del PS

### Threads principales

1. **Main thread** (ps_terminal.py):
   - Llama a `ps.listen()` (inicia background thread)
   - Espera Workers manualmente
   - Llama a `ps.train()` (bloqueante)
   - Recibe callbacks

2. **Accept thread** (`_accept_thread`):
   - Bloqueante en `socket.accept()`
   - Cuando Worker conecta: handshake + callback

3. **Message receiver threads** (uno por Worker implícitamente):
   - `receive_message()` es bloqueante
   - **Problema**: Si hay 3 Workers, `ps.receive_message()` recibe de primero que llegue, pero el resto está "pendiente"
   - **Solución actual**: No es verdaderamente async. El PS es semi-syncrónico.

### Sincronización con Mutex

```python
self._lock = threading.Lock()

# Protegidas:
with self._lock:
    self._next_id += 1
    self._worker_sockets[next_id] = sock
```

**Por qué**:  Si el accept thread intenta asignar ID 5 mientras el train thread lee `_worker_sockets`, race condition.

---

## Promediado de Gradientes

```python
def _average_gradients(
    self,
    epoch_gradients: Dict[int, Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Promedia gradientes de todos los Workers.
    
    Input: {
        0: {"W1": array(...), "b1": array(...), ...},
        1: {"W1": array(...), "b1": array(...), ...},
        2: {"W1": array(...), "b1": array(...), ...},
    }
    
    Output: {
        "W1": (promedio de worker 0, 1, 2),
        "b1": (promedio),
        ...
    }
    """
    if not epoch_gradients:
        return {}
    
    all_param_names = list(epoch_gradients[0].keys())
    averaged = {}
    
    for param_name in all_param_names:
        grad_list = [
            epoch_gradients[worker_id][param_name]
            for worker_id in sorted(epoch_gradients.keys())
        ]
        grad_stack = np.array(grad_list)
        averaged[param_name] = np.mean(grad_stack, axis=0)
    
    return averaged
```

**Matemática**:
```
grad_stack.shape = (n_workers, *param_shape)

Para W1: (256, 512)
  grad_stack W1: (3, 256, 512)
  mean(axis=0): (256, 512)

Para b1: (256,)
  grad_stack b1: (3, 256)
  mean(axis=0): (256,)
```

---

## Manejo de Late Joiners

Si un Worker se conecta DURANTE el entrenamiento:

```python
elif msg["type"] == MsgType.READY:
    # En middleware/accept loop
    with self._lock:
        next_id = self._next_id
        self._next_id += 1
    
    # Checar si estamos entrenando
    if self._active_training_workers is not None:
        # Sí, hay entrenamiento activo
        if self.on_worker_joined_late:
            self.on_worker_joined_late(next_id, addr)
        
        # Registrar worker pero NO incluirlo en sesión actual
        self._worker_sockets[next_id] = sock
        
        # Nota: Este worker recibirá CNN_WEIGHTS pero NO PARAMS
        # de este entrenamiento. Se incluirá en el SIGUIENTE.
    else:
        # No hay entrenamiento, registrar normally
        self._worker_sockets[next_id] = sock
```

**GUI feedback**: ps_gui mostraría "Worker 5 joined (waiting for next session)".

---

## Checkpointing (no en código base)

Idealmente, el PS debería guardar checkpoints:

```python
# Versión extendida (no está en código actual)
def save_checkpoint(self, path: str) -> None:
    checkpoint = {
        "epoch": current_epoch,
        "params": params,
        "training_mode": self.training_mode,
        "cnn_state": self._cnn.state_dict(),
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(self, path: str) -> None:
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    # Restaurar estado
```

**Beneficio**: Poder continuar entrenamiento después de crash o cambio de modo.

---

## Gestión de Errores Robusta (Ausente)

Problemas NO manejados actualmente:

1. **Worker timeout**: Si Worker 2 falla, PS espera forever en `receive_message()`
2. **Red cortada**: Si conexión TCP cae, `receive_message()` levanta excepción
3. **Worker desconecta a mitad de época**: Otros Workers ya escribieron gradientes, Worker 3 no

**Ideal**:
```python
try:
    for _ in range(len(self._worker_sockets)):
        msg = receive_message(timeout=30)  # 30 segundo TIMEOUT
        ...
except socket.timeout:
    print("Worker timeout. Aborting epoch.")
    break
```

---

## Evaluación en Test

El PS evalúa en el MISMO proceso (bloqueante):

```python
if self._X_test_features is not None:
    test_acc, test_loss = mlp.evaluate(
        params,
        self._X_test_features,  # (10000, 512)
        self._Y_test,           # (10000,)
    )
```

**Timing**: 10000 imágenes × forward MLP ≈ 100-200ms en NumPy.

**Alternativa posible**: Enviar a un Worker designado, pero actualmente no está.

---

## Logging y Debugging

El PS usa un logger con colores:

```python
_logger.ps(f"Listening on {host}:{port}")
_logger.ps(f"Worker {worker_id} connected from {addr}")
_logger.ps(f"Received PARAMS epoch={epoch} from worker {worker_id}")
_logger.ps(f"Epoch {epoch}: Train Acc={acc:.2f}% Loss={loss:.4f}")
```

**Debug mode**: `ParameterServer(debug=True)` imprime tracebacks adicionales.

