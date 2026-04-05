# Worker: Nodo de Entrenamiento Asincrónico

## Ciclo de Vida del Worker

```
START
  ↓
_connect()            → socket.connect() al PS, send(READY), recv(WORKER_ID), recv(CONFIG)
  ↓
_init_stream()        → build_worker_stream(), stream.start()
  ↓
_handshake_loop()     → recv(CNN_WEIGHTS), send(CNN_ACK), recv(START)
  ↓
_training_loop()      → **INDEFINIDO** REQUEST_PARAMS → train → UPDATES
  ↓
_cleanup()            → stream.stop(), socket.close()
  ↓
END
```

---

## Inicialización: _connect()

```python
def _connect(self) -> None:
    """Conectar al PS y obtener WORKER_ID y CONFIG."""
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._sock.connect((self.server_host, self.server_port))
    
    # 1. Señalizar que estamos listos
    send_message(self._sock, MsgType.READY, {})
    
    # 2. Recibir ID asignado por PS
    msg = receive_message(self._sock)
    if msg["type"] != MsgType.WORKER_ID:
        raise ConnectionError(f"Esperaba WORKER_ID, recibí {msg['type']}")
    self._worker_id = msg["payload"]["worker_id"]
    
    # 3. Recibir CONFIG (batch_size, image_size)
    msg = receive_message(self._sock)
    if msg["type"] != MsgType.CONFIG:
        raise ConnectionError(f"Esperaba CONFIG, recibí {msg['type']}")
    config = msg["payload"]
    self.batch_size = config["batch_size"]
    self.image_size = config["image_size"]
    
    self._log(f"Conectado con ID={self._worker_id}, batch_size={self.batch_size}, image_size={self.image_size}")
```

**Tiempo**: ~100-500 ms (depende de latencia red)

---

## Inicialización de Stream: _init_stream()

```python
def _init_stream(self) -> None:
    """Construir y arrancar pipeline de streaming."""
    self._stream = build_worker_stream(
        worker_rank=self.worker_rank,
        num_workers=self.num_workers,
        batch_size=self.batch_size,
        dataset_name=self.dataset_name,
        image_size=self.image_size,
        shuffle_buffer=self.shuffle_buffer,
        prefetch_batches=self.prefetch_batches,
        seed=(self.seed + self.worker_rank) if self.seed is not None else None,
        hf_token=self.hf_token,
    )
    self._stream.start()  # ← Inicia hilo background de prefetch
```

El stream:
- Crea connection a HuggingFace
- Descarga metadatos de dataset
- Comienza a prefetching batches en background
- **No bloquea** el training loop

---

## Handshake: _handshake_loop()

```python
def _handshake_loop(self) -> None:
    """Esperar CNN_WEIGHTS, confirmar, esperar START, entrar en training."""
    assert self._sock is not None

    while True:
        msg = receive_message(self._sock)  # Bloqueante
        t = msg["type"]

        if t == MsgType.STOP:
            self._log("STOP recibido.")
            return

        elif t == MsgType.CNN_WEIGHTS:
            self._load_cnn(msg["payload"])

        elif t == MsgType.START:
            self._log("START recibido — iniciando loop de entrenamiento.")
            self._training_loop()
            return
```

**Fallback automático**: Si CNN no llega del PS, Worker crea CNN por defecto:
```python
if self._cnn is None:
    self._cnn = CNNExtractor(arch="resnet18", device=str(self.device))
    self._log("⚠  CNN no recibida del PS — usando ResNet18 por defecto")
```

---

## Loop de Entrenamientos: _training_loop()

```python
def _training_loop(self) -> None:
    """Indefinido: REQUEST_PARAMS → train → UPDATES."""
    
    if self._cnn is None:
        self._cnn = CNNExtractor(arch="resnet18", device=str(self.device))
    
    assert self._sock is not None
    assert self._cnn is not None
    assert self._stream is not None

    stream_iter = iter(self._stream)
    version_read = 0
    iterations = 0

    while True:
        iterations += 1
        
        # ══════════════════════════════════════════════════════════
        # 1. SOLICITAR PARÁMETROS
        # ══════════════════════════════════════════════════════════
        try:
            send_message(self._sock, MsgType.REQUEST_PARAMS, {})
            msg = receive_message(self._sock)
        except Exception as e:
            self._log(f"Error de comunicación: {e}")
            return

        if msg["type"] == MsgType.STOP:
            return
        if msg["type"] != MsgType.PARAMS:
            self._log(f"Mensaje inesperado: {msg['type']}")
            continue

        payload = msg["payload"]
        mlp_state = payload["mlp_state"]
        cnn_state = payload["cnn_state"]
        version_read = payload["version"]
        lr = payload["lr"]

        # ══════════════════════════════════════════════════════════
        # 2. SINCRONIZAR PARÁMETROS
        # ══════════════════════════════════════════════════════════
        self._sync_cnn(cnn_state)
        self._mlp = self._sync_mlp(mlp_state, self._mlp)

        # ══════════════════════════════════════════════════════════
        # 3. ENTRENAR N BATCHES (accumulation)
        # ══════════════════════════════════════════════════════════
        total_loss, total_acc, total_n = 0.0, 0.0, 0

        for step_idx in range(self.accum_steps):
            try:
                X_np, Y_np = next(stream_iter)
            except StopIteration:
                assert self._stream is not None
                stream_iter = iter(self._stream)  # Reiniciar stream
                X_np, Y_np = next(stream_iter)

            loss, acc, n = self._train_batch(X_np, Y_np, lr)
            total_loss += loss * n
            total_acc += acc * n
            total_n += n

        if total_n == 0:
            continue

        avg_loss = total_loss / total_n
        avg_acc = total_acc / total_n
        self._batches_done += self.accum_steps

        if self.verbose and self._batches_done % 10 == 0:
            assert self._stream is not None
            self._log(
                f"batch={self._batches_done} | "
                f"loss={avg_loss:.4f} | acc={avg_acc:.2f}% | "
                f"v={version_read} | q={self._stream.queue_size}"
            )

        # ══════════════════════════════════════════════════════════
        # 4. ENVIAR ACTUALIZACIONES AL PS
        # ══════════════════════════════════════════════════════════
        try:
            send_message(
                self._sock,
                MsgType.UPDATES,
                {
                    "loss": avg_loss,
                    "accuracy": avg_acc,
                    "batch_size": total_n,
                    "version_read": version_read,
                    "mlp_weights": self._serialize_mlp(),
                    "cnn_weights": self._serialize_cnn(),
                },
            )
        except Exception as e:
            self._log(f"Error enviando UPDATES: {e}")
            return
        
        # VUELVE AL INICIO DEL LOOP (paso 1)
```

---

## Forward Pass Completo: _train_batch()

```python
def _train_batch(self, X_np, Y_np, lr) -> Tuple[float, float, int]:
    """Entrenar 1 batch: S2E forward + backward + SGD local."""
    assert self._cnn is not None
    assert self._mlp is not None

    # Convertir numpy a tensores
    X = torch.from_numpy(X_np).to(self.device)  # (64, 3, 224, 224)
    Y = torch.from_numpy(Y_np.astype(np.int64)).to(self.device)  # (64,)

    # Preparar para gradientes
    self._cnn._model.train()
    for p in self._cnn._model.parameters():
        p.requires_grad_(True)  # Solo para SimpleCNN, ResNet-18 permanece False
    self._mlp.train()

    # Forward: SimpleCNN entrenable, ResNet-18 congelada
    self._cnn._model.zero_grad()
    self._mlp.zero_grad()
    
    features = self._cnn._model(X)  # (64, 512)
    logits = self._mlp(features)    # (64, 1000)
    loss = nn.functional.cross_entropy(logits, Y)

    # Backward
    loss.backward()

    # SGD local (sin momentum, sin wd)
    # SimpleCNN: CNN gradientes se propagan, se SGD
    # ResNet-18: CNN congelada, sin gradientes
    with torch.no_grad():
        for p in self._cnn._model.parameters():
            if p.grad is not None:
                p.data -= lr * p.grad  # Cambios no persisten (cambios locales descartan en REQUEST_PARAMS)
        for p in self._mlp.parameters():
            if p.grad is not None:
                p.data -= lr * p.grad

    # Métricas
    with torch.no_grad():
        correct = (logits.argmax(1) == Y).sum().item()
    
    n = len(Y_np)
    loss_val = loss.item()
    acc_val = 100.0 * correct / n

    # Restaurar CNN a eval
    self._cnn._model.eval()
    for p in self._cnn._model.parameters():
        p.requires_grad_(False)

    # Liberar memoria
    del X, Y, features, logits, loss
    return loss_val, acc_val, n
```

**Tiempo para este paso**: ~50-100ms (depende de device)

---

## Sincronización de Parámetros

### Sincronizar CNN

```python
def _sync_cnn(self, cnn_state: Dict[str, np.ndarray]) -> None:
    """Cargar state_dict del CNN desde PS."""
    assert self._cnn is not None
    base = getattr(self._cnn._model, "model", self._cnn._model)
    
    with torch.no_grad():
        for name, param in base.named_parameters():
            if name in cnn_state:
                param.data.copy_(
                    torch.from_numpy(cnn_state[name]).to(param.device)
                )
        for name, buf in base.named_buffers():
            if name in cnn_state:
                buf.copy_(torch.from_numpy(cnn_state[name]).to(buf.device))
```

### Sincronizar MLP

```python
def _sync_mlp(self, mlp_state, existing) -> MLPPyTorch:
    """Cargar o crear MLP con estado actual."""
    if existing is None:
        if not mlp_state or "fc1.weight" not in mlp_state:
            # MLP no disponible en PS → crear con defaults
            feature_dim = self._cnn.feature_dim
            hidden1 = self.hidden1
            hidden2 = self.hidden2
            existing = MLPPyTorch(feature_dim, hidden1, hidden2, 1000).to(self.device)
            self._log(
                f"⚠  MLP no recibido del PS — "
                f"usando valores por defecto ({feature_dim}→{hidden1}→{hidden2}→1000)"
            )
        else:
            # MLP disponible → inferir dimensiones
            feature_dim = mlp_state["fc1.weight"].shape[1]
            hidden1 = mlp_state["fc1.weight"].shape[0]
            hidden2 = mlp_state["fc2.weight"].shape[0]
            existing = MLPPyTorch(feature_dim, hidden1, hidden2, 1000).to(self.device)
    
    # Cargar parámetros
    with torch.no_grad():
        for name, param in existing.named_parameters():
            if name in mlp_state:
                param.data.copy_(
                    torch.from_numpy(mlp_state[name]).to(param.device)
                )
    return existing
```

---

## Serialización de Parámetros

### Exportar para TCP

```python
def _serialize_cnn(self) -> Dict[str, np.ndarray]:
    """CNN state_dict → numpy arrays."""
    assert self._cnn is not None
    base = getattr(self._cnn._model, "model", self._cnn._model)
    return {
        name: tensor.cpu().numpy().copy()
        for name, tensor in base.state_dict().items()
    }

def _serialize_mlp(self) -> Dict[str, np.ndarray]:
    """MLP state_dict → numpy arrays."""
    assert self._mlp is not None
    return {
        name: param.data.cpu().numpy().copy()
        for name, param in self._mlp.named_parameters()
    }
```

**Tamaño de serialización**:
- MLP: 4.5 MB (6 arrays)
- CNN: 44 MB (49 arrays con BN buffers)
- **Total por UPDATE**: ~48.5 MB

**Tiempo de serialización**: ~50-100ms (copy a CPU + numpy conversion)

---

## Fallbacks Automáticos

| Componente | Fallback |
|---|---|
| **CNN no recibida** | Crear ResNet-18 local (default) |
| **MLP no recibida** | Crear MLP con valores default (h1=1024, h2=512) |
| **Stream agotado** | Reiniciar iterador automáticamente |
| **Network timeout** | Registrar error y desconectar |

**Objetivo**: Sistema resiliente que no se cuelga incluso con problemas de sincronización PS-Worker

---

## Logging y Monitoreo

### Mensaje de Log Típico

```
[W0] batch=100 | loss=7.2341 | acc=1.56% | v=15 | q=3
     ┬    ┬     ┬     ┬       ┬    ┬     ┬    ┬   ┬
     │    │     │     │       │    │     │    │   └─ queue_size (prefetch)
     │    │     │     │       │    │     │    └───── version_read
     │    │     │     │       │    │     └────────── assignment
     │    │     │     │       │    └─────────────── accuracy %
     │    │     │     │       └────────────────── loss
     │    │     │     └───────────────────────── avg_loss actual
     │    │     └────────────────────────────── metric keyword
     │    └────────────────────────────────── batches entrenadas
     └───────────────────────────────────── worker ID
```

### Niveles de Detalle

- **verbose=False**: Solo conexión/desconexión
- **verbose=True**: Log cada 10 batches

---

## Recursos y Memoria

### RAM por Worker

```
Batch X:          64 × 3 × 224 × 224 × 4 bytes = 12 MB
Batch Y:          64 × 8 bytes = 512 bytes
Features:         64 × 512 × 4 bytes = 128 KB
Logits:           64 × 1000 × 4 bytes = 256 KB
Gradients (CNN):  ~44 MB (mismas dimensiones que parámetros)
Gradients (MLP):  ~4.5 MB

Buffer Prefetch:  prefetch_batches × 12 MB = 4 × 12 = 48 MB

TOTAL: ~100-120 MB por Worker (típico)
```

### CPU vs GPU

- **CPU**: ~500ms por batch (foward+backward)
- **CUDA**: ~50-100ms per batch
- **MPS (Apple)**: ~80-150ms per batch

---

## Posibles Problemas

| Problema | Esta causa | Solución |
|---|---|---|
| Worker se desconecta repentinamente | Timeout en recv (PS muerto) | Reiniciar PS |
| Loss = NaN | Gradientes explotan | Reducir LR o batch size |
| Memory error | Buffer demasiado grande | Reducir `--prefetch` |
| muy lento | Throughput bajo | Aumentar batch size o usar GPU |
| Accuracy = 0% siempre | MLP no se actualiza | Ver métricas del PS |

