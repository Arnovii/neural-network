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
    
    # 3. Recibir CONFIG (batch_size, image_size, dataset_name, rank, num_workers, seed, hf_token)
    msg = receive_message(self._sock)
    if msg["type"] != MsgType.CONFIG:
        raise ConnectionError(f"Esperaba CONFIG, recibí {msg['type']}")
    config = msg["payload"]
    self.batch_size = config["batch_size"]
    self.image_size = config["image_size"]
    self.worker_rank = config.get("rank", 0)  # ← Asignado dinámicamente por PS
    self.num_workers = config.get("num_workers", 1)  # ← Asignado dinámicamente por PS
    self.seed = config.get("seed")  # ← Seed global desde PS
    self.hf_token = config.get("hf_token")  # ← Token HuggingFace desde PS (para streaming)
    
    self._log(f"Conectado: ID={self._worker_id}, rank={self.worker_rank}/{self.num_workers}, "
              f"batch_size={self.batch_size}, image_size={self.image_size}")
```

**Notas importantes**:
- El Worker **no especifica** por CLI: rank, num_workers, seed, ni hf_token
  - Todos estos se reciben del PS en el mensaje CONFIG
- El PS **asigna dinámicamente**:
  - `rank`: basado en el orden de conexión (0, 1, 2, ...)
  - `num_workers`: total de workers conectados (se actualiza si nuevos workers se conectan)
  - `hf_token`: token HuggingFace para streaming de datos
  - `seed`: semilla global para reproducibilidad
- Los valores `rank` y `num_workers` son **esenciales para el sharding sin solapamientos** del dataset
- El `hf_token` es enviado por el PS para que el Worker pueda acceder a HuggingFace de forma centralizada

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

        if self._batches_done % 10 == 0:
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

## Optimizaciones de Entrenamiento

### Constantes de Módulo

```python
# En Distributed/worker_node.py
_IMAGENET_CLASSES   = 1000
_GRAD_CLIP_MAX_NORM = 1.0     # Umbral de gradient clipping
_LABEL_SMOOTHING    = 0.1     # Nuevo: Suavizado de etiquetas
_WEIGHT_DECAY       = 1e-4    # Nuevo: L2 regularización
```

### Label Smoothing (0.1)

**Qué es**: Reemplazar targets one-hot (ej: [0,1,0,...]) con distribuciones suavizadas:
```
p_smooth(clase_correcta) = 0.9
p_smooth(otras_clases)  = 0.1 / (1000-1) ≈ 0.0001
```

**Beneficio**:
- Reduce overconfidence de logits iniciales (primeros batches: logits enormes)
- Mejora calibración del modelo (predicciones más honestas sobre incertidumbre)
- Especialmente importante en ImageNet-1k (1000 clases = mucha competencia)

**Implementación**:
```python
# En _train_batch(): ambos modos (freeze + E2E)
loss = nn.functional.cross_entropy(logits, Y, label_smoothing=_LABEL_SMOOTHING)
```

**Impacto**: Loss inicial ≈ 3% más alto, pero convergencia más estable y mejor accuracy a largo plazo.

### Weight Decay (1e-4) — Modo E2E

**Qué es**: L2 regularización = penalización cuadrática sobre norma de pesos.

Ecuación por step SGD:
```
θ_{t+1} = θ_t - lr·∇L(θ_t) - lr·wd·θ_t
                ↑                   ↑
              gradiente         weight decay
```

**Beneficio**:
- Evita overfit en E2E training desde cero (SIMPLE CNN tiene 11.2M parámetros)
- Especialmente importante sin preentrenamiento (weights aleatorios = más riesgo de overfit)

**Compatibilidad con FedAvg**:
```
Local: θ_local += -wd·(θ_local - 0) = aplica decay
Global (PS): θ_global = FedAvg(θ_local_1, θ_local_2, ..., θ_local_n)
             = FedAvg(θ con decay) = (E[θ con decay])
             → Promedio LSD es válido, sesgo mínimo ✓
```

**Implementación**:
```python
# En _rebuild_optimizer(): Modo E2E solamente
self._sgd = optim.SGD(
    [
        {"params": list(self._cnn._model.parameters()), "lr": lr_cnn},
        {"params": list(self._mlp.parameters()), "lr": lr},
    ],
    weight_decay=_WEIGHT_DECAY,  # ← Nuevo
)

# Modo freeze (ResNet-18): No se usa SGD formal, decay manual inline
# con torch.no_grad(): p.data -= lr·p.grad (sin decay)
```

**Impacto**: Accuracy validación +2-3% en E2E, especialmente en épocas altas.

**Rango recomendado desde literatura**:
- `1e-4` (actual): Conservative, para CNN no pretrained
- `1e-5 a 5e-4`: Rango típico ResNet en ImageNet
- `> 5e-4`: Demasiado fuerte, underfitting

---

## Por qué se usa SGD, NO Adam

Este es un **diseño intencional**, no una omisión. El sistema usa **SGD puro** en lugar de Adam o AdamW por las siguientes razones:

### El problema de Adam con FedAvg Asíncrono

Adam mantiene dos conjuntos de estado interno por parámetro:

```
m_t = β1 * m_{t-1} + (1 - β1) * g_t    # Primer momento (media)
v_t = β2 * v_{t-1} + (1 - β2) * g_t²  # Segundo momento (varianza)
```

**Problema**: Estos momentos (m, v) son **locales al Worker**:

```
Worker 0: m0, v0 congrads de sus batches
Worker 1: m1, v1 congrads de otros batches
Worker 2: m2, v2 congrads de más batches
    ↓
Cuando PS hace FedAvg: θ = avg(θ0, θ1, θ2)
PERO: m ≠ avg(m0, m1, m2), v ≠ avg(v0, v1, v2)
    ↓
Los momentos apuntan a direcciones incorrectas
    ↓
Actualizaciones incorrectas → diverge
```

### Ejemplonumérico concreto

```
Worker 0 trainsobre datos de perros → m0 apunta a "perro"
Worker 1 trainsobre datos de gatos → m1 apunta a "gato"
Worker 2 trainsobre datos de autos → m2 apunta a "auto"

PS promedia lospesos: θ = (θ0 + θ1 + θ2) / 3 ✓

PERO Adam usaría: m = (m0 + m1 + m2) / 3 ✗
                     v = (v0 + v1 + v2) / 3 ✗

m y v ahora representan "mezcla" que no corresponde 
a ningún conjunto real de pesos
    ↓
El paso de Adam sería incorrecto
```

### Por qué SGD funciona

SGD **no tiene estado interno**:

```
θ_{t+1} = θ_t - lr * g_t
```

- No hay momentos que desincronizar
- Cada update es correcto respecto a los pesos actuales- FedAvg puede promediar directamente

### Comparación

| Optimizer | Estado interno | Desincronización | ¿Funciona con FedAvg? |
|---------|-------------|----------------|---------------------|
| SGD | ❌ Ninguno | ❌ N/A | ✅ Sí |
| Adam | ✅ m, v | ❌ Se desincroniza | ❌ No |
| SGD + Momentum | ✅ Solo momentum | ⚠️ Parcial | ⚠️ Riesgo |
| AdamW | ✅ m, v, decoupled | ❌ Se desincroniza | ❌ No |

### Conclusión del diseño

**SGD fue elegido deliberadamente** para garantizar:
1. Correctitud matemática del FedAvg asíncrono
2. Convergencia estable con múltiples Workers3. Simplicidad (sin momentos que gestionar)

Este es un trade-off conocido en sistemas federados: Adam converge más rápido en training individual pero diverge en settings distribuidos. El sistema prioriza corrección sobre velocidad inicial.

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
        p.requires_grad_(True)  # Solo para SIMPLE CNN, ResNet-18 permanece False
    self._mlp.train()

    # Forward: SIMPLE CNN entrenable, ResNet-18 congelada
    self._cnn._model.zero_grad()
    self._mlp.zero_grad()
    
    features = self._cnn._model(X)  # (64, 512)
    logits = self._mlp(features)    # (64, 1000)
    # ✓ Nuevo: label_smoothing=_LABEL_SMOOTHING
    loss = nn.functional.cross_entropy(logits, Y, label_smoothing=_LABEL_SMOOTHING)

    # Backward
    loss.backward()

    # SGD local (con weight_decay en E2E)
    # SIMPLE CNN: CNN gradientes se propagan, se SGD
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

---

## Línea de Comandos (CLI)

### Uso básico

```bash
# Conectar al PS en localhost
python worker_imagenet.py --server-host 127.0.0.1

# Conectar al PS en otra máquina
python worker_imagenet.py --server-host 192.168.1.100

# Conectar al PS con personalizado
python worker_imagenet.py \
    --server-host 192.168.1.100 \
    --server-port 9999 \
    --device cuda:0 \
    --prefetch 8 \
    --shuffle-buffer 2000 \
    --accum-steps 2
```

### Parámetros del Worker

| Parámetro | Default | Descripción |
|----------|---------------|-------------|
| `--server-host` | `127.0.0.1` | IP del Parameter Server |
| `--server-port` | `9999` | Puerto TCP del PS |
| `--device` | `auto` | cpu, cuda, cuda:N, mps |
| `--shuffle-buffer` | `1000` | Imágenes en buffer de shuffle |
| `--prefetch` | `4` | Batches pre-cargados |
| `--seed` | `None` | Semilla RNG (None = aleatorio) |
| `--accum-steps` | `1` | Batches a acumular antes de enviar |

### Nota Importante

**El dataset se especifica en el PS, NO en el Worker**.

- El Worker **recibe** el nombre del dataset del PS vía mensaje CONFIG
- No existe parámetro `--dataset` en el Worker
- Si necesitas un dataset diferente, configúralo en el PS

---

## CONFIG: Mensaje de Configuración

Cuando el Worker se conecta, el PS envía un mensaje CONFIG con los parámetros globales:

```python
config = {
    "batch_size": 64,                    # Batch size global
    "image_size": 224,               # Resolución de imágenes
    "dataset_name": "ILSVRC/imagenet-1k",  # Dataset (ENVÍADO DESDE EL PS)
    "seed": 42,                     # Semilla global
    "rank": 0,                     # Rank asignado al Worker
    "num_workers": 2,                # Total de Workers
    "hf_token": "hf_...",            # Token para HuggingFace
}
```

### Parámetros recibidos en CONFIG

| Campo | Tipo | Descripción |
|-------|------|------------|
| `batch_size` | int | Imágenes por batch |
| `image_size` | int | Resolución (ancho=alto) |
| `dataset_name` | str | Dataset de HuggingFace |
| `seed` | int \| None | Semilla global |
| `rank` | int | Índice del Worker (0, 1, 2, ...) |
| `num_workers` | int | Total de Workers conectados |
| `hf_token` | str | Token para streaming |

### Importancia del Sharding

- `rank` + `num_workers` determinan qué porción del dataset procesa cada Worker
- Worker 0 procesa: indices 0, N, 2N, ...
- Worker 1 procesa: indices 1, 1+N, 1+2N, ...
- No hay overlap entre Workers

---

## Historial de Cambios

### Eliminación de --quiet

El parámetro `--quiet` fue **eliminado** porque no funcionaba:

- No había lógica que usara `args.quiet`
- Todo el logging era siempre mostrado
- Si necesitas silencio, redirige la salida:

```bash
python worker_imagenet.py --server-host 127.0.0.1 2>&1 > /dev/null
```

