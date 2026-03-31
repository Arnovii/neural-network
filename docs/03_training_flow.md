# 03. Flujo de Entrenamiento: Ejecución por Época

## Visión General (Macro)

Cada **época** es una unidad atómica de sincronización. El flujo es:

```
Época N:
1. PS envía PARAMS a todos los Workers → BROADCAST (downlink)
2. Cada Worker (en paralelo, desincronizado):
   - Recibe PARAMS
   - Reconstruye indices (determinista, con seed)
   - Extrae features (caché en PRECOMPUTED)
   - Forward + backward MLP
   - Envía GRADIENTS → UPLINK
3. PS recibe GRADIENTS de todos Workers (orden arbitrario)
4. PS promedia: grad_avg = sum(grad) / N
5. PS actualiza pesos: θ ← θ - lr * grad_avg
6. PS evalúa en test (si datos disponibles)
7. PS imprime progreso
8. Si epoch < N_epochs, vuelve a paso 1
```

**Duración típica (PRECOMPUTED, 3 Workers)**:
- Broadcast PARAMS: ~100-200 ms
- Compute (parallelo en Workers): ~2000 ms (dominante)
- Collect GRADIENTS: ~300-500 ms
- Update parametros PS: ~50 ms
- **Total por época**: ~2.5-3 segundos

---

## Desglose Temporal Detallado (PRECOMPUTED Mode)

### T=0ms: PS Envía PARAMS

```python
# En Parameter Server
params = {
    "W1": np.array(..., shape=(256, 512), dtype=float32),  # ~1 MB
    "b1": np.array(..., shape=(256,), dtype=float32),      # ~1 KB
    "W2": np.array(..., shape=(128, 256), dtype=float32),  # ~128 KB
    "b2": np.array(..., shape=(128,), dtype=float32),      # ~512 B
    "W3": np.array(..., shape=(10, 128), dtype=float32),   # ~5 KB
    "b3": np.array(..., shape=(10,), dtype=float32),       # ~40 B
}
# Total: ~1.135 MB (float32)

message = {
    "type": "PARAMS",
    "payload": {
        "epoch": 5,
        "params": params,
        "seed": 12345,  # para reproducibilidad
    }
}

# Serialize y enviar a cada Worker (en serie)
for worker_id in active_workers:
    send_message(worker_socket[worker_id], message)
    # broadcast toma ~100-200ms por worker
```

**Red**: 1.135 MB × 3 workers = 3.4 MB downlink (en LAN es insignificante)

**Broadcast es secuencial** (no paralelo): PS envía a Worker 0, luego Worker 1,luego Worker 2. Pero TCP no bloquea indefinidamente, así que es rápido (~100ms/worker).

### T=100-300ms: Workers Reciben PARAMS

```python
# En cada Worker (en paralelo, pero desfasado un poco)
msg = receive_message(socket)
assert msg["type"] == "PARAMS"

epoch_num = msg["payload"]["epoch"]
new_params = msg["payload"]["params"]
seed = msg["payload"]["seed"]

print(f"Worker {worker_id}: Received PARAMS for epoch {epoch_num}")
```

**Nota**: Los Workers no esperan entre sí. Worker 0 recibe a T=100ms, Worker 1 a T=200ms, etc. Luego comienzan a calcular **sin sincronización**.

### T=300-2300ms: Workers Computan Localmente (Paralelo)

```python
# En cada Worker i (en paralelo, desincronizado)

# Paso 1: Reconstruir índices (determinista)
rng = np.random.RandomState(seed)
shuffled = np.arange(n_train)  # 50000
rng.shuffle(shuffled)

# Dividir round-robin: cada worker i toma elementos donde idx % n_workers == i
my_indices = [idx for idx in shuffled if idx % n_workers == i]
# Si n_train=50000, n_workers=3: Worker 0 obtiene indices [0, 3, 6, 9, ...]
# Cada worker: ~16667 muestras

# Paso 2: Extraer features (CACHÉ)
if PRECOMPUTED:
    # El caché fue generado al inicio con la CNN congelada
    # Hash de pesos CNN es siempre el mismo → cache hit 100%
    X_batch = X_features[my_indices]  # ~16667×512 array, subindexing en memoria
    Y_batch = Y_raw[my_indices]  # ~16667 array
    # Tiempo: ~50 ms (indexing en RAM)
elif END_TO_END:
    # Pesos CNN cambian cada época → cae cache
    X_batch_raw = X_raw[my_indices]
    X_batch = cnn.extract(X_batch_raw)  # Forward pass de 16667 imágenes
    Y_batch = Y_raw[my_indices]
    # Tiempo: ~2000 ms (CNN forward)

# Paso 3: Forward + Backward MLP (NumPy, local)
grads, loss, accuracy = mlp.forward_and_gradients(
    params=new_params,
    X=X_batch,  # (16667, 512)
    Y=Y_batch,  # (16667,)
)
# Forward: 1 multiplication W1@X (~16667×512×256 ops) + ReLU + W2@A1 + ... = O(n_hidden^2)
# Backward: similar
# Tiempo: ~1000-1500 ms (NumPy con BLAS)

print(f"Worker {worker_id}: Epoch {epoch_num} loss={loss:.4f} accuracy={accuracy:.2f}%")
```

**Parallelismo**: Los 3 Workers hacen esto en paralelo. En máquinas separadas, toman ~2000ms. En threads (CPU-bound), más como 6000ms secuencial pero SO intercala.

**Punto clave**: Los datos (X_batch, Y_batch) nunca salen de la máquina del Worker. Solo los gradientes salen.

### T=2300-2800ms: Workers Envían GRADIENTS

```python
# En cada Worker (tan pronto como terminan de computar)
message = {
    "type": "GRADIENTS",
    "payload": {
        "worker_id": my_id,
        "epoch": epoch_num,
        "gradients": grads,  # Dict[W1, b1, W2, b2, W3, b3] (mismo shape que pesos)
        "loss": loss,
        "accuracy": accuracy,
    }
}

send_message(socket, message)

# Serialización Pickle: 1.135 MB (mismo tamaño que pesos)
# Pero Workers terminan a tiempos distintos:
#   - Worker 0 termina a T≈2100ms → envía T≈2200ms
#   - Worker 1 termina a T≈2200ms → envía T≈2300ms  (llegó después)
#   - Worker 2 termina a T≈2400ms → envía T≈2500ms  (llegó después)
```

**Red ascendente**: 1.135 MB × 3 workers = 3.4 MB uplink (en LAN es trivial).

**Desincronización**: Los Workers NO esperan entre sí. Conforme terminan, envían. El PS recepciona en el orden que llegan (no necesariamente 0, 1, 2).

### T=2800ms: PS Recibe GRADIENTS (No determinista)

```python
# En Parameter Server (mientras Workers computaban)
ps.epoch_gradients = {}  # Buffer para esta época

# Recibir de cada worker (bloqueante en socket.recv)
expected_workers = [0, 1, 2]
for _ in range(len(expected_workers)):
    msg = receive_message()  # Bloqueante hasta que llega UN mensaje
    
    if msg["type"] == "GRADIENTS":
        worker_id = msg["payload"]["worker_id"]
        epoch = msg["payload"]["epoch"]
        gradients = msg["payload"]["gradients"]
        loss = msg["payload"]["loss"]
        accuracy = msg["payload"]["accuracy"]
        
        ps.epoch_gradients[worker_id] = gradients
        ps.epoch_metrics[worker_id] = (loss, accuracy)
        
        print(f"[PS] Received GRADIENTS from Worker {worker_id} epoch {epoch}")

# Cuando tenemos de los 3, procedemos a siguiente paso
assert len(ps.epoch_gradients) == 3
```

**Orden de llegada**: Típicamente el orden es 0, 1, 2, pero NO garantizado. Un Worker más lento puede llegar último.

### T=2800-2850ms: PS Promedia Gradientes

```python
# En Parameter Server
# Promediar gradientes (Batch SGD equivalente)

averaged_grads = {}
all_param_names = list(ps.epoch_gradients[0].keys())  # ["W1", "b1", "W2", "b2", "W3", "b3"]

for param_name in all_param_names:
    grad_stack = []
    for worker_id in expected_workers:
        grad_stack.append(ps.epoch_gradients[worker_id][param_name])
    
    # Stack: 3 arrays, each (256, 512) for W1 → (3, 256, 512)
    grad_stack = np.array(grad_stack)
    averaged_grads[param_name] = np.mean(grad_stack, axis=0)

# Result: averaged_grads ha misma shape que pesos
# averaged_grads["W1"] shape (256, 512), promedio de los 3 workers
```

**Matemática**: Si Worker 0 tiene batch de 16667, Worker 1 de 16667, Worker 2 de 16667, cada uno calcula gradientes sobre su batch:

$$\nabla_i = \frac{1}{n_i} \frac{\partial L(B_i)}{\partial \theta}$$

PS promedia:
$$\bar{\nabla} = \frac{1}{3}(\nabla_0 + \nabla_1 + \nabla_2) = \frac{1}{50000}\left(\frac{50000}{3} \sum_{i=0}^{2} \nabla_i \right)$$

Esto es **exactamente** equivalente a batch SGD con batch size 50000 dividido en 3 sub-batches.

### T=2850-2900ms: PS Actualiza Pesos

```python
# En Parameter Server
learning_rate = 0.01

mlp.apply_gradients(params, averaged_grads, learning_rate)

# Inside mlp.apply_gradients():
for param_name in params:
    params[param_name] -= learning_rate * averaged_grads[param_name]

# E.g., params["W1"] -= 0.01 * averaged_grads["W1"]
# In-place update. Total: ~50ms de NumPy
```

### T=2900-2950ms: PS Evalúa en Test (Opcional)

```python
# Si se proporcionaron datos de test
if X_test is not None:
    test_accuracy, test_loss = mlp.evaluate(params, X_test_features, Y_test)
    # evaluate() es solo forward pass, sin backward
    # Tiempo: ~500ms para 10000 imágenes
else:
    test_accuracy = None
    test_loss = None
```

**Observación**: El test se evalúa en el MISMO proceso PS (CPU principal). Toma tiempo. Para acelerar, podría async-ificarse, pero por ahora es sincrónico.

### T=2950-2970ms: PS Callback y Logging

```python
# En Parameter Server / ps_gui.py en hilo de UI
ps.on_epoch_end(
    epoch=5,
    total_epochs=10,
    train_accuracy=94.3,  # promedio de los 3 workers
    train_loss=0.234,
    test_accuracy=93.8,
    test_loss=0.248,
)

# ps_gui.py actualiza gráficos, ps_terminal.py imprime
print(f"Epoch 5/10 | Train Acc=94.3% Loss=0.234 | Test Acc=93.8% Loss=0.248")
```

### T=2970ms: Vuelve a Paso 1 (Siguiente Época)

Si remain epochs > 0, volver al envío de PARAMS con epoch_num+1 y nueva semilla.

---

## Timing Breakdown (PRECOMPUTED, 3 Workers, LAN)

| Fase | Duración | Cuello Botella |
|------|----------|---|
| PS → PARAMS broadcast | 200 ms | TCP pero rápido |
| Workers reciban PARAMS | 100 ms | Propagación |
| Compute MLP (Worker) | **2000-2500 ms** | ← DOMINANTE |
| Workers → GRADIENTS upload | 200-300 ms | TCP |
| PS average+update | 50 ms | CPU |
| PS evaluate test | 500 ms | (si test data) |
| **Total/epoch** | **~3-3.5 s** | Compute |

**Conclusión**: El 80% del tiempo es cómputo local de forward/backward. La red es marginal (LAN rápida).

---

## Timing Breakdown (END-TO-END, 3 Workers, GPU/CUDA)

| Fase | Duración | Notas |
|------|----------|---|
| PS → PARAMS + CNN_WEIGHTS broadcast | 300 ms | CNN weights = 50-150 MB |
| Workers reciban | 150 ms | Red |
| Test features extraction (Worker 0 solo) | 300 ms | GPU fast |
| CNN feature extraction (cada Worker) | **3000-5000 ms** | ← NUEVA (dominant) |
| Compute MLP (Worker) | 1500-2000 ms | Similar a PRECOMPUTED |
| Workers → GRADIENTS + CNN_GRADIENTS | 500 ms | Datos más grandes |
| PS average+update | 100 ms | Dos sets |
| **Total/epoch** | **~6-8 s** | CNN computing dominates |

---

## Variaciones Importantes

### Caso: Network Latency (WAN, 100ms latency)

```
Broadcast: 200 ms (antes)    → 400 ms (ahora)
Collect:   200-300 ms        → 400-500 ms
Total overhead: +400 ms

PRECOMPUTED: 3.5s → 3.9s (10% overhead)
END-TO-END:  7.5s → 7.9s (5% overhead)
```

**Conclusión**: Latencia de red importa poco si el cómputo es largo.

### Caso: Straggler (Worker 2 lento)

```
Época típica:
  T=0ms:     PS envía PARAMS
  T=300ms:   Workers comienzan compute
  T=2300ms:  Workers 0,1 terminan, envían GRADIENTS
  T=2500ms:  PS recibe de 0,1
  T=3500ms:  Worker 2 termina (1s detrás), envía
  T=3500ms:  PS recibe de 2, ahora puede promediar
```

**Problema**: PS está bloqueado en `receive_message()` esperando Worker 2. La época tarda 3.5s en lugar de 2.5s.

**Solución** (no implementada): Timeout. Si PS no recibe de todos en 10s, da por perdido ese Worker.

---

## Determinismo y Reproducibilidad

### Seed-based Partitioning

```python
# Época 0, seed=42
rng = RandomState(42)
shuffled_idx = arange(50000)
rng.shuffle(shuffled_idx)  # [3124, 18734, 202, ...] (pseudoaleatorio, determinista)

# Worker 0: idx[0], idx[3], idx[6], ...
# Worker 1: idx[1], idx[4], idx[7], ...
# Worker 2: idx[2], idx[5], idx[8], ...

# Ejecutar DOS VECES CON EL MISMO SEED. RESULTADO: IDÉNTICO.
# Ejecutar CON DISTINTO SEED: Distinto subset, pero orden determinista.
```

**Garantía**: Si ejecutas dos sesiones con los mismo hiperparámetros y seeds, el entrenamiento es idéntico epoch a epoch (hasta máquina epsilon, float precision).

### Diferencia PRECOMPUTED vs END-TO-END en Reproducibilidad

En **PRECOMPUTED**:
- CNN congelada, features cacheados → determinista 100%
- Volver a ejecutar en 6 meses → resultados idénticos

En **END-TO-END**:
- CNN trainable, pesos iniciales dependen de seed
- Volver a ejecutar → resultados estadísticamente similares pero no idénticos (float precision)
- Pero si usas el MISMO state snapshot (pesos en checkpoint), reproducible

---

## Ejemplo Concreto: Época 3

Vamos a trazar una epoch con números reales:

```
Hyperparámetros:
- n_train = 50000
- n_workers = 3
- n_epochs = 10
- learning_rate = 0.01
- batch_norm en CNN

Época 3 (index 0-based):
  seed_epoch = 42 + 3 = 45

T=0:
  PS envía PARAMS a Workers con seed=45

T=300ms:
  Worker 0 recibe, comienza compute
  shuffled_0 = shuffle(arange(50000), seed=45)
    → [3040, 2994, 18203, 1024, ...]
  my_indices_0 = [i for i in shuffled_0 if i % 3 == 0]
    → [3040, 18203, ...] (16667 elementos)
  X_0 = X_features[my_indices_0]  # (16667, 512), MLP input
  Y_0 = Y_raw[my_indices_0]  # (16667,), MLP labels

  Worker 1, 2 análogamente con my_rank=1,2

T=500ms:
  Forward pass Worker 0: Z1 = W1 @ X_0.T → (256, 16667)
    A1 = relu(Z1) → (256, 16667)
    ...
    A3 = softmax(...) → (10, 16667)
  Loss: -E[log P(y|x)] = 0.328
  Accuracy: 91.2% (15218 correctas de 16667)

T=1500ms:
  Backward Worker 0: δ3 = A3 - one_hot(Y_0) → (10, 16667)
    dW3 = (1/16667) * δ3 @ A2.T → (10, 128)
    ...
  Gradients computed

T=2200ms:
  Worker 0 envía GRADIENTS: {"W1": (256,512), "b1": (256,), ...}

T=2300ms:
  Worker 1 envía GRADIENTS

T=2500ms:
  PS recibió W0, W1. Esperando W2.

T=2500ms:
  Worker 2 termina compute.

T=2600ms:
  Worker 2 envía GRADIENTS

T=2600ms:
  PS promedia:
    grad_avg_W1 = (W0_grad_W1 + W1_grad_W1 + W2_grad_W1) / 3
    ... (foreach param)

T=2650ms:
  PS updateParams: W1 -= 0.01 * grad_avg_W1

T=2700ms:
  PS evalúa test: 10000 images → 93.8% accuracy

T=2720ms:
  Print: "Epoch 3/10 | Train Acc=91.2% Loss=0.328 | Test Acc=93.8%"

T=2720ms:
  Vuelve a PARAMS con epoch=4, seed=46
```

---

## Sincronización de Características Asincrónicas

A pesar de que los cálculos de los Workers son paralelos y asincronizados, el PS **asegura global synchronization** en estos puntos:

1. **CNN_READY**: A nadie se le envía PARAMS hasta que los N Workers confirmen que extrajeron features
2. **GRADIENTS collection**: A nadie se le envía siguiente PARAMS hasta que los N Workers enviaron GRADIENTS de época actual
3. **Parameter distribution**: Todos los Workers reciben los MISMOS pesos (distribuidos desde PS)

**Invariante**: Después de cada `epoch_end`, todos los Workers están en el MISMO estado (mismo modelo global, aunque datos locales distintos).

