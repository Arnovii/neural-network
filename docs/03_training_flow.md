# 3. FLUJO DE ENTRENAMIENTO

## 🔄 Visión general por época

```
INICIO DE ÉPOCA E
│
├─ [PS] Genera seed_e = random()
├─ [PS] Envía PARAMS (params_mlp, seed_e) a todos los Workers
│
├─ [Worker 0] Usa seed_e para reconstruir índices locales
│   └─ Calcula: forward CNN + MLP → logits
│   └─ Calcula: backward MLP → ∇_W, ∇_b
│   └─ Envía: GRADIENTS(∇_W, ∇_b)
│
├─ [Worker 1] Lo mismo
├─ …
├─ [Worker N] Lo mismo
│
├─ [PS] Recibe gradientes de todos los Workers
│   └─ Promedia: ∇̄_W = (1/N) * Σ ∇_W[i]
│   └─ Actualiza: W ← W − lr * ∇̄_W
│
├─ [PS] Evalúa en datos de prueba
│   └─ Calcula accuracy y pérdida
│
└─ FIN DE ÉPOCA E
```

---

## 📋 Flujo detallado de sesión completa

### **Fase 0: Conexión de Workers**

**Orden**:
1. PS escucha en Puerto 9999
2. Cada Worker se conecta en una máquina diferente
3. Worker envía `READY` (sin ID)
4. PS responde con `WORKER_ID` (0, 1, 2, …)
5. Callback: `on_worker_connected(worker_id, addr)`

```
Worker 0         Worker 1         Worker N         PS
  │                 │                 │              │
  ├─ READY ────────────────────────────────────┐   │
  │                 │                 │         └──►│
  │                 ├─ READY ────────────────────┐  │
  │                 │                 │         └─►│
  │                 │                 │ READY ───┐ │
  │                 │                 │          └►│
  │                 │                 │            │
  │◄─────────────── WORKER_ID (0) ────────────────┤
  │                 │                 │            │
  │                 │◄────── WORKER_ID (1) ────────┤
  │                 │                 │            │
  │                 │                 │◄─ WORKER_ID (N)
```

---

### **Fase 1: Distribución de CNN**

**Orden**:
1. PS crea/carga CNN preentrenada
2. PS serializa pesos CNN: `torch.save(model.state_dict()) → bytes`
3. PS envía `CNN_WEIGHTS` a todos los Workers

**En cada Worker**:
1. Recibe `CNN_WEIGHTS`
2. Reconstruye la CNN con esos pesos
3. En precomputed: congelica CNN, extrae features de 50K imágenes
4. En E2E: habilita CNN para gradientes, no extrae features
5. Envía `CNN_READY` cuando listo

```
PS                                     Workers 0, 1, 2, ...
│                                            │
├─ CNN_WEIGHTS ────────────────────────────►│
│   (arch="simple" o "resnet18")             │
│   (weights_bytes)                          │
│                                            ├─ Cargar CNN
│                                            │
│ (precomputed)                              ├─ Frozen=True
│                                            ├─ Extract 50K imgs → features
│                                            │  (caché si existe)
│                                            │
│ (end_to_end)                               ├─ requires_grad=True
│                                            ├─ NO extraer features
│                                            │
│                                            ├─ CNN_READY
│ ◄─────────────── CNN_READY ────────────────┤
(barrera: espera a TODOS
 antes de continuar)
```

---

### **Fase 2: Solicitud de datos de prueba (E2E)**

**Solo en modo END-TO-END**:

1. PS envía `REQUEST_TEST_FEATURES` al Worker 0
2. Worker 0 extrae features de test con su GPU/CPU
3. Worker 0 envía `TEST_FEATURES` con (X_test_features, Y_test)
4. PS almacena en `_X_test_features` para usar en evaluación

*En precomputed*: Este paso se omite — PS usa datos de prueba raw directamente.

```
PS                                     Worker 0 (only)
│                                            │
├─ REQUEST_TEST_FEATURES ───────────────────►│
│                                            │
│                                            ├─ Extract TEST imgs with CNN
│                                            ├─ Caché (< 0.5s si hash igual)
│                                            │
│ ◄──────────── TEST_FEATURES ───────────────┤
│                  (X_test_features, Y_test)
```

---

### **Fase 3: Entrenamiento (EPOCH LOOP)**

**Orden por cada época**:

```
PS                     Worker 0                Worker 1                ...
│                           │                      │
│ epoch = 0                 │                      │
├─ TRAIN_START ────────────►│                      │
│   (epochs=10,             │                      │
│    n_train=50000,         │                      │
│    n_workers=2,           │                      │
│    training_mode="precomp")                      │
│                           │                      │
├─ ═════════ ÉPOCA 0 ═════════                     │
│                           │                      │
├─ seed_0 = 42              │                      │
│                           │                      │
├─ PARAMS ──────────────────►│                      │
│   epoch=0                 │                      │
│   params={W1, b1, ...}    │                      │
│   seed=42                 │                      │
│   (cnn_params=None)       ├─ Reconstruct indices (seed=42, rank=0, N=2)
│                           ├─ Load features[indices]
│                           ├─ Forward MLP
│   ├─ PARAMS ──────────────────────────►│
│   │ (seed=42 igual para todos)        ├─ Reconstruct indices (seed=42, rank=1, N=2)
│   │                                   ├─ Load features[indices]
│   │                                   ├─ Forward MLP
│   │
│   │                           ├─ Backward MLP → ∇L_0
│   │                           │
│   ◄────── GRADIENTS ──────────┤ ∇L_0 (shape: W1, b1, W2, b2, W3, b3)
│                               │
│   │                           ├─ Backward MLP → ∇L_1
│   │                           │
│   └──────────────────────────────► GRADIENTS
│                                     ∇L_1
│
│ Recibidos: ∇L_0, ∇L_1
│
├─ Promedia: ∇̄ = (∇L_0 + ∇L_1) / 2
├─ Actualiza: W := W - lr * ∇̄
├─ Evalúa: test_acc, test_loss
│
├─ Callback: on_epoch_end(0, 10, train_acc, train_loss, test_acc, test_loss)
│
├─ ═════════ ÉPOCA 1 ═════════
│ (seed_1 = 123, etc.)
│ (repite para cada época hasta epoch = N-1)
│
├─ ═════════ FIN ═════════
│
└─ STOP ────────────────────────────────────────────────►
```

---

## 📊 Detalles de cada hito

### **Reconstrucción de índices (stratified round-robin)**

**En cada Worker, después de recibir seed de la época **:

```python
seed_epoch = 42  # PS envía este valor
n_train = 50000  # Total imágenes
n_workers = 2    # 2 workers en sesión
worker_rank = 0  # Este worker

# Resultado: 25000 índices para este worker
# Distribuido: todas las clases, round-robin entre workers
```

**Algoritmo**:
1. Para cada clase (0-9):
2. Mezcla los índices de esa clase usando seed_epoch
3. Distribuye en round-robin: clase 0 → workers 0, 1, 0, 1, …
4. Worker[rank] toma los índices que le corresponden

**Ejemplo (2 workers, 5000 imgs por clase)**:
```
Clase 0: [0, 1, 2, 3, 4, …, 4999]
Worker 0 obtiene: [0, 2, 4, 6, 8, …] (pares)
Worker 1 obtiene: [1, 3, 5, 7, 9, …] (impares)

Clase 1: [5000, 5001, …, 9999]
Worker 0 obtiene: [5000, 5002, …]
Worker 1 obtiene: [5001, 5003, …]

… (repetir para clases 2-9)

Resultado: Worker 0 tiene 25000 imgs (5000 de cada clase)
           Worker 1 tiene 25000 imgs (5000 de cada clase)
           Ambos tienen datos de todas las clases.
```

**Ventajas**:
- ✅ Balanceo perfecto (cada worker obtiene ~25% del dataset)
- ✅ Todas las clases en cada worker (sin sesgos por clase)
- ✅ Determinístico (seed garantiza reproducibilidad)

---

### **Forward pass (CNN + MLP)**

```
Worker local:
│
├─ Recibir PARAMS: {epoch, params, seed, cnn_params?}
│
├─ X_batch = features[indices]  (shape: 256, 512)  [PRECOMPUTED]
│   O
│   X_batch = CNN(X_raw[indices])  (shape: 256, 512)  [E2E]
│
├─ Forward MLP:
│   Z1 = X_batch @ W1 + b1      (256, 256)
│   A1 = ReLU(Z1)               (256, 256)
│   Z2 = A1 @ W2 + b2           (256, 128)
│   A2 = ReLU(Z2)               (256, 128)
│   Logits = A2 @ W3 + b3       (256, 10)
│
├─ Loss = CrossEntropy(Logits, Y_batch)
```

---

### **Backward pass (gradientes)**

```
Worker local:
│
├─ Backward MLP:
│   dLogits = SoftmaxGrad(Logits, Y_batch)
│   dW3 = A2.T @ dLogits        (128, 10)
│   db3 = sum(dLogits)          (10,)
│   dA2 = dLogits @ W3.T        (256, 128)
│   dZ2 = dA2 * (Z2 > 0)        ReLU backward
│   dW2 = A1.T @ dZ2            (256, 128)
│   db2 = sum(dZ2)              (128,)
│   dA1 = dZ2 @ W2.T            (256, 256)
│   dZ1 = dA1 * (Z1 > 0)
│   dW1 = X_batch.T @ dZ1       (512, 256)
│   db1 = sum(dZ1)              (256,)
│
├─ [E2E ONLY] Backward CNN:
│   dX_batch = dZ1 @ W1.T       (256, 512)
│   Backward through CNN layers
│   Calcula gradientes de pesos CNN
│
├─ Gradients = {dW1, db1, dW2, db2, dW3, db3, [dCNN params]}
│
└─ Enviar GRADIENTS al PS
```

---

### **Sincronización en PS**

```
PS (después de recibir GRADIENTS de todos):

├─ GRADIENTS_dict = {
│    0: {dW1, db1, …},
│    1: {dW1, db1, …},
│    ...
│  }
│
├─ Promediar cada parámetro:
│   ∇̄W1 = (1/N) * Σ(∇W1[i])  para cada i en workers
│   ∇̄b1 = (1/N) * Σ(∇b1[i])
│   ... (todos los parámetros)
│
├─ Aplicar SGD:
│   W1 := W1 - lr * ∇̄W1
│   b1 := b1 - lr * ∇̄b1
│   ... (todos los parámetros)
│
└─ [E2E ONLY] Actualizar CNN también
   CNN_params := CNN_params - lr * ∇̄_CNN
```

---

## ⏱️ Timings típicos (2 workers, CPU)

| Fase | Precomputed | End-to-End |
|------|-------------|-----------|
| CNN_WEIGHTS + CNN_READY | ~45s | ~2s |
| Época 0 (forward+backward) | ~2s | ~25s |
| Época N (caché caliente) | ~2s | ~25s |
| Sincronización PS | ~0.5s | ~0.5s |
| **Total por época** | **~2.5s** | **~25.5s** |
| **10 épocas** | **~25s** | **~255s** |

*Nota*: Con GPU, E2E se reduce a ~5-10s/época.

---

## 🔍 Qué ocurre si algo falla

| Escenario | Efecto |
|-----------|--------|
| Worker se desconecta durante CNN_WEIGHTS | Otros Workers esperan barrera indefinidamente (timeout necesario) |
| Worker envía gradientes tarde | PS espera — bloquea toda la época |
| PS se cae | Todos los Workers quedan en espera indefinidamente |
| Red se interrumpe en mitad de GRADIENTS | Worker levanta excepción, PS detecta desconexión |

**Recomendación**: Implementar timeouts en barrera de sincronización.

---

## 🎯 Flujo simplificado (pseudo-código)

```python
# PS
ps = ParameterServer()
ps.listen()                              # Acepta Workers en background
ps.set_cnn(cnn_final)                   # Carga CNN preentrenada

# Espera a que se conecten workers
workers_connected = wait_for_workers(n_workers=2)

# Sesión de entrenamiento
ps.train(
    epochs=10,
    n_train=50000,
    training_mode="precomputed",
    test_data=(X_test, Y_test)
)
# Internamente:
#   for epoch in range(10):
#       ps.broadcast_params()           # Envía PARAMS + seed
#       gradients = ps.collect_gradients()  # Espera GRADIENTS
#       ps.update_params(gradients)     # Promedia y actualiza
#       test_acc = ps.evaluate()

ps.shutdown()                           # Envía STOP
```

```python
# Worker
worker = WorkerNode(
    server_host="192.168.1.100",
    X_train=X_train,
    Y_train=Y_train,
    training_mode="precomputed"
)

worker.run()  # Iniciar loop persistente
# Internamente:
#   connect_to_ps()
#   while True:
#       msg = receive_message()
#       if msg.type == CNN_WEIGHTS:
#           load_cnn(msg.weights)
#           send_cnn_ready()
#       elif msg.type == TRAIN_START:
#           for epoch in range(epochs):
#               msg = receive_params()
#               indices = reconstruct_indices(msg.seed)
#               gradients = compute_gradients(indices)
#               send_gradients(gradients)
#       elif msg.type == STOP:
#           break
```

---

**Documento**: `docs/03_training_flow.md`  
**Última actualización**: 2026-03-27  
**Nivel**: Intermedio → Avanzado
