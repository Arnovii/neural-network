# Flujo de Entrenamiento Distribuido Asincrónico

## Ciclo Completo: Paso a Paso

```
TIEMPO         PARAMETER SERVER                       WORKER 0
────────────────────────────────────────────────────────────────────

t=0      [Esperando conexión de Workers]             [Conectando]
         ps.listen()                                 socket.connect()
                                                     send(READY)
         ◄──── READY
t=1      Asigna wid=0
         send(WORKER_ID=0)
         ────► WORKER_ID

t=2      Distribuye CNN
         send(CNN_WEIGHTS)
         ────► CNN_WEIGHTS
                                                      recv(CNN_WEIGHTS)
                                                      cnn.load_weights()
                                                      send(CNN_ACK)
         ◄──── CNN_ACK

t=3      send(START)
         ────► START
                                                      recv(START)
                                                      enter _training_loop()

         ══════════════════════════════════════════╦═════════════════
         LOOP ASINCRÓNICO (indefinido)             ║ iter=1
                                                   ║
         _serve_worker(wid=0):                     ║ send(REQUEST_PARAMS)
         while True:                               ║ ────►
         recv(REQUEST_PARAMS)      ◄───────────────╝
t=4      
         copy mlp_state (thread-safe)
         copy cnn_state
         ver = version (0)
         send(PARAMS) ────────────────────────────► recv(PARAMS)
                                                     mlp_state = {...}
                                                     cnn_state = {...}
                                                     version_read = 0

t=5                                                  _train_batch():
                                                     X, Y = next(stream)  # (64, 3, 224, 224)
                                                     # Forward
                                                     features = CNN(X)    # (64, 512)
                                                     logits = MLP(feats)  # (64, 1000)
                                                     loss = CrossEntropy(logits, Y)
                                                     
t=6                                                  # Backward
                                                     loss.backward()
                                                     
                                                     # SGD local (lr=0.001)
                                                     for param in model.params:
                                                       param -= 0.001 * param.grad
                                                     
                                                     # Métricas
                                                     acc = compute_accuracy(logits, Y)

t=7                                                  Serializar pesos:
                                                     mlp_dict = mlp.state_dict_numpy()
                                                     cnn_dict = cnn.state_dict_numpy()
                                                     
                                                     send(UPDATES)
         recv(UPDATES) ◄────────────────────────────
t=8      
         payload = {
           'mlp_weights': {...},
           'cnn_weights': {...},
           'loss': 8.37,
           'accuracy': 0.00,
           'version_read': 0,
           'batch_size': 64
         }
         
         ═══════════════════════════════════════════╦═══════════════
         _apply_update(wid=0, payload):             ║ [PS ACTUALIZA]
                                                    ║
         staleness = version - version_read         ║ s = 0 - 0 = 0
                   = 0 - 0 = 0                      ║
                                                    ║
         alpha = 1 / (1 + 0.1 * 0)                  ║ α = 1.0
                = 1.0                               ║
                                                    ║
         for key in mlp_state:                      ║ Actualización:
           mlp_state[key] += 1.0 *                  ║  Δmlp = mlp_weights - mlp_state
             (mlp_weights[key] -                    ║  mlp_state += Δmlp
              mlp_state[key])                       ║
         version += 1                               ║ version = 1
         record_metrics(loss, acc, staleness)       ║
         
         ═══════════════════════════════════════════╝ (VUELVE AL LOOP)
         
         send(PARAMS) si hay REQUEST_PARAMS pending
         
         [Ahora version=1, mlp_state contiene
          pesos entrenados por Worker 0]
         
                                                     iter=2
                                                     REQUEST_PARAMS
         ◄────────────────────────────────────────────
t=9      
         version=1, mlp_state=<updated>
         send(PARAMS)
         ────►
                                                     recv(PARAMS)
                                                     version_read = 1
                                                     mlp_state = {...} (updated)
                                                     _train_batch()
                                                     loss se reduce más
                                                     send(UPDATES)
                                                     
         _apply_update()
         staleness = 1 - 1 = 0  (sin antigüedad!)
         version = 2
         [CICLO CONTINÚA INDEFINIDAMENTE]
```

---

## Iter vs Step vs Batch

| Término | Significado | Frecuencia | Ejemplo |
|---|---|---|---|
| **Batch** | N imágenes (64) | Cada ~1ms | [64, 3, 224, 224] tensores |
| **Step** | Forward+Backward de 1 batch | Cada ~100ms | Registrado en PS |
| **Iteración** | REQUEST_PARAMS→train→UPDATES | Cada ~100ms | Equivalente a Step en Async-SGD |
| **Epoch** | Todos los datos una vez | ~5 horas (1M imgs / 64) | **NO EXISTE** en streaming infinito |

**Relación**:
```
1 Iteración = 1 Step (generalmente)
1 Iteración = accum_steps Batches (si accumulation activado)
Indefinidas Iteraciones = 1 "Epoch virtual"
```

---

## Cálculo de Loss y Accuracy

### Loss

```python
# En _train_batch()
X = torch.from_numpy(X_np).to(device)      # (64, 3, 224, 224)
Y = torch.from_numpy(Y_np).to(device)      # (64,) con labels 0-999

features = cnn._model(X)                    # (64, 512)
logits = mlp(features)                      # (64, 1000)

loss = nn.functional.cross_entropy(logits, Y)  # scalar ≈ 6.9-8.0

avg_loss = total_loss / total_n             # Promedio sobre batches
```

**Rango esperado por loss**:
- **Primer batch**: ~8.3 (modelo no entrenado, predicciones aleatorias)
- **Después 100 iters**: ~7.0-6.0 (distribución se asimila a labels)
- **Después 10k iters**: ~2.0-1.0 (modelo converge)
- **Después 100k iters**: <0.5 (overfitting en train)

### Accuracy

```python
# En _train_batch()
correct = (logits.argmax(1) == Y).sum().item()   # Predicciones correctas
acc_pct = 100 * correct / len(Y)                 # Porcentaje

# Rango: 0-100%
# Primer batch: ~0.1% (aleatorio para 1000 clases)
# Después 1k iters: ~5-10%
# Después 100k iters: 40-60%
# Convergencia teórica: ~80%+ (depende de arquitectura)
```

---

## Mecanismo de Corrección de Staleness

### ¿Qué es Staleness?

**Staleness = Antigüedad de la información**: Cuántas versiones de parámetros globales ha habido desde que el Worker leyó los parámetros.

```
PS version timeline:
────────────────────────────────────────────
0         5        10        15        20

Worker A: Lee en v=0, entrena, actualiza en v=15
          staleness = 15 - 0 = 15

Worker B: Lee en v=5, entrena rápido, actualiza en v=6
          staleness = 6 - 5 = 1  (mucho mejor!)
```

### Factor de Corrección: α(s)

```
α(s) = 1 / (1 + λ · s)

donde:
  λ = hiperparámetro (default 0.1)
  s = staleness = version_actual - version_leído
```

**Valores típicos**:

| staleness s | λ=0.0 | λ=0.1 | λ=0.5 | λ=1.0 |
|---|---|---|---|---|
| s=0 (fresco) | α=1.00 | α=1.00 | α=1.00 | α=1.00 |
| s=1 | α=1.00 | α=0.91 | α=0.67 | α=0.50 |
| s=5 | α=1.00 | α=0.67 | α=0.29 | α=0.17 |
| s=10 | α=1.00 | α=0.50 | α=0.17 | α=0.09 |
| s=100 | α=1.00 | α=0.09 | α=0.02 | α=0.01 |

**Interpretación**:
- λ=0: Sin corrección (puro Async-SGD, más rápido pero inestable)
- λ grande: Gradientes viejos contribuyen menos (más estable pero lento)

### Aplicación en el PS

```python
def _apply_update(wid, payload):
    version_read = payload['version_read']    # Versión que leyó el Worker
    mlp_weights = payload['mlp_weights']      # Parámetros tras entrenar
    
    staleness = max(0, self._version - version_read)
    alpha = 1.0 / (1.0 + self.staleness_lambda * staleness)
    
    # Actualización: θ_new = θ + α·Δθ
    for key in self._mlp_state:
        if key in mlp_weights:
            delta = mlp_weights[key] - self._mlp_state[key]
            self._mlp_state[key] += alpha * delta
    
    self._version += 1
    self._metrics.add(loss, acc, staleness)
```

---

## Estados Principales del Sistema

```
WORKER ESTADOS:
════════════════════════════════════════════════════════════════

[Inicio] ──socket.connect──► [Conectando]

[Conectando] ──send(READY)──► PS
                              ◄──recv(WORKER_ID)

[Conectando] ──recv(CNN)──► [Cargando CNN]

[Cargando CNN] ──send(CNN_ACK)──► PS
                                  ◄──recv(START)

[Entrenando] ──loop repetido─── ┐
             REQUEST_PARAMS     │
             ├─► recv(PARAMS)   │ Cada 100-500ms
             ├─► _train_batch() │
             ├─► _serialize()   │
             └─► send(UPDATES)  │
                                Vuelve al inicio del loop


PS ESTADOS:
════════════════════════════════════════════════════════════════

[Off] ──init()──► [Offline]

[Offline] ──listen()──► [Listening]

[Listening] ──recv(READY from Worker)──► [Handshaking]

[Handshaking] ──send(START)──► [Serving Workers]

[Serving Workers] ──recv(REQUEST_PARAMS / UPDATES)──► (loop)

[Serving Workers] ──recv(STOP signal)──► [Shutdown]
```

---

## Sincronización de Parámetros

### Inicialización

1. PS inicializa CNN (ResNet-18 con ImageNet weights)
2. PS inicializa MLP (Kaiming init., random)
3. Worker conecta → recibe CNN_WEIGHTS del PS
4. Worker espera START → entra en training_loop
5. Worker solicita PARAMS → recibe mlp_state actual + cnn_state actual

### Durante Entrenamiento

**Cada iteración (indefinido)**:
1. Worker solicit parámetros: `REQUEST_PARAMS`
2. PS responde: `PARAMS` (mlp_state actual, cnn_state, version, lr)
3. Worker entrena 1 batch → genera Δθ
4. Worker envía: `UPDATES` (nuevos pesos, losses, version_read)
5. PS aplica actualización asincrónica: `θ_new = θ + α(s)·Δθ`
6. **VUELVE A PASO 1** (sin esperar a otros Workers)

### Garantía: No Divergence (Teórica)

Con:
- α(s) ≤ 1.0 (actualización es contractive)
- λ > 0 (staleness atenuada)
- Learning rate pequeño (0.001)
- CNNpráct congelada (no explota)

→ El sistema debería converger (en línea recta, no necesariamente a óptimo global)

---

## Ejemplo Numérico: 3 Workers Entrenando en Paralelo

```
Timeline (ms):

t=0    W0 REQUEST_PARAMS (v=0)    W1 [idle]              W2 [idle]
       W0 recv PARAMS (v=0)       W1 REQUEST_PARAMS (v=0) W2 [idle]
       
t=50   W0 training batch           W1 recv PARAMS (v=0)   W2 REQUEST_PARAMS (v=0)
       W0 [compute]               W1 training            W2 recv PARAMS (v=0)

t=100  W0 send UPDATES (v=0→v=1) W1 [compute]           W2 training
       PS aplica: α(0) = 1.0
       mlp_state se actualiza
       version = 1
       
       W0 REQUEST_PARAMS (v=1)    W1 send UPDATES (v=0)  W2 [compute]
                                   PS aplica: α(1) = 0.91
                                   version = 2

t=150  W0 recv PARAMS (v=1)       W1 REQUEST_PARAMS (v=1) W2 send UPDATES (v=0)
       W0 training                                        PS aplica: α(2) = 0.83
                                                          version = 3

t=200  W0 [compute]               W1 recv PARAMS (v=1)    W2 REQUEST_PARAMS (v=3)
                                  W1 training            W2 recv PARAMS (v=3)

[Patrón: Cada Worker avanza asincronicamente, PS actualiza cada vez que recibe]
```

**Conclusión en t=200ms**:
- W0: Completó 2 iteraciones (staleness máximo 0)
- W1: Completó 1 iteración (staleness = 1 en su segunda iter)
- W2: Completó 1 iteración (staleness = 0 en su primera iter)
- PS version: 3 (tres actualizaciones aplicadas)
- **Throughput**: 3 updates en 200ms = 15 updates/s

---

## Posibles Problemas Durante Entrenamiento

| Problema | Causa | Solución |
|---|---|---|
| **Loss constante** | MLP no entrena | Verificar learning rate (default 0.001) |
| **Loss NaN/Inf** | Gradientes explotan | Reducir LR, revisar inicialización |
| **Memory error** | Buffer prefetch demasiado grande | Reducir `--prefetch` |
| **Worker desconecta** | Timeout en recv | Aumentar timeout, revisar red |
| **Loss diverge** | Staleness Lambda muy bajo | Aumentar `--staleness-lambda` |
| **Lentitud** | Network overhead | Reducir `--batch-size` (menos datos/iter) |

