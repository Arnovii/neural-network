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
                                                      recv(WORKER_ID)

t=2   Envía configuración
         send(CONFIG={batch_size=64, image_size=224})
         ────► CONFIG
                                                      recv(CONFIG)
                                                      batch_size = 64
                                                      image_size = 224

t=3      Distribuye CNN
         send(CNN_WEIGHTS)
         ────► CNN_WEIGHTS
                                                      recv(CNN_WEIGHTS)
                                                      cnn.load_weights()
                                                      send(CNN_ACK)
         ◄──── CNN_ACK

t=4    send(START)
         ────► START
                                                      recv(START)
                                                      enter _training_loop()

         ══════════════════════════════════════════╦═════════════════
         LOOP ASINCRÓNICO (indefinido)             ║ iter=1
                                                   ║
         _serve_worker(wid=0):                     ║ send(REQUEST_PARAMS)
         while True:                               ║ ────►
         recv(REQUEST_PARAMS)      ◄───────────────╝
t=5    
         copy mlp_state (thread-safe)
         copy cnn_state
         ver = version (0)
         send(PARAMS) ────────────────────────────► recv(PARAMS)
                                                     mlp_state = {...}
                                                     cnn_state = {...}
                                                     version_read = 0

t=6                                                  _train_batch():
                                                     X, Y = next(stream)  # (64, 3, 224, 224)
                                                     # Forward
                                                     features = CNN(X)    # (64, 512)
                                                     logits = MLP(feats)  # (64, 1000)
                                                     loss = CrossEntropy(logits, Y)
                                                     
t=7                                                  # Backward
                                                     loss.backward()
                                                     
                                                     # SGD local (lr=0.001)
                                                     for param in model.params:
                                                       param -= 0.001 * param.grad
                                                     
                                                     # Métricas
                                                     acc = compute_accuracy(logits, Y)

t=8                                                  Serializar pesos:
                                                     mlp_dict = mlp.state_dict_numpy()
                                                     cnn_dict = cnn.state_dict_numpy()
                                                     
                                                     send(UPDATES)
         recv(UPDATES) ◄────────────────────────────
t=9    
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
t=10      
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

---

---

# Fundamentos Teóricos: Async-SGD y Staleness Correction

## 1. SGD Sincrónico vs Asincrónico

### Definición: SGD Sincrónico (Barrier Synchronous Gradient Descent)

```
ITERACIÓN t:
  1. Todos los Workers entrenan un batch localmente
     θ_i^(t) ← θ^(t)
     g_i^(t) = ∇L_i(X_i ; θ_i^(t))  [Calcula gradiente localmente]
  
  2. ESPERAR que TODOS terminen (BARRIER)
  
  3. PS promedia gradientes
     Δθ = (1/N) * Σ_i g_i^(t)  [Agregación síncrona]
  
  4. PS actualiza parámetros
     θ^(t+1) = θ^(t) - η * Δθ
  
  5. Workers reciben θ^(t+1), vuelven a PASO 1
```

**Propiedades**:
- ✅ **Determinístico**: Orden de pasos predecible
- ✅ **Convergencia garantizada**: Bajo condiciones estándar (Lipschitz smooth, bounded variance)
- ❌ **Bottleneck**: Más lento = velocidad del Worker más lento
- ❌ **Escalabilidad pobre**: Con N workers lentos, espera N*t_slow

**Complejidad de convergencia** (teórica):
```
Para SGD sincrónico con N workers:
Loss(T) ≈ c * log(1/ε) / (T * N)  [O(1/(T*N))]

donde T es el número de iteraciones y N el número de workers.
```

### Definición: Asincrónico SGD (ASGD)

```
ITERACIÓN t (POR WORKER, INDEFINIDAMENTE):
  
  Worker i:
  1. Lee parámetros actuales (en alguna versión v)
     θ_read = copia de θ^(v)
     version_read = v
  
  2. Entrena batch localmente
     g_i = ∇L_i(X_i ; θ_read)  [Gradiente con θ viejo]
  
  3. SIN ESPERAR: Envía gradiente al PS
     (No hay barrier, otros Workers pueden estar en cualquier fase)
  
  4. PS aplica actualización
     θ^(v') = θ^(v') - η * g_i
     [pero g_i fue calculado con θ_read que es antiguo!]
```

**Propiedades**:
- ✅ **Throughput alto**: No hay sincronización, máxima utilización
- ✅ **Escalable**: Agregar workers acelera (no espera al más lento)
- ⚠️ **Inestable**: Gradientes viejos pueden diverger
- ⚠️ **No determinístico**: Orden depende de timings de red

**El Problema Clave: Staleness**
```
Si Worker i lee en versión v_read y aplica en versión v_actual:

Staleness s = v_actual - v_read

Ejemplo:
  PS versión: 0 → 1 → 2 → 3 → 4 → 5 → ...
  
  Worker A lee en v=0
  [mientras Worker A entrena...]
  PS recibe updates de otros workers: v=1, 2, 3, 4, 5
  
  Worker A aplica su gradiente:
  staleness = 5 - 0 = 5 [su información es 5 versiones antigua!]
  
  Gradiente con información tan vieja puede ser:
  - En dirección opuesta al gradiente actual
  - Apuntar hacia mínimos locales pasados (ya superados)
  - Causar divergencia
```

---

## 2. Análisis Teórico: Factor de Corrección α(s)

### Derivación Intuitiva

**Hipótesis**: Un gradiente más viejo debe contribuir menos a la actualización.

Queremos una función α(s) tal que:
```
α(s=0) = 1.0     # Gradiente fresco contribuye completamente
α(∞) → 0         # Gradiente muy viejo casi no contribuye
α es monótona decreciente
```

**Opción 1: Lineal** α(s) = 1 - s
- Problema: α(s) se vuelve negativo para s > 1, no tiene sentido

**Opción 2: Exponencial** α(s) = e^(-λs)
- Teoría válida pero cálculo Σ es más complejo

**Opción 3: Racional (Elegida)** α(s) = 1 / (1 + λs)
```
α(s) = 1 / (1 + λ·s)
```

**Por qué esta forma específica**:

1. **Derivada respecto a s**:
   ```
   dα/ds = -λ / (1 + λs)² < 0  [Always decreasing]
   ```

2. **Comportamiento en límites**:
   ```
   lim_{s→0}   α(s) = 1 / (1 + 0) = 1.0           [Fresco ✓]
   lim_{s→∞}   α(s) = 1 / (1 + ∞) = 0            [Viejo ✓]
   ```

3. **Parámetro de control**:
   ```
   λ pequeño (ej: 0.01)  → α decae lentamente → Tolera staleness
   λ grande (ej: 1.0)    → α decae rápidamente → Rechaza updates viejos
   ```

4. **Justificación matemática formal** (simplificada):
   
   Consideremos el **error de aproximación** si usamos gradiente de versión v_old en versión v_new:
   
   ```
   Sea: θ* = parámetro óptimo (desconocido)
        θ_t = parámetro en tiempo t
        g_t = gradiente en tiempo t
   
   En SGD sincrónico:
   θ_{t+1} = θ_t - η·g_t  [gradiente actual]
   
   En Async-SGD sin corrección:
   θ_{t+1} = θ_t - η·g_{t-s}  [gradiente de s pasos atrás]
   
   Error adicional ≈ s·||∇²L||·Δθ  [Taylor expansion]
   
   Para amortiguar este error, usar:
   θ_{t+1} = θ_t - η·α(s)·g_{t-s}  donde α(s) ↓ con s
   
   La forma α(s) = 1/(1+λs) es una elección "razonable"
   que balanza: descenso rápido pero suave.
   ```

### Valores de α(s) Recomendados por Staleness

```
Tabla de decisión para λ:
═════════════════════════════════════════════════════════════

Caso                          λ Recomendado    Intuición
─────────────────────────────────────────────────────────────

1 Worker (sin competencia)    λ ≈ 0            (α siempre = 1)
                                               → No hay staleness

2-4 Workers (LAN)             λ ≈ 0.1          → Tolerancia media
  staleness típico: 1-2             α(1)=0.91
                                     α(2)=0.83

10+ Workers (WAN)             λ ≈ 0.5-1.0      → Mayor tolerancia
  staleness típico: 5-20        α(5)=0.29-0.67
                                 α(20)=0.09-0.33

Conexión muy lenta            λ ≈ 1.5-2.0      → Máxima tolerancia
  staleness típico: 50+        α(50)=0.012-0.032
```

---

## 3. Convergencia de Async-SGD: Análisis Teórico

### Modelo de Convergencia (Bosquejo)

Para **función de loss convexa** L(θ) y Async-SGD con correccion α(s):

```
TEOREMA (informal):

Si:
  (a) L(θ) es Lipschitz smooth: ||∇L(θ1) - ∇L(θ2)|| ≤ G·||θ1 - θ2||
  (b) Varianza de gradientes acotada: E[||g - ∇L(θ)||²] ≤ σ²
  (c) λ > 0 y learning rate η suficientemente pequeño
  (d) Staleness s ≤ s_max (acotado, típicamente O(N) con N workers)

ENTONCES:

  E[L(θ_T)] - L(θ*) ≤ O(η·σ²) + O(s_max)·O(η²·G²)

  En palabras:
  - Primer término: error de varianza estándar (como SGD)
  - Segundo término: penalty por staleness (decrece con η)
  
  Si η ≈ 1/√T (learning rate estándar):
  - Convergencia es O(1/√T + s_max/T)
  - Con s_max acotado, converge a O(1/√T) [casi como SGD]
```

### Implicaciones Prácticas

```
1. ESTABILIDAD GARANTIZADA (si condiciones se cumplen):
   - Con λ > 0, el factor α(s) amortigua updates viejos
   - Sistema NO diverge (matemáticamente sound)
   - Loss es serie monótona decreciente (en expectativa)

2. CONVERGENCIA MÁS LENTA QUE SGD SINCRÓNICO:
   - Async-SGD: O(1/√T + s_max/T)
   - SGD Sincrónico: O(1/√T)
   - Diferencia: factor O(s_max/T) por staleness
   
   Con T = 100k iteraciones, s_max = 100:
   - Async: 1/√100k + 100/100k ≈ 0.01 + 0.001 = 0.011
   - Sync: 1/√100k ≈ 0.01
   - Degradación: ~10% de overhead por staleness

3. THROUGHPUT vs CONVERGENCIA:
   - Async-SGD: Bajo staleness (rápido), alto throughput
   - Sync-SGD: Sin staleness, bajo throughput
   
   Decisión: ¿Qué prefieres?
   - 100 iters rápidas (Async, cada una con ruido) → Resuelve en 1 hora
   - 100 iters lentas (Sync, limpias) → Resuelve en 2 horas
```

---

## 4. Inicialización de Parámetros: Why Kaiming Uniform

### El Problema: Vanishing/Exploding Gradients

**Sin inicialización cuidadosa**:

```
Setup incorrecto: CNN congelada, MLP con pesos aleatorios (0 mean, 1 std)

Forward pass:
  x_0 = features from CNN       # shape (batch, 512), std ≈ 1
  z_1 = W_1 @ x_0 + b_1        # shape (batch, 1024)
  
  Si W_1 ~ N(0, 1):
  E[||z_1||] = E[||W_1 @ x_0||] = √(512) ≈ 22.6 [EXPLOTA!]
  
  x_1 = ReLU(z_1)              # "saturación": la mayoría son 0 o grandes
  z_2 = W_2 @ x_1 + b_2
  ...
  logits ~ N(0, σ²_huge)       # Predicciones con varianza gigantesca

Backward pass:
  dL/dW_1 ~ N(0, σ²_enormous)
  Gradientes MUY grandes → Actualización SGD explota
```

**Solución: Kaiming Uniform Initialization**

```
Idea: Mantener VARIANZA CONSISTENTE a través de las capas
      forward y backward pass.

Para capa i con fan_in entradas y ReLU:

W_i ~ U[-bound, bound]  donde bound = √(6 / fan_in)

En nuestro código MLPPyTorch._init_weights():
  for layer in [fc1, fc2, fc3]:
    nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
    nn.init.zeros_(layer.bias)
```

### Por qué Funciona (Matemáticamente)

**Propagación de Varianza Forward**:

```
Layer 1:
  x_0 = features ~ N(0, σ₀²)   where σ₀² = 1  (CNN normaliza internamente)
  z_1 = Σ_j W_1[i,j] * x_0[j]  + b_1[i]
  
  Var[z_1[i]] = Var[Σ_j W_1[i,j] * x_0[j]]
               = Σ_j Var[W_1[i,j]] * Var[x_0[j]]
               = fan_in * Var[W] * Var[x_0]
               
  Con Var[W] = 2/(6/fan_in) = 1/fan_in:
  
  Var[z_1[i]] = fan_in * (1/fan_in) * 1 = 1 ✓ [Constante!]
```

**Propagación de Varianza Backward**:

```
Similar analysis para gradientes:
  Var[dL/dW_i] ∝ fan_out

  Con Kaiming init ajustando por fan_out también:
  Var[dL/dW_i] ≈ constante en todas las capas ✓
```

**Resultado**:
```
✓ Loss inicial ≈ log(num_classes) ≈ log(1000) ≈ 6.9 (no 0, no ∞)
✓ Gradientes iniciales en escala razonable (~0.01-0.1)
✓ Entrenamiento estable desde el primer batch
✓ No necesita learning rate "ajustado finamente"
```

---

## 5. Análisis de Hiperparámetros: Impacto en Convergencia

### Learning Rate (η)

```
Hipótesis: θ_{t+1} = θ_t - η · ∇L(θ_t)

- η muy pequeño (0.0001):
  + Actualización estable, evita divergencia
  - Convergencia lenta: tan pequeño que cambios son imperceptibles
  - Tiempo a convergencia: ~1000x más lento
  
- η razonable (0.001):
  + Balance: avanza rápido pero sin explotar
  - Predeterminado en nuestro proyecto
  
- η grande (0.1):
  + Converge rápido si empiezas cerca del óptimo
  - Riesgo: zigzag alrededor del óptimo, puede diverger
  - ReRisk: especialmente con Async-SGD + staleness

Fórmula del impacto (para función cuadtrática):
  Convergence rate ∝ 1 - 2ηG  [G = Lipschitz const]
  
  Si η > 1/(2G): Divergencia
  Si η ≈ 0: Convergencia lenta
  Si η ≈ 1/(4G): Convergencia óptima (teoría)
```

### Staleness Lambda (λ)

```
Factor de corrección: α(s) = 1 / (1 + λ·s)

- λ = 0 (sin corrección):
  + Máximo throughput (actualiza completamente)
  - Alto riesgo de divergencia con muchos workers
  - Viejo paper: "Hogwild!" SGD, necesitaba CPU NUMA tricks
  
- λ pequeño (0.01):
  + Tolera hasta ~100 workers en buena red
  - Aún requiere monitoreo de loss
  
- λ = 0.1 (predeterminado):
  + Buen balance: tolera ~10-50 workers
  + Garantiza estabilidad
  
- λ grande (1.0):
  + Ultra-estable, tolera WAN con latencia alta
  - Convergencia más lenta (valida menos updates viejos)
```

### Batch Size (B)

```
Ecuación de actualización: Δθ = (1/B) · Σ ∇L(x_i)

- B pequeño (16):
  + Menos memoria
  - High variance en gradientes: ruido estadístico
  - Riesgo: zig-zag en convergencia, tarda más
  
- B = 64 (predeterminado):
  + Balance: varianza moderada, memoria razonable
  + Suficiente para cancel noise pero mantener dinamismo
  
- B grande (256):
  + Low variance: gradientes "limpios", menos ruido
  - Convergencia más determinística pero rígida
  - Risk: cae en mínimos locales malos
  - Red overhead: enviar 256 images vs 64
```

### Métricas Window Size (para GUI)

```
Parámetro: window_size = 50  (antes era 200)

Significado: Promediar últimos 50 steps para graficar

- window_size = 10:
  + Respuesta rápida a cambios en loss
  - Gráfica muy "ruidosa", difícil de leer
  - Cambios de LR ó λ se ven inmediatamente (bueno para tuning)
  
- window_size = 50 (actual):
  + Balance: suavidad visual + responsividad
  - Loss converge = cambios se ven en ~5 segundos
  
- window_size = 200 (antes):
  + Gráfica muy suave ("limpia")
  - Feedback lento: cambios de LR se ven en ~20 segundos
  - Malo para debugging dinámico
```

### Tabla: Impacto Relativo en Convergencia

```
╔═══════════════════════╦══════════════╦════════════════════════════════════╗
║ Hiperparámetro        ║ Rango Recom. ║ Impacto en Convergencia            ║
╠═══════════════════════╬══════════════╬════════════════════════════════════╣
║ Learning Rate η       ║ 0.001-0.01   ║ CRÍTICO (alta sensibilidad)        ║
║                       ║              ║ Diverge si > 0.1, muy lento si < 10^-5║
║ Staleness Lambda λ    ║ 0.05-1.0     ║ IMPORTANTE (estabilidad con N>4)   ║
║                       ║              ║ λ=0 falla con múltiples workers    ║
║ Batch Size B          ║ 32-256       ║ MODERADO (varianza de gradientes)  ║
║                       ║              ║ B=64: standard, B>256: overhead red║
║ MLP Hidden Units      ║ 512-2048     ║ BAJO (arquitectura, no convergencia)║
║                       ║              ║ Afecta capacidad, no velocidad     ║
║ Prefetch Buffer       ║ 2-8          ║ BAJO (throughput, no loss)         ║
║                       ║              ║ Más data paralela = menos espera   ║
╚═══════════════════════╩══════════════╩════════════════════════════════════╝
```

---

## Resumen: Por Qué el Algoritmo Funciona

```
1. ASYNC-SGD (Sin Staleness Correction):
   ❌ Rápido pero inestable
   ❌ Gradientes viejos apuntan en dirección equivocada
   ❌ Divergencia frecuente con N > 2 workers

2. ASYNC-SGD + STALENESS CORRECTION α(s) = 1/(1+λ·s):
   ✅ Rápido (sin barriers)
   ✅ Estable (updates viejos atenuados)
   ✅ Convergencia teórica O(1/√T + s_max/T)
   ✅ Escalable a N workers (si λ bien tuneado)

3. KAIMING INITIALIZATION:
   ✅ Varianza consistente en todas las capas
   ✅ No hay vanishing/exploding gradients
   ✅ Training estable desde batch 0
   ✅ Loss inicial sensible ≈ log(1000)

4. TUNING DE HIPERPARÁMETROS:
   - η: Controla velocidad y estabilidad (crítico)
   - λ: Controla tolerancia a staleness (importante con N > 4)
   - B: Trade-off ruido vs throughput (moderado)

RESULTADO:
═══════════════════════════════════════════════════════════════
El sistema converge sin divergencia, tolerando múltiples workers
simultáneos con latencias de red, logrando throughput Y estabilidad.
═══════════════════════════════════════════════════════════════
```

