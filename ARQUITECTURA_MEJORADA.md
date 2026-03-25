# Arquitectura Mejorada - Flujo de Ejecución

## 🏗️ Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                   GUI (ps_gui.py)                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ [SELECTOR MODO]                                            │ │
│  │  🔒 Precomputación (CNN fija + MLP distribuido)            │ │
│  │  ⚙️  End-to-End (Para futuro)                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ [LOG CON FASES]                                            │ │
│  │  [LOAD] → [PREP] → [TRAIN] → [EVAL]                      │ │
│  │  + Progreso + Métricas                                     │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│         Parameter Server (parameter_server.py)                   │
│                                                                  │
│  ┌──────────────────────────────────────┐                       │
│  │ 1. PREP: Distribuir CNN a Workers    │                       │
│  │    Logger: [PREP FEAT]               │                       │
│  └──────────────────────────────────────┘                       │
│                   ↓                                               │
│  ┌──────────────────────────────────────┐                       │
│  │ 2. PREP: Extraer Features de Test    │                       │
│  │    Logger: [PREP FEAT]               │                       │
│  └──────────────────────────────────────┘                       │
│                   ↓                                               │
│  ┌──────────────────────────────────────┐                       │
│  │ 3. TRAIN: Loop por Épocas            │                       │
│  │    Logger: [TRAIN MLP]               │                       │
│  │    - Broadcast params a Workers      │                       │
│  │    - Wait para gradients             │                       │
│  │    - Promediar + Actualizar         │                       │
│  │    - Evaluar en test (opcional)      │                       │
│  └──────────────────────────────────────┘                       │
│                   ↓                                               │
│  ┌──────────────────────────────────────┐                       │
│  │ 4. EVAL: Resultados finales          │                       │
│  │    Logger: [EVAL]                    │                       │
│  │    - Accuracy test final             │                       │
│  │    - Pérdida final                   │                       │
│  └──────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                   ↕  (TCP sockets)
┌─────────────────────────────────────────────────────────────────┐
│                  Worker Nodes (worker.py x N)                    │
│                                                                  │
│  ┌──────────────────────────────────────┐                       │
│  │ PREP: Cargar imágenes CIFAR-10       │                       │
│  │ Logger: [LOAD DATA]                  │                       │
│  └──────────────────────────────────────┘                       │
│                   ↓                                               │
│  ┌──────────────────────────────────────┐                       │
│  │ PREP: Recibir CNN pesos del PS       │                       │
│  │ Logger: [PREP FEAT]                  │                       │
│  └──────────────────────────────────────┘                       │
│                   ↓                                               │
│  ┌──────────────────────────────────────┐                       │
│  │ PREP: Extraer features localmente    │                       │
│  │ Logger: [PREP FEAT] (progreso)       │                       │
│  │ (CNN forward pass en GPU/CPU)        │                       │
│  └──────────────────────────────────────┘                       │
│                   ↓                                               │
│  ┌──────────────────────────────────────┐                       │
│  │ TRAIN: Loop por épocas               │                       │
│  │ Logger: [TRAIN MLP]                  │                       │
│  │ - Recibir params globales            │                       │
│  │ - Reconstruir datos locales          │                       │
│  │ - Forward + Backward del MLP         │                       │
│  │ - Enviar gradientes al PS            │                       │
│  └──────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Flujo de Ejecución Detallado

### **FASE 1: CARGA DE DATOS**

```
User selecciona "Precalculated" mode → GUI habilita selector CNN

[LOAD DATA] Cargando CIFAR-10 entrenamiento...     (50000 imágenes)
[LOAD DATA] Cargando CIFAR-10 prueba...           (10000 imágenes)
```

**En Worker**:
- `load_cifar10_train()` → RAM con imágenes CIFAR-10
- `load_cifar10_test()` → RAM con imágenes de test

**Duración**: ~2-5 segundos

---

### **FASE 2: PREPROCESAMIENTO & EXTRACCIÓN DE CARACTERÍSTICAS**

#### **2.1 Distribución de CNN**

```
[PARAM SRV] Distribuyendo CNN a Workers    | arch=resnet18
  └─ Broadcast: CNN_WEIGHTS → todos os Workers
```

**Parameter Server**:
1. Serializa pesos CNN: `_get_weights_bytes()`
2. Envía en paralelo a todos los Workers
3. Espera confirmación CNN_READY de cada uno

**Cada Worker**:
1. Recibe CNN_WEIGHTS
2. Deserializa y carga en su GPU/CPU
3. Responde CNN_READY

**Duración**: ~5-20 segundos (depende de tamaño CNN)

#### **2.2 Extracción de Features**

```
[PREP FEAT] Extrayendo features de entrenamiento           (45%)
[PREP FEAT] Extrayendo features de prueba...              (100%)
```

**En Parameter Server**:
1. Pide test features a un Worker (failover available)
2. Worker ejecuta: `features = CNN(test_images)`
3. Envía features X de test (10000 imágenes → 512-dim cada una)

**En cada Worker**:
1. Ejecuta: `X_features = CNN(X_train)` internamente
2. Cacheado automáticamente en `Data/feature_cache/`
3. No se transmite (50 000 imágenes × 512 floats = ~100 MB de datos)

**Duración**: ~30-120 segundos (depende de GPU/CPU)

---

### **FASE 3: ENTRENAMIENTO DISTRIBUIDO**

```
═══════════════════════════════════════════════════════════════
[TRAIN MLP] Iniciando entrenamiento distribuido              
[TRAIN MLP] Épocas=100 | Workers=2 | Learning_rate=0.01     
═══════════════════════════════════════════════════════════════

[TRAIN MLP] Época 1/100    | train_acc=42.3% | pérdida=2.301
[TRAIN MLP] Época 2/100    | train_acc=63.7% | pérdida=1.045
[TRAIN MLP] Época 3/100    | train_acc=72.1% | pérdida=0.645
...
[TRAIN MLP] Época 100/100  | train_acc=92.3% | pérdida=0.198
```

**Por cada época**:

1. **Broadcast de Parámetros** (Parameter Server → Workers)
   ```
   Send: PARAMS { epoch, global_weights, epoch_seed }
   ```
   - Todos reciben MISMOS parámetros
   - Misma seed → reconstruyen MISMOS datos localmente

2. **Procesamiento Local** (En cada Worker en paralelo)
   ```
   For worker in workers:
       X_train_chunk = reconstruct(X_features, epoch_seed, worker_rank)
       Y_train_chunk = reconstruct(Y, epoch_seed, worker_rank)
       
       # Forward + Backward del MLP
       predictions = MLP(X_train_chunk, weights)
       loss = CrossEntropy(predictions, Y_train_chunk)
       gradients = backward(loss)
   ```

3. **Recepción de Gradientes** (Parameter Server)
   ```
   Wait for all workers:
       gradients_W1 = recv_from_worker_0
       gradients_W2 = recv_from_worker_1
   ```

4. **Promediado y Actualización** (Parameter Server)
   ```
   avg_grads = (gradients_W1 + gradients_W2) / 2
   weights ← weights - lr * avg_grads
   ```

5. **Evaluación Opcional** (Si se pasaron datos de test)
   ```
   test_acc, test_loss = MLP_evaluate(test_features, weights)
   ```

**Esta secuencia se repite** 100 veces (una por época)

**Duración**: ~0.5-2 segundos por época (depende de hardware)

---

### **FASE 4: EVALUACIÓN FINAL**

```
[EVAL] Evaluando en dataset de prueba...                (10000 imgs)
[EVAL] Precisión final: 90.2%  | Pérdida: 0.287

[PS] Entrenamiento completado
```

**Acciones**:
1. Forward pass del MLP sobre **todos** los datos de test
2. Calcula accuracy y pérdida
3. Exporta resultados a `Exports/resultado_TIMESTAMP.json`

**Duración**: ~2-5 segundos

---

## 🔄 Manejo de Errores y Recuperación

### Desconexión de Worker Durante Entrenamiento

```
[ERROR] Error recibiendo de Worker 0: conexión perdida
[WARN]  Worker 0 eliminado de la sesión
```

**Acción**: Continúa con workers restantes (N-1)  
**Impacto**: Datos de Worker 0 se pierden, descenso en accuracy

### Fallos en Extracción de Features de Test

```
[PREP FEAT] Solicitando features de prueba | Worker 0
[WARN]     Worker 0 falló. Intentando siguiente...
[PREP FEAT] Solicitando features de prueba | Worker 1
[PREP FEAT] Features extraídos                | shape=(10000,512)
```

**Acción**: Failover automático a siguiente Worker  
**Si todos fallan**: PS extrae features localmente (cpuintensivo, fallback)

---

## 📊 Uso de Memoria y Red

### Transferencia de Datos

| Concepto | Tamaño | Frecuencia |
|----------|--------|-----------|
| CNN weights (ResNet18) | ~45 MB | 1 vez al inicio |
| Test features (10K × 512) | ~20 MB | 1 vez al inicio |
| MLP weights | ~100 KB | Por época (100 veces) |
| Gradients MLP | ~100 KB | Por época (100 veces) |
| **TOTAL por sesión** | ~46 MB + (100K × 100) | - |

**Imágenes NUNCA se transmiten** (privacy + eficiencia)

### Memoria en Workers

- X_train raw: 50000 × 3 × 32 × 32 × 4 bytes = **600 MB**
- X_train features: 50000 × 512 × 4 bytes = **100 MB**  
- Cached after first run ✅

---

## 🎯 Mejoras Implementadas - Mapeo

### Cambio solicitado → Cómo se implementó

| Solicitación | Implementación | Archivo |
|-------------|----------------|---------|
| **Control explícito del modo** | Selector radio button en GUI | ps_gui.py |
| **Feedback en procesos largos** | Logs con progreso (PREP FEAT) | parameter_server.py |
| **Consistencia en logs** | logging_util.py con fases | todos |
| **Adaptación dinámica** | _on_system_mode_change() | ps_gui.py |
| **Claridad del flujo** | Fases [LOAD]→[PREP]→[TRAIN]→[EVAL] | logging_util.py |

---

## 🔮 Ready para End-to-End

La estructura actual es **agnóstica** respecto a CNN congelada vs entrenada:

```python
# Precomputación (ACTUAL)
cnn_freeze = True
gradients_sent = [mlp_grads]  # Solo MLP

# End-to-End (FUTURO)
cnn_freeze = False
gradients_sent = [cnn_grads, mlp_grads]  # Ambas redes
```

**Sin cambios en arquitectura de comunicación**.

---

**Version**: 1.0  
**Date**: 2026-03-24
