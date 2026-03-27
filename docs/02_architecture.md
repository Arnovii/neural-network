# 2. ARQUITECTURA DEL SISTEMA

## Componentes principales

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                    ┌──────────────────────────────┐                         │
│                    │   PARAMETER SERVER (PS)      │                         │
│                    ├──────────────────────────────┤                         │
│                    │ • CNNExtractor (PyTorch)     │                         │
│                    │ • MLP params (NumPy)         │                         │
│                    │ • Sincronización             │                         │
│                    │ • Promedio de gradientes     │                         │
│                    │ • Caché de features (E2E)    │                         │
│                    └──────────────────────────────┘                         │
│                              ▲                                              │
│                              │                                              │
│         ┌────────────────────┼────────────────────┐                         │
│         │                    │                    │                         │
│         ▼                    ▼                    ▼                         │
│    ┌─────────┐          ┌─────────┐         ┌─────────┐                     │
│    │WORKER 0 │          │WORKER 1 │         │WORKER N │                     │
│    ├─────────┤          ├─────────┤         ├─────────┤                     │
│    │ Datos   │          │ Datos   │         │ Datos   │                     │
│    │(50K)    │          │(50K)    │         │(50K)    │                     │
│    │         │          │         │         │         │                     │
│    │ CNN ──► Features    │ CNN ──► Features  │ CNN ──► Features             │
│    │ MLP ──► ∇L          │ MLP ──► ∇L        │ MLP ──► ∇L                   │
│    └─────────┘          └─────────┘         └─────────┘                     │
│         TCP                  TCP                 TCP                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Responsabilidades de cada componente

### **Parameter Server (PS)**

**Rol**: Orquestador central de la sesión de entrenamiento

**Responsabilidades**:
- ✅ Aceptar conexiones de Workers (asignación de IDs)
- ✅ Preentrenar/validar la CNN localmente
- ✅ Distribuir pesos CNN a todos los Workers via `CNN_WEIGHTS`
- ✅ Esperar a que todos confirmen `CNN_READY` (barrera)
- ✅ Generar `epoch_seed` aleatorio cada época
- ✅ Enviar PARAMS (MLP + seed + opcionalmente CNN) a cada Worker
- ✅ Recibir GRADIENTS de todos los Workers
- ✅ Promediar gradientes: ∇̄ = (1/N) * Σ∇
- ✅ Actualizar pesos: W ← W − lr * ∇̄
- ✅ Evaluar en datos de prueba (si se proporcionan)
- ✅ Repetir por cada época

**Qué NO hace**:
- ❌ Ver datos de entrenamiento locales de Workers
- ❌ Calcular gradientes de cifrar-10 (Workers lo hacen)
- ❌ Almacenar features en disco (Workers lo hacen)

**Archivos**:
- `Distributed/parameter_server.py` — Implementación
- `ps_gui.py` — Interfaz gráfica (Tkinter)
- `ps_terminal.py` — Interfaz por terminal

---

### **Worker Node**

**Rol**: Cálculo local en paralelo

**Responsabilidades**:
- ✅ Conectar al PS enviando `READY`
- ✅ Recibir `WORKER_ID` asignado
- ✅ Cargar CIFAR-10 (50K imágenes) en RAM
- ✅ Recibir y cargar pesos CNN via `CNN_WEIGHTS`
- ✅ Extraer features (precomputed) O habilitar CNN (E2E)
- ✅ Confirmar `CNN_READY` cuando listo
- ✅ Esperar sesión de entrenamiento (`TRAIN_START`)
- ✅ Por cada época:
  - Recibir `PARAMS` + seed
  - Reconstruir índices localmente (stratified sampling)
  - Calcular forward CNN (si E2E) + MLP
  - Calcular backward MLP (+ CNN si E2E)
  - Enviar `GRADIENTS` al PS
- ✅ Persistente: no se desconecta entre épocas/sesiones

**Qué NO hace**:
- ❌ Ver datos de otros Workers
- ❌ Comunicarse con otros Workers (solo con PS)
- ❌ Decidir qué modelo entrenar (PS decide)

**Archivos**:
- `Distributed/worker_node.py` — Implementación
- `worker.py` — Entry point (script a ejecutar)

---

### **CNN Extractor (PyTorch)**

**Rol**: Extracción de characteristics convolucionales

**Arquitecturas soportadas**:
1. **"simple"** (custom):
   - 3 bloques Conv → BN → ReLU → MaxPool
   - Optimizada para CIFAR-10 (32×32)
   - feature_dim = 512
   - Requiere preentrenamiento local (1-2 min)

2. **"resnet18"** (torchvision):
   - ResNet-18 estándar
   - Con `pretrained=True`: pesos ImageNet
   - Upscale 32×32 → 224×224
   - feature_dim = 512
   - Listo para usar (sin preentrenamiento local)

**Responsabilidades**:
- ✅ Forward pass: (N, 3, 32, 32) → (N, 512)
- ✅ Serializar pesos (torch.save → bytes)
- ✅ Cargar pesos (bytes → model)
- ✅ Congelar/habilitar gradientes (set_trainable)
- ✅ Calcular hash MD5 de pesos (para caché)
- ✅ Caché de features con validación

**Hecho en PyTorch porque**:
- Operaciones convolucionales eficientes
- Pesos preentrenados (ResNet) disponibles
- CUDA/GPU aceleración automática

**Archivos**:
- `Model/cnn_extractor.py` — Implementación

---

### **MLP Classifier (NumPy)**

**Rol**: Clasificador lineal distribuido

**Arquitectura**:
```
features (N, feature_dim=512)
    ↓
[Linear] W1 (512 × 256) + b1
    ↓
ReLU (hidden1 = 256)
    ↓
[Linear] W2 (256 × 128) + b2
    ↓
ReLU (hidden2 = 128)
    ↓
[Linear] W3 (128 × 10) + b3
    ↓
Softmax → logits → cross-entropy loss
```

**Responsabilidades**:
- ✅ Forward pass: (N, feature_dim) → (N, 10) logits
- ✅ Backward pass: gradientes de W1, b1, W2, b2, W3, b3
- ✅ Serializar gradientes con Pickle
- ✅ Aplicar actualización SGD: W ← W − lr * ∇

**Hecho en NumPy porque**:
- Serialización Pickle es eficiente (arrays binarios)
- Gradientes pequeños (feature_dim × hidden1 ≈ 128 KB)
- Control total sobre forward/backward (pedagogía)

**Archivos**:
- `Model/mlp.py` — Implementación

---

### **Protocolo de comunicación**

**Rol**: Mensajería confiable entre PS y Workers

**Formato de cada mensaje**:
```
┌──────────────┬────────────────────────────┐
│  4 bytes     │  N bytes                   │
│ Longitud N   │ pickle.dumps(mensaje)      │
│ (big-endian) │                            │
└──────────────┴────────────────────────────┘
```

**Tipos de mensajes**:

| Tipo | Dirección | Payload | Propósito |
|------|-----------|---------|-----------|
| `READY` | Worker → PS | {} | Worker solicita conexión |
| `WORKER_ID` | PS → Worker | {worker_id: int} | PS asigna ID |
| `CNN_WEIGHTS` | PS → Worker | {arch, weights_bytes} | PS distribuye CNN |
| `CNN_READY` | Worker → PS | {worker_id} | Worker confirmó CNN |
| `TRAIN_START` | PS → Worker | {epochs, n_train, n_workers, training_mode} | Inicia sesión |
| `PARAMS` | PS → Worker | {epoch, params: MLPWeights, seed, cnn_params?} | Parámetros de época |
| `GRADIENTS` | Worker → PS | {gradients: MLPGrads, cnn_gradients?} | Gradientes calculados |
| `STOP` | PS → Worker | {} | Finalizar Worker |
| `REQUEST_TEST_FEATURES` | PS → Worker | {} | PS pide features test |
| `TEST_FEATURES` | Worker → PS | {X_features, Y_test} | Features de test |

**Flujo de un Worker típico**:
```
READY ──────►  ◄──── WORKER_ID
                        ↓
              ◄──── CNN_WEIGHTS
                        ↓
              CNN_READY ────►
                        ↓
         Esperando TRAIN_START...
                        ↓
              ◄──── TRAIN_START
                        ↓
    ┌──► PARAMS ──►  [calcula gradientes]
    │              ◄──── GRADIENTS
    └────┴────── (repeat N veces por N épocas)
                        ↓
              ◄──── STOP
```

**Archivos**:
- `Distributed/protocol.py` — Implementación

---

### **Utilities**

**`cifar_loader.py`**:
- Descarga CIFAR-10 (torchvision)
- Normalización por canal (mean/std estándar)
- Convierte a float32 NCHW
- Caché en .npz para evitar descargas repetidas

**`logging_util.py`**:
- Logger unificado con colores
- Fases: LOAD, PREP, TRAIN, EVAL, INFO
- Evita mensajes inconsistentes

**`results_exporter.py`**:
- Exporta métricas a JSON (`Exports/resultado_*.json`)
- Historial de accuracy/loss por época
- Metadatos de sesión (arch, workers, epochs, etc.)

---

## 🔌 Qué viaja por TCP

| Concepto | Tamaño | Dirección |
|----------|--------|-----------|
| **WORKER_ID** | < 1 KB | PS → Worker (una sola vez) |
| **CNN_WEIGHTS** | 5-50 MB | PS → Worker (inicio sesión) |
| **PARAMS (MLP)** | ~60 KB | PS → Worker (cada época) |
| **PARAMS (CNN)** | ~50 MB | PS → Worker (cada época en E2E) |
| **GRADIENTS (MLP)** | ~60 KB | Worker → PS (cada época) |
| **GRADIENTS (CNN)** | ~50 MB | Worker → PS (cada época en E2E) |
| **TEST_FEATURES** | ~40 MB | Worker (0) → PS (una sola vez) |

**Observación**: En modo E2E, cada época envía ~100 MB de cada Worker — costoso si workers están en máquinas diferentes.

---

## 🔀 Qué NO viaja por TCP

| Concepto | Razón | Ubicación |
|----------|-------|-----------|
| **Imágenes CIFAR-10** | Cada Worker las carga localmente | RAM del Worker |
| **Features (precomputed)** | Se cachean localmente | Disco del Worker |
| **Features (E2E)** | Se calculan on-the-fly | RAM del Worker |
| **Raw MLP features** | Solo se usan localmente | RAM del Worker |

---

## 🏛️ Arquitectura de sincronización

```
┌─────────────────────────────────────────────────────────┐
│                    PS listening()                        │
│          (hilo de aceptación en background)              │
└────────────────────┬────────────────────────────────────┘
                     │
        [Workers se conectan w/ READY]
                     │
        ┌────────────┴───────────┬─────────────┐
        │                        │             │
        ▼                        ▼             ▼
    ┌─────────┐             ┌─────────┐  ┌─────────┐
    │Worker 0 │             │Worker 1 │  │Worker N │
    └─────────┘             └─────────┘  └─────────┘
        │                        │             │
        ├────────CNN_WEIGHTS─────┼─────────────┤
        │                        │             │
        ├─────────CNN_READY──────┼─────────────┤
        │ (barrera: espera a todos)
        │
        ├─────────TRAIN_START────┼─────────────┤
        │
        [EPOCH LOOP]
        │ ├─────────PARAMS───────┼─────────────┤
        │ │ (Workers calculan)
        │ ├────────GRADIENTS─────┼─────────────┤
        │ │ (PS promedia)
        │ └─ (repeat N épocas)
        │
        ├──────────STOP──────────┼─────────────┤
        ▼                        ▼             ▼
    [desconexión]          [desconexión]  [desconexión]
```

---

## Invariantes arquitectónicos

**I1**: Todos los Workers tienen **exactamente los mismos pesos CNN y MLP** en el inicio de cada época.

**I2**: Cada Worker procesa un **chunk disjunto de datos** (definido por epoch_seed).

**I3**: Los gradientes que Worker[i] envía son gradientes de la **pérdida en su chunk**, no del dataset completo.

**I4**: El PS **promedia** los gradientes recibidos, no los suma.

**I5**: En modo PRECOMPUTED, la CNN es **inmutable** — sus pesos nunca cambian.

**I6**: En modo END-TO-END, la CNN es **entrenable** — se actualiza como el MLP.

---

## Componentes secundarios

### **Data Loader (cifar_loader.py)**
- Interfaz única para cargar CIFAR-10
- Garantiza formato consistente (NCHW float32)

### **Feature Cache (cnn_extractor.py)**
- Almacena features con hash de pesos
- Evita recalcular si los pesos no cambian

### **Results Exporter (results_exporter.py)**
- Exporta métricas a JSON para análisis posterior
- Permite reproducir experimentos

---

**Documento**: `docs/02_architecture.md`  
**Última actualización**: 2026-03-27  
**Nivel**: Intermedio
