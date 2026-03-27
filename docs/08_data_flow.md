# 8. DATA FLOW COMPLETO

## 📍 Diagrama general

```
╔════════════════════════════════════════════════════════════════════╗
║                    DATOS CIFAR-10 (50,000 imágenes)              ║
║                   √ uint8, shape (32×32×3)                        ║
╚════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
           ┌──────────────────────────────────────┐
           │      NORMALIZACIÓN (cifar_loader)     │
           ├──────────────────────────────────────┤
           │ • float32 / 255                       │
           │ • (X - mean) / std per channel        │
           │ • Transpose NHWC → NCHW              │
           │ Shape: (50000, 3, 32, 32)            │
           │ Size: 6.4 GB → 200 MB (float32)      │
           └──────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
    ┌───────────────────────┐   ┌──────────────────────┐
    │   WORKER Node         │   │   PARAMETER SERVER   │
    │                       │   │                      │
    │ Almacena en RAM:      │   │ Mantiene:            │
    │ _X_raw (50K) 200 MB   │   │ • Modelo CNN         │
    │                       │   │ • Parámetros MLP     │
    └───────────────────────┘   └──────────────────────┘
                │
    ┌───────────┴──────────┬───────────────────┐
    │                      │                   │
    ▼                      ▼                   ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────┐
│ PRECOMPUTED  │  │    END-TO-END    │  │ TEST DATA    │
│   MODE       │  │      MODE        │  │              │
└──────────────┘  └──────────────────┘  └──────────────┘
```

---

## 🔵 FLUJO PRECOMPUTED

### **Fase 1: Extracción de features (ONCE)""

```
WORKER NODE:
┌──────────────────────────────────────────────────────┐
│                                                      │
│  _X_raw (50000, 3, 32, 32)  [200 MB en RAM]         │
│          │                                          │
│          ├─ CACHE CHECK?                            │
│          │   ├─ Hash CNN weights → "abc123de"      │
│          │   ├─ Look for:                           │
│          │   │   Data/feature_cache/                │
│          │   │   simple_abc123de_train_X.npy        │
│          │   │                                      │
│          │   ├─ [CACHE HIT] → _X_features          │
│          │   │  (~0.5s via np.load)                │
│          │   │                                      │
│          │   └─ [CACHE MISS]                        │
│          │      │                                   │
│          ▼      ▼ CNN Forward (PyTorch)             │
│       ┌─────────────────────────────────┐           │
│       │ For each batch (2048 imgages):  │           │
│       │ │                               │           │
│       │ ├─ To GPU/device                │           │
│       │ ├─ Conv → BN → ReLU → MaxPool   │           │
│       │ ├─ Conv → BN → ReLU → MaxPool   │           │
│       │ ├─ Conv → BN → ReLU → MaxPool   │           │
│       │ ├─ AdaptiveAvgPool(1)           │           │
│       │ ├─ Result: (2048, 512)          │           │
│       │ └─ Back to CPU, append          │           │
│       │                                 │           │
│       │ Total time: ~30-60s             │           │
│       └─────────────────────────────────┘           │
│             │                                      │
│             ▼                                      │
│       _X_features (50000, 512)  [200 MB in RAM]   │
│             │                                      │
│             ├─ Save to cache:                     │
│             │   Data/feature_cache/                │
│             │   simple_abc123de_train_X.npy        │
│             │   (~200 MB, ~1-2s write)             │
│             └─ _class_indices pre-computed        │
│
│  ✓ CNN_READY sent to PS
│
└──────────────────────────────────────────────────────┘

SETUP TIME: ~0.5-60s (depend cache hit/miss)
```

### **Fase 2: Training session (EACH EPOCH)**

```
┌─────────────────────────────────────────────────────────┐
│ PARAMETER SERVER                                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [epoch = 0]                                            │
│                                                         │
│  seed_0 = 42  (random)                                 │
│       │                                                 │
│       ├─ Generate PARAMS:                              │
│       │  ├─ epoch: 0                                   │
│       │  ├─ params: {W1, b1, W2, b2, W3, b3}          │
│       │  ├─ seed: 42                                   │
│       │  └─ cnn_params: None  ← CRITICAL              │
│       │                                                 │
│       └─► [BROADCAST to all Workers]                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌──────────────┐            ┌──────────────┐
│ WORKER 0     │            │ WORKER 1     │
├──────────────┤            ├──────────────┤
│ rank = 0     │            │ rank = 1     │
│ n_workers = 2│            │ n_workers = 2│
│              │            │              │
│ RECONSTRUCT  │            │ RECONSTRUCT  │
│ indices:     │            │ indices:     │
│              │            │              │
│ seed = 42    │            │ seed = 42    │
│ round-robin  │            │ round-robin  │
│ by class     │            │ by class     │
│              │            │              │
│ indices[0] = │            │ indices[0] = │
│ [0, 2, 4, …] │            │ [1, 3, 5, …] │
│ (pares)      │            │ (impares)    │
│              │            │              │
│ X_batch =    │            │ X_batch =    │
│ features     │            │ features     │
│ [indices]    │            │ [indices]    │
│ (25K, 512)   │            │ (25K, 512)   │
│              │            │              │
│ Y_batch =    │            │ Y_batch =    │
│ Y_train[idx] │            │ Y_train[idx] │
│              │            │              │
└──────────────┘            └──────────────┘
       │                            │
       ├────────────────────────────┤
       │     FORWARD + BACKWARD     │
       │        (both parallel)     │
       │                            │
       ▼                            ▼
┌──────────────────────┐   ┌──────────────────────┐
│ MLP ONLY:            │   │ MLP ONLY:            │
│                      │   │                      │
│ Z1 = X @ W1 + b1     │   │ Z1 = X @ W1 + b1     │
│ A1 = ReLU(Z1)        │   │ A1 = ReLU(Z1)        │
│ Z2 = A1 @ W2 + b2    │   │ Z2 = A1 @ W2 + b2    │
│ A2 = ReLU(Z2)        │   │ A2 = ReLU(Z2)        │
│ Logits = A2@W3+b3    │   │ Logits = A2@W3+b3    │
│                      │   │                      │
│ Loss = XE(L, Y)      │   │ Loss = XE(L, Y)      │
│                      │   │                      │
│ Backward:            │   │ Backward:            │
│ dW1, db1, …          │   │ dW1, db1, …          │
│ dW3 = 128×10         │   │ dW3 = 128×10         │
│ dW2 = 256×128        │   │ dW2 = 256×128        │
│ dW1 = 512×256        │   │ dW1 = 512×256        │
│ Size: ~60 KB         │   │ Size: ~60 KB         │
│                      │   │                      │
└──────────────────────┘   └──────────────────────┘
       │                            │
       │       SEND GRADIENTS       │
       ├────────────────────────────┤
       │                            │
       ▼                            ▼
┌─────────────────────────────────────────────────────────┐
│ PARAMETER SERVER (COLLECTING GRADIENTS)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Received from Worker 0: ∇L_0                          │
│  Received from Worker 1: ∇L_1                          │
│                                                         │
│  ∇̄ = (∇L_0 + ∇L_1) / 2  [average]                    │
│                                                         │
│  W ← W - lr * ∇̄         [SGD update]                  │
│                                                         │
│  Evaluate test:                                         │
│  ├─ X_test_raw (10K, 3, 32, 32)                       │
│  ├─ X_test_feat = CNN(X_test_raw)  [PS CPU]           │
│  ├─ Logits = MLP(X_test_feat)                         │
│  ├─ Accuracy, Loss → history                          │
│  │                                                    │
│  └─ Callback: on_epoch_end(0, 10, 96.5%, 0.12, …)   │
│                                                         │
└─────────────────────────────────────────────────────────┘

[REPEAT for epochs 1-9]

TIME PER EPOCH:
├─ Send PARAMS: 0.1s
├─ Worker compute: 2s (MLP forward+backward)
├─ Receive GRADIENTS: 0.1s (60 KB per worker)
├─ Average + update: 0.1s
└─ Evaluate test: 10s (CNN forward PS CPU)

TOTAL: ~12s / epoch = ~120s for 10 epochs

[BREAKDOWN]
├─ Setup (first time): 60s (extract features)
├─ Setup (cached): 0.5s
├─ 10 epochs: 120s
└─ TOTAL: 60-120s for session
```

---

## 🟠 FLUJO END-TO-END

### **Fase 1: SETUP (menor)**

```
WORKER NODE:
┌──────────────────────────────────────┐
│ Recibe CNN_WEIGHTS                   │
│                                      │
│ NO EXTRAE FEATURES:                  │
│ _X_features = np.empty((0,))        │
│                                      │
│ set_trainable(True)                  │
│ → Pesos CNN requieren gradientes    │
│                                      │
│ Guarda _X_raw en RAM (200 MB)        │
│                                      │
│ ✓ CNN_READY sent to PS               │
│                                      │
│ SETUP TIME: ~1-2s                    │
└──────────────────────────────────────┘
```

### **Fase 2: Training session (EACH EPOCH)**

```
┌─────────────────────────────────────────────────────────┐
│ PARAMETER SERVER                                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [epoch = 0]                                            │
│                                                         │
│  seed_0 = 42  (random)                                 │
│       │                                                 │
│       ├─ Generate PARAMS:                              │
│       │  ├─ epoch: 0                                   │
│       │  ├─ params: {W1, b1, W2, b2, W3, b3}  [MLP]   │
│       │  ├─ seed: 42                                   │
│       │  └─ cnn_params: <weights_bytes>  ← CRITICAL   │
│       │     Size: 50 MB (torch.save CNN state)        │
│       │                                                 │
│       └─► [BROADCAST to all Workers] (50 MB/worker)   │
│              Total network: 50 MB × K workers         │
│                                                         │
└─────────────────────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌──────────────┐            ┌──────────────┐
│ WORKER 0     │            │ WORKER 1     │
├──────────────┤            ├──────────────┤
│              │            │              │
│ Load CNN     │            │ Load CNN     │
│ weights from │            │ weights from │
│ cnn_params   │            │ cnn_params   │
│ (50 MB)      │            │ (50 MB)   [NETWORK TRAFFIC]
│              │            │              │
│ indices =    │            │ indices =    │
│ reconstruct  │            │ reconstruct  │
│ (seed=42)    │            │ (seed=42)    │
│              │            │              │
│ X_raw_batch  │            │ X_raw_batch  │
│ [indices]    │            │ [indices]    │
│ (25K,3,32,32)│            │ (25K,3,32,32)│
│              │            │              │
└──────────────┘            └──────────────┘
       │                            │
       ├────────────────────────────┤
       │  CNN FORWARD+BACKWARD      │
       │  (GPU accelerated if avail)│
       │                            │
       ▼                            ▼
┌──────────────────────┐   ┌──────────────────────┐
│ CNN FORWARD:         │   │ CNN FORWARD:         │
│ X_raw → Conv → … →   │   │ X_raw → Conv → … →   │
│ Features (25K, 512)  │   │ Features (25K, 512)  │
│                      │   │                      │
│ Time: 5-10s          │   │ Time: 5-10s          │
│                      │   │                      │
│ MLP FORWARD:         │   │ MLP FORWARD:         │
│ Features → MLP       │   │ Features → MLP       │
│ Logits, Loss         │   │ Logits, Loss         │
│                      │   │                      │
│ Time: 0.1s           │   │ Time: 0.1s           │
│                      │   │                      │
│ CNN BACKWARD:        │   │ CNN BACKWARD:        │
│ ∂L/∂CNN_weights      │   │ ∂L/∂CNN_weights      │
│ Size: 50 MB gradients│   │ Size: 50 MB gradients│
│ Time: 5-10s          │   │ Time: 5-10s          │
│                      │   │                      │
│ MLP BACKWARD:        │   │ MLP BACKWARD:        │
│ ∂L/∂MLP_weights      │   │ ∂L/∂MLP_weights      │
│ Size: 60 KB          │   │ Size: 60 KB          │
│                      │   │                      │
│ Time: 0.1s           │   │ Time: 0.1s           │
│                      │   │                      │
└──────────────────────┘   └──────────────────────┘
       │                            │
       │    SEND GRADIENTS (BOTH!)  │
       ├────────────────────────────┤
       │                            │
       ▼                            ▼
┌─────────────────────────────────────────────────────────┐
│ PARAMETER SERVER (COLLECTING GRADIENTS)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Received from Worker 0:                                │
│  ├─ MLP gradients: 60 KB                               │
│  ├─ CNN gradients: 50 MB  ← BIG!                       │
│                                                         │
│  Received from Worker 1:                                │
│  ├─ MLP gradients: 60 KB                               │
│  ├─ CNN gradients: 50 MB                               │
│                                                         │
│  [NETWORK TRAFFIC] 50 MB/worker × 2 = 100 MB received  │
│                                                         │
│  Average gradients:                                     │
│  ├─ ∇̄_MLP = (∇MLP[0] + ∇MLP[1]) / 2                  │
│  ├─ ∇̄_CNN = (∇CNN[0] + ∇CNN[1]) / 2                  │
│                                                         │
│  Update parameters:                                     │
│  ├─ MLP: W ← W - lr * ∇̄_MLP                           │
│  ├─ CNN: W ← W - lr * ∇̄_CNN  ← CNN CAMBIA            │
│                                                         │
│  Request TEST_FEATURES from Worker 0:                  │
│  ├─ Worker 0 extracts X_test_feat (5-10s GPU)        │
│  ├─ Sent to PS: 40 MB                                  │
│                                                         │
│  Evaluate test:                                         │
│  ├─ X_test_feat (received from worker) → MLP          │
│  ├─ Logits, Accuracy, Loss → history                  │
│                                                         │
└─────────────────────────────────────────────────────────┘

[REPEAT for epochs 1-9]

TIME PER EPOCH:
├─ Send PARAMS (MLP + CNN weights): 5-10s (network)
├─ Worker CNN forward: 5s
├─ Worker MLP forward: 0.1s
├─ Worker CNN backward: 5s
├─ Worker MLP backward: 0.1s
├─ Receive GRADIENTS: 10-20s (100 MB network)
├─ Average + update: 0.1s (now for both MLP + CNN)
├─ Request + receive TEST_FEATURES: 10-20s (network + extraction)
├─ Evaluate test (MLP only, features from worker): 0.1s
└─ Total: ~35-50s per epoch

TOTAL FOR SESSION:
├─ Setup: 1-2s
├─ 10 epochs: 350-500s
└─ TOTAL: ~6-8 minutes for session

[BOTTLENECK] Network bandwidth (sending/receiving 50 MB CNN params)
```

---

## 📊 Tamaños de datos

| Concepto | Formato | Tamaño | Notas |
|----------|---------|--------|-------|
| **X_raw (train)** | (50K, 3, 32, 32) uint8 | 150 MB | original |
| **X_raw normalized** | (50K, 3, 32, 32) float32 | 6.4 GB | memory, normalizado |
| **X_features** | (50K, 512) float32 | 100 MB | CNN output |
| **Y_train** | (50K,) int32 | 200 KB | etiquetas |
| **W1** | (512, 256) float32 | 512 KB | MLP weight layer 1 |
| **W2** | (256, 128) float32 | 128 KB | MLP weight layer 2 |
| **W3** | (128, 10) float32 | 5 KB | MLP weight layer 3 |
| **MLP total** | all weights + bias | ~1.3 MB | ~60 KB gradients |
| **CNN state** | for simple or resnet | 50-100 MB | depends arch |
| **X_test (raw)** | (10K, 3, 32, 32) uint8 | 30 MB | test data |
| **X_test_features** | (10K, 512) float32 | 20 MB | CNN output test |

---

## 🚨 Network traffic analysis

### **PRECOMPUTED (2 workers, 10 epochs)**

```
Baseline: zero before training
│
├─ CNN_WEIGHTS broadcast
│  └─ 50 MB (simple) × 2 workers = 100 MB ↓
│
├─ Testing (PS CPU):
│  ├─ (no extra network traffic for test)
│  └─ Eval happens on PS
│
├─ Per epoch loop (× 10):
│  ├─ PARAMS send: 1 MB × 2 = 2 MB ↓
│  ├─ GRADIENTS receive: 60 KB × 2 = 120 KB ↑
│  └─ (repeat 10 times)
│
└─ STOP message: <1 KB

TOTAL: ~100 MB down + ~1.2 MB up = 101.2 MB
```

### **END-TO-END (2 workers, 10 epochs)**

```
Baseline: zero before training
│
├─ CNN_WEIGHTS broadcast
│  └─ 50 MB × 2 = 100 MB ↓
│
├─ Per epoch loop (× 10):
│  ├─ PARAMS send (MLP + CNN): 50 MB × 2 = 100 MB ↓
│  ├─ GRADIENTS receive (MLP + CNN): 50 MB × 2 = 100 MB ↑
│  └─ (repeat 10 times)
│
├─ TEST_FEATURES  (epoch 0 or end):
│  └─ 40 MB (features) ↑
│
└─ STOP message: <1 KB

TOTAL: ~1.1 GB down + ~1.1 GB up = 2.2 GB
└─ 20x more than PRECOMPUTED!
```

---

## 🎯 Flujo TEST data

### **PRECOMPUTED**

```
┌─────────────────────────────────┐
│ PS Constructor                   │
├─────────────────────────────────┤
│ X_test, Y_test = load_cifar10.. │ (provided at init)
│        │                         │
│        ▼                         │
│ Stored for evaluation            │
│ (PS CPU only, never to workers)  │
│                                 │
└─────────────────────────────────┘
                │
    ┌───────────┴──────────────────┐
    │                              │
    ▼                       (each epoc)
    ├─ X_feat = CNN(X_test)  [PS local]
    ├─ logits = MLP(X_feat)
    └─ Accuracy, Loss computed locally
       (no network traffic for test)
```

### **END-TO-END**

```
┌─────────────────────────────────┐
│ PS Constructor                   │
├─────────────────────────────────┤
│ X_test, Y_test (provided)         │ (optional in E2E)
│        │                         │
│ Cannot use directly              │
│ reason: CNN on Worker GPU,        │
│        PS CPU                     │
│                                 │
└─────────────────────────────────┘
                │
    ┌───────────┴──────────────────┐
    │       After CNN_READY        │
    ▼                              │
    ├─ REQUEST_TEST_FEATURES       │
    │  to Worker 0                 │
    │                              │
    └─► Worker 0:                  │
        ├─ X_test_feat =           │
        │  CNN.forward(X_test)      │
        │  (uses worker GPU)        │
        │                           │
        └─ SEND TEST_FEATURES       │
           ~40 MB ↑ (network!)      │
           │                        │
    ┌──────┴────────────────────────┐
    │                               │
    ▼  (each epoch)                 │
    ├─ X_feat = _X_test_features   │
    │  (cached from worker)         │
    ├─ logits = MLP(X_feat)        │
    └─ Accuracy, Loss computed      │
       (no further network traffic) │
```

---

## 📝 Resumen: PRECOMPUTED vs END-TO-END

| Aspecto | PRECOMPUTED | END-TO-END |
|---------|-------------|-----------|
| **Setup data flow** | Raw → CNN → Features → Caché | Raw → stored |
| **Epoch data flow** | Features → MLP → Gradients | Raw → CNN → Features → MLP → Gradients (both) |
| **CNN participating** | PS (read-only) | Worker GPU (training) |
| **Test flow** | PS on CPU (fast) | Worker extracts, sends to PS |
| **Network bytes/epoch** | 2 MB (PARAMS) + 120 KB (GRADS) | 100 MB down + 100 MB up |
| **Storage** | Features cached (100-200 MB) | Raw always in RAM |
| **Parallelism** | Perfect (workers independent) | Network becomes bottleneck |

---

**Documento**: `docs/08_data_flow.md`  
**Última actualización**: 2026-03-27  
**Nivel**: Avanzado
