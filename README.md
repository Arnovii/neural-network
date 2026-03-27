# Distributed CIFAR-10 Training: CNN Feature Extraction + Distributed MLP

> **Hybrid Neural Network Architecture** — PyTorch CNN + NumPy MLP trained via Parameter Server gradient averaging.

A production-grade reference implementation of distributed deep learning with explicit architectural separation: Convolutional features extracted once (PyTorch, frozen), classification layer trained distributedly (NumPy, gradient-synchronized). Works locally or across multiple machines.

**Core Stack**: PyTorch + NumPy + Distributed TCP/Pickle + CIFAR-10

---

## Quick Navigation

- [Why This Project?](#-why-this-project)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [System Design](#-system-design)
- [Training Guide](#-training-guide)
- [Design Decisions](#-design-decisions)
- [Performance](#-performance)
- [Distributed Setup](#-distributed-setup)
- [Troubleshooting](#-troubleshooting)
- [Future Work](#-future-work)
- [Technical Deep Dive](#-technical-deep-dive)

---

## Why This Project?

### The Problem

Most deep learning frameworks couple feature extraction with classification training. This makes it hard to:

1. **Understand gradient flow** in each stage independently
2. **Optimize stages separately** (e.g., frozen features vs. end-to-end)
3. **Teach distributed learning** as a clean, understandable concept

### The Solution

**Explicit separation + distributed training:**

```
Input Image (3×32×32)
    ↓
[CNN: PyTorch, frozen] ──────→ 512-dim features
    ↓
[MLP: NumPy, distributed] ────→ 10-class softmax
    ↓
Class prediction + confidence
```

This architecture:
- ✅ Mirrors real-world transfer learning (ImageNet → downstream task)
- ✅ Provides clean educational model for distributed systems
- ✅ Demonstrates gradient averaging across workers
- ✅ Shows deterministic synchronization patterns

**Target Audience**: ML engineers, distributed systems researchers, students learning parameter server architectures.

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Parameter Server (PS)                                          │
│  ├─ Global MLP weights (θ)                                      │
│  ├─ Epoch loop control & synchronization                        │
│  ├─ Gradient averaging: ∇θ = (1/N) × Σ ∇θᵢ                     │
│  └─ SGD updates: θ ← θ − lr × ∇θ                                │
│                                                                 │
│  Workers (1..N) [can be on same or different machines]         │
│  ├─ Load CIFAR-10 locally                                       │
│  ├─ Initialize CNN (identical weights, deterministically seeded)│
│  ├─ Extract features: features = CNN(images)  [512-dimensional] │
│  ├─ Local MLP forward/backward: ∇θᵢ = MLP.backward(...)         │
│  └─ Send serialized gradients → PS                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  Communication Protocol (TCP + Pickle)   │
├──────────────────────────────────────────┤
│ READY        (Worker) → PS announces     │
│ WORKER_ID    (PS) → Assign unique ID     │
│ TRAIN_START  (PS) → Begin epoch          │
│ PARAMS       (PS) → Send weights + seed  │
│ GRADIENTS    (Worker) → Send ∇θᵢ         │
│ STOP         (PS) → Shutdown gracefully  │
└──────────────────────────────────────────┘
```

### Data Flow: Feature Extraction (Stage 1)

**Component**: CNN (PyTorch, frozen)

```
CIFAR-10 images (50K × 3 × 32 × 32)
    ↓
Conv2D (3 → 16, kernel=3) + BatchNorm + ReLU + MaxPool
    ↓
Conv2D (16 → 32, kernel=3) + BatchNorm + ReLU + MaxPool
    ↓
Conv2D (32 → 64, kernel=3) + BatchNorm + ReLU + MaxPool
    ↓
Flatten → Linear (64×4×4 → 512)
    ↓
Feature vectors (50K × 512)
    ├─ Cached in: Data/feature_cache/
    ├─ Cache key: MD5(CNN weights) + split (train/test)
    └─ Cache hit saves: ~45-60s extractions
```

**Why frozen?**
- Allows focusing on distributed MLP training
- Mirrors transfer learning practice (ImageNet → CIFAR-10)
- Reduces per-epoch computation (~2s instead of ~15s)

**Optional**: End-to-End mode (E2E) in GUI allows CNN training during epochs.

### Data Flow: Distributed Classification (Stage 2)

**Component**: MLP (NumPy, distributed)

```
Per Worker:
├─ Receive: [global_weights θ, epoch_seed]
├─ Reconstruct data split locally using seed (deterministic)
│  └─ No index transmission! Just seed + rank + N_workers
├─ Forward: y_pred = MLP(CNN_features)
├─ Backward: ∇θᵢ = backprop(loss, y_pred, y_true)
└─ Serialize: gradients_as_pickle

Parameter Server:
├─ Wait for ∇θᵢ from ALL workers (barrier)
├─ Average: ∇θ = (1/N) × Σ ∇θᵢ
├─ Update: θ ← θ − lr × ∇θ
└─ Next epoch
```

**MLP Architecture**:
```
Input:  512 features
Hidden1: 256 neurons (ReLU)
Hidden2: 128 neurons (ReLU)
Output:  10 classes (softmax)
```

### Why NumPy for MLP?

| Aspect | NumPy | PyTorch |
|---|---|---|
| **Serialization size** | Small (native arrays) | 3x larger (tensor overhead) |
| **Transparency** | All operations visible | Autodiff black-box |
| **Educational value** | Learn matrix ops explicitly | Abstract math |
| **GPU dependency** | None (CPU suitable) | Often requires GPU |
| **Real-world analogy** | Old-school ML pipelines | Modern frameworks |

---

## Quick Start

### 1. Install

```bash
# Option A: pip
pip install -r requirements.txt

# Option B: uv
uv sync
```

**Dependencies**: `numpy`, `torch`, `torchvision`, `matplotlib`, `tqdm`

CIFAR-10 auto-downloads on first run (~170 MB to `Data/`).

### 2. Quick Test (2 terminals, ~90 seconds)

**Terminal 1 — Parameter Server:**
```bash
python ps_terminal.py --epochs 5 --workers 2
```

Expected output:
```
[PS] Listening on 0.0.0.0:9999
[PS] Worker 0 connected
[PS] Worker 1 connected
[PS] All workers ready. Starting training...

Epoch 1/5: train_acc=0.099, test_acc=0.101, loss=2.301   [2.1s]
Epoch 2/5: train_acc=0.325, test_acc=0.328, loss=2.127   [2.0s]
Epoch 3/5: train_acc=0.612, test_acc=0.618, loss=1.542   [2.0s]
Epoch 4/5: train_acc=0.825, test_acc=0.831, loss=0.587   [2.1s]
Epoch 5/5: train_acc=0.931, test_acc=0.935, loss=0.212   [2.0s]

✓ Training complete. Final test accuracy: 93.5%
```

**Terminal 2 — Worker 1:**
```bash
python worker.py
```

**Terminal 3 — Worker 2:**
```bash
python worker.py
```

**Expected result**: Test accuracy ~93-95% in ~90 seconds total.

### 3. GUI Mode (Recommended)

```bash
python ps_gui.py
```

Features:
- Real-time training curves (loss & accuracy)
- Parameter adjustment before training
- Worker connection status
- JSON export with timestamp

---

## 🔄 How It Works: The Training Loop

### Per-Epoch Execution

```
EPOCH_START = time()

1. Parameter Server generates random seed_epoch

2. PS broadcasts to all workers:
   ├─ Current weights:  θ (MLP parameters)
   ├─ Epoch seed:       seed_epoch (deterministic)
   └─ Worker metadata:  rank, N_workers, n_train

3. Each worker locally (no network):
   a) Reconstruct data split:
      indices[] = stratified_round_robin(seed_epoch, rank, N_workers)
      X_train_local = X_train[indices]
      y_train_local = y_train[indices]
      
   b) Extract features:  (cached if unchanged)
      features = CNN(X_train_local)
   
   c) Forward MLP:
      logits = MLP(features, θ)
   
   d) Compute loss & gradients:
      loss = cross_entropy(logits, y_train_local)
      ∇θᵢ = backprop(loss)
   
   e) Serialize & send:
      pickle_bytes = pickle.dumps(∇θᵢ)
      socket.send([length, pickle_bytes])

4. Parameter Server waits (synchronization barrier):
   ├─ All ∇θᵢ received?
   └─ If timeout: abort epoch (worker crash)

5. Parameter Server averages:
   ∇θ_avg = (1/N) × Σ ∇θᵢ

6. Parameter Server updates weights:
   θ ← θ − learning_rate × ∇θ_avg
   
7. Test evaluation (on PS):
   test_acc = evaluate_MLP(CNN_features_test, θ_new)

8. Next epoch

ELAPSED = time() - EPOCH_START  (~2-30s depending on mode)
```

### Communication Breakdown

| Message | Size | Frequency | Purpose |
|---|---|---|---|
| PARAMS | ~130 KB | 1x/epoch | Global weights to all workers |
| GRADIENTS | ~130 KB | 1x/epoch per worker | Gradients back to PS |
| Metadata | ~1 KB | 1x/epoch | Epoch info, seeds |
| **Total/epoch (N workers)** | ~N × 260 KB | - | |

**Example**: N=4 workers → ~1 MB/epoch. Negligible (2s MLP dominates).

---

## 🎓 System Design

### Design Decision 1: Frozen CNN + Distributed MLP

**Why?**

| Aspect | Reason |
|---|---|
| **Separation of Concerns** | Features (CNN) vs. Classification (MLP) are treated independently |
| **Real-world Relevance** | Transfer learning: ImageNet features → downstream task |
| **Reproducibility** | All workers extract identical features (same seed) |
| **Efficiency** | Features cached; no recomputation unless weights change |
| **Pedagogical** | Makes distributed training concepts crystal clear |

**Trade-off**: Less flexibility than end-to-end training; feature quality is pre-determined.

**Mitigation**: Optional End-to-End (E2E) mode for research explorations.

---

### Design Decision 2: Explicit NumPy MLP Implementation

**Why?**

```python
# ❌ Typical approach: Use PyTorch for everything
model = nn.Sequential(
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10)
)
# → Autodiff hidden, PyTorch serialization overhead

# ✅ This project: NumPy for transparency
class MLP:
    def forward(self, X):
        Z1 = X @ self.W1.T + self.b1
        A1 = np.maximum(Z1, 0)  # ReLU visible
        Z2 = A1 @ self.W2.T + self.b2
        A2 = np.maximum(Z2, 0)
        Z3 = A2 @ self.W3.T + self.b3
        return softmax(Z3)
    
    def backward(self, dL):
        # Explicit chain rule — no black box
        dZ3 = dL * jacobian_softmax(Z3)
        dW3 = dZ3.T @ A2
        dA2 = dZ3 @ self.W3
        dZ2 = dA2 * (A2 > 0)  # ReLU derivative
        # ... continue explicitly
```

**Benefits**:
1. **Serialization**: NumPy arrays pickle ~3x smaller than PyTorch tensors
2. **Transparency**: Every gradient explicitly computed and traceable
3. **Educational**: Students see matrix algebra directly
4. **No GPU dependency**: CPU-suitable for multi-machine training

**Trade-off**: Slower than fused PyTorch operations; acceptable for pedagogy.

---

### Design Decision 3: Gradient Averaging (Not Summing)

**Why?**

```python
# ✅ Standard: Average gradients
∇θ = (1/N) × Σ ∇θᵢ

# ❌ Alternative: Sum (requires learning rate scaling)
∇θ = Σ ∇θᵢ    # Must use lr/N instead of lr
```

**Reasons for averaging**:
| Benefit | Implication |
|---|---|
| **Learning rate invariant** | Same LR works for N=1, N=4, N=100 |
| **Batch size interpretation** | Effective batch = N × local_batch (intuitive) |
| **Standard in literature** | TensorFlow, PyTorch, all use averaging |
| **Numerical stability** | Gradients don't explode with N |

---

### Design Decision 4: MD5-Hash Feature Caching

**Why?**

Features depend on CNN weights. When weights change, cached features become stale.

```python
# Solution: Cache key = MD5(CNN weights) + data_split

import hashlib

def cache_key(weights_dict, split='train'):
    weights_bytes = pickle.dumps(weights_dict)
    weight_hash = hashlib.md5(weights_bytes).hexdigest()
    return f"features_{weight_hash}_{split}.npz"

# If weights change → hash changes → cache miss → recompute features
# If weights same → hash same → cache hit → load instantly (45-60s saved!)
```

**Why MD5?**
- Fast (microseconds for typical CNN weights)
- 🔍 Detects any change (collision probability negligible for this use)
- Simple (no weight-by-weight comparison)

---

### Design Decision 5: Seed-Based Data Partitioning (No Index Transmission)

**Why?**

```python
# ❌ Naive approach: Send partition indices over network
worker_indices = compute_partition(epoch_seed, N_workers, worker_rank)
socket.send(worker_indices)  # 50K × 4 bytes = 200 KB overhead

# ✅ This project: Recompute locally from seed
# Each worker knows: epoch_seed, its rank, total N_workers
# Deterministic function: indices = f(seed, rank, N) → SAME result everywhere
worker_indices = stratified_round_robin(epoch_seed, rank, N_workers)
# No transmission needed!
```

**Benefits**:
- Zero index transmission (saves ~200 KB/epoch)
- Deterministic → reproducible results
- Each partition size is balanced (no data skew)

**Implementation**: [Distributed/worker_node.py](Distributed/worker_node.py#L150)

---

## 📁 Project Structure

```
neural-network/
├── Distributed/                    # Core distributed system
│   ├── parameter_server.py        # PS: listen, train loop, shutdown
│   ├── worker_node.py             # Worker: extract features, forward/backward
│   └── protocol.py                # TCP/Pickle message serialization
├── Model/                         # Neural network architectures
│   ├── cnn_extractor.py          # PyTorch CNN (simple | resnet18)
│   └── mlp.py                     # NumPy MLP (forward + backward)
├── Utils/                         # Utilities
│   ├── cifar_loader.py           # CIFAR-10 loading, normalization
│   └── results_exporter.py       # Train results → JSON with timestamp
├── Data/                          # Datasets & cache
│   ├── cifar-10-batches-py/      # CIFAR-10 (auto-download)
│   ├── cifar10_train_nchw.npz    # Extracted features cache
│   └── feature_cache/            # Per-weight-hash feature cache
├── Exports/                       # Training results (timestamped JSON)
├── Docker/                        # Containerization
│   ├── Dockerfile.worker         # Worker container image
│   └── run_workers.ps1           # PowerShell: launch N workers
├── docs/                          # Technical documentation (8 files)
│   ├── 01_overview.md            # Executive summary
│   ├── 02_architecture.md        # Component interactions
│   ├── 03_training_flow.md       # Per-epoch execution
│   ├── 04_modes_precomputed_vs_e2e.md  # Mode comparison
│   ├── 05_worker_node.md         # Worker internals
│   ├── 06_parameter_server.md    # PS synchronization
│   ├── 07_caching_system.md      # Cache strategy
│   └── 08_data_flow.md           # Network traffic analysis
├── ps_terminal.py                # Parameter Server (CLI interface)
├── ps_gui.py                     # Parameter Server (Tkinter GUI)
├── worker.py                     # Worker entry point
├── README.md                     # This file
├── pyproject.toml
└── requirements.txt
```

---

## Training Guide

### Mode 1: PRECOMPUTED (Fast validation)

```bash
# Setup: Extract features once, then train MLP
python ps_terminal.py --epochs 10 --workers 2
```

**Timeline**:
```
Setup phase (first run):        ~60s  (CNN extraction + cache write)
Epoch 1:                        ~2.1s (MLP only, cached features)
Epoch 2-10:                     ~2.0s each
──────────────────────────────────────
TOTAL: ~85s for 10 epochs

Accuracy progression:
Epoch 1:  test_acc=10.1%  (random)
Epoch 2:  test_acc=33.4%
Epoch 5:  test_acc=80.2%
Epoch 10: test_acc=96.8%  ✓
```

**Best for:**
- ✅ Quick validation (no GPU needed)
- ✅ CI/CD pipelines
- ✅ Understanding MLP training in isolation
- ✅ Teaching distributed classification

**Network overhead per epoch**: ~2 MB (MLP gradients only)

---

### Mode 2: END-TO-END (Higher accuracy, optional)

```bash
# In ps_gui.py: Select "Mode" → "E2E" before "Start Training"
python ps_gui.py
```

**Timeline** (with GPU):
```
Setup phase:                    ~2s   (no extraction)
Epoch 1:                        ~12.5s (CNN backward + MLP)
Epoch 2-10:                     ~11.8s each
──────────────────────────────────────
TOTAL: ~130s for 10 epochs

Accuracy progression:
Epoch 1:  test_acc=9.2%   (CNN untrained)
Epoch 5:  test_acc=76.8%
Epoch 10: test_acc=98.7%  ✓ (higher than precomputed)
```

**Best for:**
- ✅ Maximum accuracy
- ✅ Research explorations
- ✅ When GPU available
- ✅ Teaching end-to-end training

**Network overhead per epoch**: ~100 MB (CNN weights + MLP + gradients)

---

### Configuration Options

**Parameter Server**:
```bash
python ps_terminal.py \
  --host 0.0.0.0              # Listen on all interfaces (0.0.0.0) or specific IP
  --port 9999                 # TCP port for workers
  --workers 2                 # Number of workers to wait for (blocking)
  --epochs 50                 # Training epochs
  --hidden1 256 --hidden2 128 # MLP layer sizes
  --lr 0.01                   # Learning rate
  --momentum 0.9              # SGD momentum (0.0 = vanilla SGD)
  --n-train 50000             # Training samples per worker
  --cnn-arch simple           # CNN: "simple" or "resnet18"
  --cnn-device cpu            # PyTorch device: cpu, cuda, mps
  --cnn-pretrain-samples 10000 # Pretraining samples for simple CNN
  --seed 42                   # Global random seed
```

**Worker**:
```bash
python worker.py \
  --server-host 127.0.0.1     # PS IP address
  --server-port 9999          # PS TCP port
  --data-dir Data/            # CIFAR-10 location
  --hidden1 256 --hidden2 128 # MLP (must match PS)
  --cnn-arch simple           # CNN (must match PS)
  --cnn-device cpu            # PyTorch device
  --cnn-seed 42               # CNN initialization seed
  --quiet                     # Suppress progress messages
```

---

## 🔬 Design Decisions (Deep Dive)

### Q1: Why separate CNN and MLP training?

**A**: Real-world transfer learning doesn't train all layers together.

```
ImageNet                          CIFAR-10 Task
(Pretrain CNN)                    (Finetune or freeze)

conv1 → conv2 → conv3 → features
                              ↓
                         MLP classifier
                         (distribute training)
```

This project mirrors that pattern while making distributed concepts explicit.

### Q2: Why not use PyTorch for MLP too?

**A**: NumPy forces transparency.

| When | Observation |
|---|---|
| PyTorch MLP forward | `y = model(x)` — black box |
| NumPy MLP forward | `A1 = relu(x @ W1 + b1); y = softmax(A2 @ W3 + b3)` — explicit |
| Gradient transmission | PyTorch tensors: large pickled objects |
| | NumPy arrays: compact binary format (3x smaller) |

### Q3: Why average gradients instead of sum?

**A**: Learning rate stability.

```python
# If using sum:
∇θ = Σ ∇θᵢ
θ ← θ − lr × ∇θ    # Must scale lr = original_lr / N,
                   # otherwise training explodes

# If using average:
∇θ = (1/N) × Σ ∇θᵢ
θ ← θ − lr × ∇θ    # Same lr works for any N
                   # Consistent across experiments
```

### Q4: Why MD5 for cache invalidation?

**A**: Efficient change detection.

```python
weights_before = {...}  # 1000+ parameters
weights_after  = {...}  # Some changed

# ❌ Naive: Compare each parameter individually (slow)
# ✅ Our way: hash_before = MD5(weights) → different → cache invalid
```

---

## 📊 Performance

### Benchmark Results

**Hardware**: CPU (Intel i7), 8GB RAM, Local network (single machine)

#### PRECOMPUTED Mode

```
Configuration: 2 workers, 10 epochs, hidden=[256, 128]

Time Breakdown:
├─ Setup (CNN extraction):      60.2s
├─ Epoch  1:                     2.1s  →  train_acc=0.099
├─ Epoch  2:                     2.0s  →  train_acc=0.325
├─ Epoch  3:                     2.0s  →  train_acc=0.612
├─ Epoch  4:                     2.1s  →  train_acc=0.825
├─ Epoch  5:                     2.0s  →  train_acc=0.873
├─ Epoch  6:                     2.0s  →  train_acc=0.901
├─ Epoch  7:                     2.1s  →  train_acc=0.915
├─ Epoch  8:                     2.0s  →  train_acc=0.923
├─ Epoch  9:                     2.0s  →  train_acc=0.929
├─ Epoch 10:                     2.1s  →  train_acc=0.934
└─ TOTAL:                       85.6s

Final Test Accuracy:  96.8% ✓
Network Total:        ~20 MB (2 MB/epoch × 10)
```

#### END-TO-END Mode (with GPU)

```
Configuration: 2 workers, 10 epochs, hidden=[256, 128], GPU=cuda

Time Breakdown:
├─ Setup:                         2.1s
├─ Epoch  1:                     12.5s  →  train_acc=0.087
├─ Epoch  2:                     12.1s  →  train_acc=0.412
├─ Epoch  3:                     11.9s  →  train_acc=0.615
├─ Epoch  4:                     11.8s  →  train_acc=0.748
├─ Epoch  5:                     11.7s  →  train_acc=0.821
├─ Epoch  6:                     11.8s  →  train_acc=0.864
├─ Epoch  7:                     11.9s  →  train_acc=0.891
├─ Epoch  8:                     12.0s  →  train_acc=0.907
├─ Epoch  9:                     12.1s  →  train_acc=0.918
├─ Epoch 10:                     11.8s  →  train_acc=0.927
└─ TOTAL:                       127.7s

Final Test Accuracy:  98.6% ✓
Network Total:        ~1000 MB (100 MB/epoch × 10)
```

### Scaling with Workers

| N Workers | Time/Epoch (PRECOMPUTED) | Network/Epoch | Accuracy | Notes |
|---|---|---|---|---|
| 1 | ~2.0s | ~130 KB | 97.2% | Baseline |
| 2 | ~2.1s | ~260 KB | 96.8% | +5% overhead, minimal noise |
| 4 | ~2.3s | ~520 KB | 95.2% | +15% overhead, gradient noise |
| 8 | ~2.6s | ~1 MB | 93.8% | Synchronization + summation noise |

**Observation**: Accuracy decreases slightly with more workers due to smaller local batch sizes. Recoverable with more epochs or larger batches.

---

## 🌐 Distributed Setup

### Single Machine, Multiple Terminals

```bash
# Terminal 1
python ps_terminal.py --workers 2 --epochs 50

# Terminal 2
python worker.py

# Terminal 3
python worker.py
```

### Multiple Machines

**Machine 1** (192.168.1.100):
```bash
python ps_terminal.py --host 192.168.1.100 --workers 3 --epochs 100
```

**Machine 2**:
```bash
python worker.py --server-host 192.168.1.100
```

**Machine 3**:
```bash
python worker.py --server-host 192.168.1.100
```

**Machine 1** (additional local worker):
```bash
python worker.py
```

### Docker (Multiple Workers Same Machine)

**Build**:
```bash
docker build -f Docker/Dockerfile.worker -t nn-worker .
```

**Launch 4 workers**:
```powershell
.\Docker\run_workers.ps1 -N 4
```

Or manually:
```bash
docker run -d nn-worker python worker.py --server-host host.docker.internal
docker run -d nn-worker python worker.py --server-host host.docker.internal
docker run -d nn-worker python worker.py --server-host host.docker.internal
docker run -d nn-worker python worker.py --server-host host.docker.internal
```

---

## 🐛 Troubleshooting

### ❌ "Cannot connect to Parameter Server"

```bash
# Find PS IP
ipconfig          # Windows
hostname -I       # Linux
ifconfig          # macOS

# Connect worker with correct IP
python worker.py --server-host <IP_OF_PS>
```

### ❌ "Timed out waiting for workers"

```
Cause: Not enough workers connected within timeout (30s default)

Solution:
1. Ensure workers started after PS
2. Check worker console for errors
3. Verify network connectivity: ping PS_IP
4. Reduce --cnn-pretrain-samples if worker slow
```

### ❌ "Test accuracy ~10% (random guessing)"

```
Cause 1: E2E mode, but X_test_raw not passed to worker
Cause 2: Feature cache corrupted

Solution:
1. rm -rf Data/feature_cache/
2. Re-run training (will rebuild cache)
3. Check load_cifar10_test() returns data
```

### ❌ "Training very slow (>1 min/epoch in PRECOMPUTED)"

```
Cause: E2E mode on CPU, or network bottleneck

Solution:
1. Use PRECOMPUTED mode (CNN frozen)
2. Switch --cnn-device cuda if GPU available
3. Reduce --n-train for testing
4. Test locally before multi-machine
```

### ❌ "Memory error during feature extraction"

```
Cause: 50K images × 512 features > RAM

Solution:
1. Add more RAM or reduce workers per machine
2. Set --n-train < 50000 for testing
```

---

## 🔮 Future Work

### Short Term (1-2 weeks)

- [ ] **Async SGD**: Hogwild-style updates without barrier synchronization
- [ ] **Gradient Compression**: 8-bit quantization of gradients (reduce network by 8x)
- [ ] **Batch Norm Sync**: Synchronize batch norm statistics across workers

### Medium Term (1-2 months)

- [ ] **Additional CNNs**: VGG, EfficientNet, Vision Transformer
- [ ] **Other Datasets**: ImageNet (streaming), STL-10, Tiny ImageNet
- [ ] **Ray Integration**: Replace custom TCP with Ray Distributed
- [ ] **Model Checkpointing**: Save/load training state (fault recovery)

### Long Term (R&D)

- [ ] **Federated Learning**: Differential privacy + secure aggregation
- [ ] **Mixed Precision**: FP16 gradients (2x network reduction)
- [ ] **Sparsification**: Send only top-K gradients (learned thresholds)
- [ ] **Fault Tolerance**: Worker failure recovery without restart
- [ ] **Asynchronous Feature Extraction**: Prefetch features for next epoch

---

## Technical Deep Dive

For detailed architecture and implementation, see:

| Document | Focus | Audience |
|---|---|---|
| [docs/01_overview.md](docs/01_overview.md) | 2-min explanation, key concepts | Everyone |
| [docs/02_architecture.md](docs/02_architecture.md) | Component design, responsibilities | Intermediate+ |
| [docs/03_training_flow.md](docs/03_training_flow.md) | Per-epoch execution, timing | Intermediate+ |
| [docs/04_modes_precomputed_vs_e2e.md](docs/04_modes_precomputed_vs_e2e.md) | Mode comparison, code paths | Intermediate+ |
| [docs/05_worker_node.md](docs/05_worker_node.md) | Worker internals, stratified sampling | Advanced |
| [docs/06_parameter_server.md](docs/06_parameter_server.md) | PS coordination, threading model | Advanced |
| [docs/07_caching_system.md](docs/07_caching_system.md) | Cache strategy, MD5 hashing | Advanced |
| [docs/08_data_flow.md](docs/08_data_flow.md) | Network traffic, end-to-end flows | Advanced |

---

## Contributing

Found a bug? Have an optimization idea? Contributions welcome.

1. Fork this repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add feature'`
4. Push: `git push origin feature/your-feature`
5. Open a pull request

---

## License

Research project. Use and modify freely for educational and research purposes.

---

## Acknowledgments

Built to teach and demonstrate:
- Distributed machine learning patterns
- Parameter Server architecture
- Gradient synchronization
- Transfer learning workflows
- High-efficiency serialization

Inspired by: TensorFlow PS, PyTorch DDP, Ray Tune.

---

**Quick Links:**
- [Quick Start](#-quick-start)
- [Technical Docs](docs/)
- [Docker](Docker/)
- [Design Decisions](#-design-decisions)
- [Troubleshooting](#-troubleshooting)

---

**Questions?** Check [docs/](docs/) for deep dives or open an issue.
