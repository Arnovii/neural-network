# 07. Sistema de Caché: Algoritmo e Invalidación

## Visión General del Sistema de Caché

El caché es un **sistema de dos niveles basado en hash**:

```
Nivel 1 — Pesos CNN:
  Data/feature_cache/{arch}_{seed}_weights.pt
  
Nivel 2 — Features pre-extraídos:
  Data/feature_cache/{arch}_{weights_hash8}_{split}_X.npy
  Data/feature_cache/{arch}_{weights_hash8}_{split}_Y.npy
```

**Idea central**: El hash de los pesos CNN es la **llave de invalidación**. Si los pesos cambian, el hash cambia → nueva clave → nuevos archivos → se re-extraen features automáticamente.

---

## Flujo Completo del Caché

### Paso 1: Computar Hash de Pesos CNN

```python
def _get_weights_hash(self) -> str:
    """
    Calcula MD5 de los pesos CNN actuales.
    
    Esto es determinístico: dado los mismos pesos, siempre el mismo MD5.
    """
    # Serializar state_dict
    weights_bytes = self._get_weights_bytes()
    
    # Calcular MD5
    weight_hash = hashlib.md5(weights_bytes).hexdigest()[:8]
    
    # Ejemplo: "a3f2c8d1"
    return weight_hash
```

### Paso 2: Construir Cache Key

```python
weight_hash = cnn._get_weights_hash()  # "a3f2c8d1"
arch = "simple"
split = "train"

cache_key = f"{arch}_{weight_hash}_{split}"
# Result: "simple_a3f2c8d1_train"
```

### Paso 3: Verificar Existencia en Disco

```python
cache_path_X = f"Data/feature_cache/{cache_key}_X.npy"
cache_path_Y = f"Data/feature_cache/{cache_key}_Y.npy"

if os.path.exists(cache_path_X) and os.path.exists(cache_path_Y):
    # CACHE HIT
    X = np.load(cache_path_X)
    Y = np.load(cache_path_Y)
    print(f"Cache HIT: {cache_key}")
else:
    # CACHE MISS
    print(f"Cache MISS: {cache_key} — extracting...")
    X, Y = cnn.extract_and_save(cache_key)
```

### Paso 4: Extract o Load

```python
def extract_with_cache(...):
    # Si hit: return X, Y
    # Si miss: 
    #   compute X_features = cnn.extract(X_raw) en batches
    #   save X_features
    #   save Y
    #   return X_features, Y
```

---

## Invalidación Automática

**Escenario 1: PRECOMPUTED mode (Hash constante)**

```
Época 0:
  CNN weights: W = [1, 2, 3, ...]
  MD5(W) = "a3f2c8d1"
  extract_with_cache() → cache_key = "simple_a3f2c8d1_train"
    ├─ Hit: No (1ª vez)
    ├─ Extract features (60s)
    ├─ Save "simple_a3f2c8d1_train_X.npy" (600 MB)
    └─ Save "simple_a3f2c8d1_train_Y.npy" (26 KB)

Épocas 1-10:
  CNN weights: W = [1, 2, 3, ...] (NO CAMBIÓ)
  MD5(W) = "a3f2c8d1" (= época anterior)
  extract_with_cache() → cache_key = "simple_a3f2c8d1_train"
    ├─ Hit: SÍ
    ├─ Load "simple_a3f2c8d1_train_X.npy" (0.1s)
    └─ Return features
```

**Timing acumulado**: 60s (epoch 0) + 0.1s × 10 = 60.1s total

**Sin caché** (hipotético): 60s × 11 = 660s total. Speedup: 11x.

---

## Escenario 2: END-TO-END Mode (Hash cambia cada época)

```
Época 0:
  CNN weights: W₀ (iniciales)
  MD5(W₀) = "a3f2c8d1"
  extract_with_cache() → "simple_a3f2c8d1_train"
    ├─ Miss
    ├─ Extract (30s, con GPU)
    └─ Save

Época 1:
  PS actualiza CNN: W₁ = W₀ - lr * gradients
  MD5(W₁) = "7f9a4e2c" (DISTINTO)
  extract_with_cache() → "simple_7f9a4e2c_train"
    ├─ Miss (nueva clave)
    ├─ Extract (30s)
    └─ Save

Época 2:
  PS actualiza CNN: W₂ = W₁ - lr * gradients
  MD5(W₂) = "c3b1d9f5" (DISTINTO)
  extract_with_cache() → "simple_c3b1d9f5_train"
    ├─ Miss (NUEVA clave)
    ├─ Extract (30s)
    └─ Save
```

**Timing**: 30s × 11 epochs = 330s total

**Caché efectividad**: 0% (nunca reutiliza). Pero el código está preparado: si hubiera dos épocas con los mismos CNN weights (hipotético), automáticamente reutilizaría.

---

## ¿Por Qué Usar MD5 en lugar de Alternativas?

### Alternativa 1: Versioning manual (❌ Frágil)

```python
# Mala práctica
cache_version = 1
cache_key = f"simple_v{cache_version}_train"

# Problema: developer olvida incrementar version
# después de cambiar CNN → stale cache
```

### Alternativa 2: Timestamp (❌ No determinista)

```python
# Mala práctica
import time
timestamp = int(time.time())
cache_key = f"simple_{timestamp}_train"

# Problema: mismo CNN boots a tiempos distintos → claves distintas
```

### Alternativa 3: Pedir a Worker (❌ Network waste)

```python
# Mala práctica
"Hey Worker, ¿qué hash tienes? Envíamelo."
# Worker responde con hash

# Problema: latencia de red innecesaria
```

### Elección: MD5 de state_dict (✓ Correcto)

```python
# Buena práctica
weights_bytes = torch.save(cnn.state_dict())
hash = md5(weights_bytes).hexdigest()[:8]

# Ventajas:
# - Determinístico: mismo CNN → mismo hash
# - Automático: no requiere dev intervention
# - Compacto: 8 caracteres hex
```

---

## Detalles de Implementación

### Serialización de Weights (PyTorch)

```python
def _get_weights_bytes(self) -> bytes:
    """Serializa state_dict a bytes."""
    import io
    import torch
    
    buffer = io.BytesIO()
    torch.save(self._model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.read()  # bytes
```

### Cálculo MD5

```python
import hashlib

weight_hash = hashlib.md5(weights_bytes).hexdigest()
# Result: "a3f2c8d1f7c9b1e4a5d3c7f9..." (32 caracteres)

weight_hash_short = weight_hash[:8]
# Result: "a3f2c8d1" (8 caracteres, suficiente para colisiones)
```

**Por qué 8 caracteres**: 16^8 ≈ 4 billones posibles valores. Probabilidad de colisión para 10000 pesos distintos ≈ 10^-9 (negligible).

### Cache Path Construction

```python
cache_dir = "Data/feature_cache"
os.makedirs(cache_dir, exist_ok=True)

arch = self.arch  # "simple" o "resnet18"
weight_hash = self._get_weights_hash()  # "a3f2c8d1"
split = "train"  # o "test"

path_X = f"{cache_dir}/{arch}_{weight_hash}_{split}_X.npy"
path_Y = f"{cache_dir}/{arch}_{weight_hash}_{split}_Y.npy"

# Ejemplo:
# Data/feature_cache/simple_a3f2c8d1_train_X.npy
# Data/feature_cache/simple_a3f2c8d1_train_Y.npy
```

---

## Tamaño del Caché en Disco

### CIFAR-10 Features (PRECOMPUTED mode)

```
X_train: (50000, 512) float32
  Bytes: 50000 × 512 × 4 = 102,400,000 bytes ≈ 100 MB
  
Y_train: (50000,) int32
  Bytes: 50000 × 4 = 200,000 bytes ≈ 200 KB

Total: 100.2 MB

CNN weights (simple): ~20 MB
CNN weights (resnet18): ~50 MB

Total per session:
  - SimpleCNN + features: 120 MB
  - ResNet18 + features: 150 MB
```

### Crecimiento en Disk

```
5 arquitecturas × 3 seeds × 2 splits (train/test) × 100 MB ≈ 3 GB

Típicamente, el disco es suficiente. Pero después de
100 experimentos, puede llegar a 3-10 GB.
```

**Limpieza**: Borrar `Data/feature_cache/` es seguro (se recrea automáticamente).

---

## Comparación: Con Caché vs Sin Caché

### PRECOMPUTED Mode (SimpleCNN)

```
CON CACHÉ:
  Sesión 1, Época 0: 60s (extract + save)
  Sesión 1, Épocas 1-9: 2s × 9 = 18s
  Sesión 2, Época 0: 0.3s (load)
  Sesión 2, Épocas 1-9: 2s × 9 = 18s
  Total 20 epochs (2 sesiones): 60 + 18 + 0.3 + 18 = 96.3s

SIN CACHÉ:
  Sesión 1, Épocas 0-9: 60s × 10 = 600s
  Sesión 2, Épocas 0-9: 60s × 10 = 600s
  Total 20 epochs: 1200s

SPEEDUP: 1200 / 96.3 ≈ 12.5x
```

### END-TO-END Mode (ResNet18)

```
CON CACHÉ:
  Época 0: 30s (extract con CNN₀)
  Época 1: 30s (extract con CNN₁, nueva clave, miss)
  ...
  Época 9: 30s (extract con CNN₉)
  Total: 30 × 10 = 300s

SIN CACHÉ (hipotético):
  Mismo: 300s

SPEEDUP: No existe (never reuses).
```

**Conclusión**: Caché beneficia PRECOMPUTED (12x) pero es neutral en END-TO-END (0x).

---

## Monitoreo de Caché

```python
import os

cache_dir = "Data/feature_cache"
files = os.listdir(cache_dir)

print(f"Cache files: {len(files)}")
total_size = sum(os.path.getsize(f) for f in files)
print(f"Total size: {total_size / 1e9:.2f} GB")

# Ejemplos de archivos:
for f in sorted(files)[:5]:
    print(f"  - {f}")
```

Resultado típico:
```
Cache files: 8
Total size: 0.41 GB
  - resnet18_a3f2c8d1_test_X.npy
  - resnet18_a3f2c8d1_test_Y.npy
  - simple_7f9a4e2c_train_X.npy
  - simple_7f9a4e2c_train_Y.npy
  - simple_a3f2c8d1_train_X.npy
  - simple_a3f2c8d1_train_Y.npy
  - resnet18_a3f2c8d1_train_X.npy
  - resnet18_a3f2c8d1_train_Y.npy
```

---

## Performance Tuning del Caché

### Batch Size en Extracción

Para reducir memoria:

```python
batch_size_extract = 512  # vs default 2048

# En extraction loop
for batch_start in range(0, len(X), batch_size_extract):
    batch_end = min(batch_start + batch_size_extract, len(X))
    X_batch_features = cnn.extract(X[batch_start:batch_end])
```

**Trade-off**:
- batch_size=512: Memory light, pero más iteraciones (lento)
- batch_size=4096: Memory heavy, menos iteraciones (rápido)

### Usar I/O Faster (SSD vs HDD)

Si `Data/` está en HDD:
- Load ~100 MB: 3-5 segundos
- En SSD: 0.5 segundos

**Solución**: Mover `Data/feature_cache/` a SSD si disponible.

```bash
# Linux/Mac
ln -s /path/to/ssd/cache Data/feature_cache
```

