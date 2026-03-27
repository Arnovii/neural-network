# 7. SISTEMA DE CACHÉ DE FEATURES

## 🎯 Propósito

Evitar recalcular las features CNN si los pesos no han cambiado. En PRECOMPUTED, esto puede ahorrar **30-60 segundos** por sesión (1 setup).

**Idea**: Si CNN tiene los mismos pesos que la última vez, los features son idénticos → reutilizar del disco.

---

## 🔑 Estrategia de cache key

### **Definición**

```
cache_key = "{arch}_{weights_hash}_{split}"

Ejemplo:
- arch = "simple"
- weights_hash = "abc123de"  (primeros 8 hex de MD5 de pesos)
- split = "train" (o "test")

cache_key = "simple_abc123de_train"
```

### **Hash de pesos**

```python
def _weights_hash(self) -> str:
    """
    Calcula MD5 de los pesos CNN.
    
    1. Serializar state_dict a bytes (determinístico)
    2. MD5(bytes) → "abc123def456…"
    3. Tomar primeros 8 hex → "abc123de"
    
    Propiedad: pesos idénticos → hash idéntico
              pesos diferentes → hash diferente (con ~99.999% prob)
    """
    import hashlib
    import io
    
    # Serializar state_dict
    buffer = io.BytesIO()
    torch.save(self._model.state_dict(), buffer)
    data = buffer.getvalue()
    
    # MD5
    full_hash = hashlib.md5(data).hexdigest()
    return full_hash[:8]  # abc123de
```

### **Rutas de caché**

```
Data/feature_cache/
├── simple_abc123de_train_X.npy      (200 MB, features 50K train)
├── simple_abc123de_train_Y.npy      (200 KB, etiquetas 50K train)
│
├── simple_abc123de_test_X.npy       (40 MB, features 10K test)
├── simple_abc123de_test_Y.npy       (40 KB, etiquetas 10K test)
│
├── resnet18_def456ab_train_X.npy
├── resnet18_def456ab_train_Y.npy
│
└── … (más combinaciones de hash)
```

---

## 🔄 Flujo de carga con caché

```python
def _load_features_with_cache(X, Y, arch, batch_size, split="train"):
    """
    Intenta cargar features con caché. Si no existe o está corrupto,
    extrae y guarda.
    """
    
    # PASO 1: Calcular cache key
    weights_hash = self._weights_hash()
    cache_key = f"{arch}_{weights_hash}_{split}"
    cache_X_path = f"Data/feature_cache/{cache_key}_X.npy"
    cache_Y_path = f"Data/feature_cache/{cache_key}_Y.npy"
    
    # PASO 2: Intentar cargar (CACHE HIT)
    if os.path.exists(cache_X_path) and os.path.exists(cache_Y_path):
        try:
            X_feat = np.load(cache_X_path, allow_pickle=False)
            Y_cached = np.load(cache_Y_path, allow_pickle=False)
            
            # Validar shape
            if X_feat.shape == (len(X), 512) and Y_cached.shape == Y.shape:
                print(f"[CACHE HIT][{split.upper()}] {cache_key}")
                return X_feat, Y
            else:
                print(f"[SHAPE INVALID][{split.upper()}] Regenerating…")
        except Exception as e:
            print(f"[CACHE CORRUPT][{split.upper()}] {e}. Regenerating…")
    
    # PASO 3: CACHE MISS — extraer features nuevas
    print(f"[CACHE MISS][{split.upper()}] Extracting with CNN…")
    
    t0 = time.perf_counter()
    X_feat = self._cnn.extract_batched(X, batch_size=batch_size, verbose=True)
    t_extract = time.perf_counter() - t0
    
    # PASO 4: Validar shape
    assert X_feat.shape == (len(X), 512), f"Shape mismatch: {X_feat.shape}"
    
    # PASO 5: Guardar en caché
    try:
        os.makedirs("Data/feature_cache", exist_ok=True)
        np.save(cache_X_path, X_feat)
        np.save(cache_Y_path, Y)
        print(f"[CACHE SAVE][{split.upper()}] {cache_key} ({t_extract:.1f}s)")
    except Exception as e:
        print(f"[CACHE SAVE ERROR][{split.upper()}] {e}")
        # Continuar sin guardar — no es fatal
    
    return X_feat, Y
```

---

## 📊 Formato de archivos .npy

**Archivo**: `simple_abc123de_train_X.npy`
```
┌─────────────────────────────────────┐
│ NumPy .npy Format                   │
├─────────────────────────────────────┤
│ Magic (6 bytes)     : \x93 N U M P Y │
│ Version (2 bytes)   : \x01 \x00     │
│ Header length (2 bytes) : depends   │
│ Header (JSON)       : {"descr": "< │
│                       f8", „shape": │
│                       [50000, 512], │
│                       "fortran…     │
│ Data: 50000 × 512 × 8 bytes = 200 MB│
│ (float64)                           │
└─────────────────────────────────────┘
```

**Propiedades**:
- ✅ Binario (eficiente)
- ✅ NumPy native (fast load)
- ✅ Datos tipados (float32/float64, int32, etc.)
- ✅ Metadatos en header (shape, dtype)

---

## 🎯 Cache hit / Miss patterns

### **Setup 1: Primera ejecución**

```
PS carga CNN (weights iniciales)
Worker recibe CNN_WEIGHTS
│
├─ Calcula hash: "abc123de"
├─ Busca: Data/feature_cache/simple_abc123de_train_X.npy
├─ NO existe (primer uso)
│
├─ [CACHE MISS]
├─ Extrae features
├─ Guarda en caché
│
└─ Tiempo: ~45s (extracción + I/O)
```

### **Setup 2: Misma CNN, nueva sesión**

```
PS carga CNN (mismo archivo de pesos que Setup 1)
Worker recibe CNN_WEIGHTS
│
├─ Calcula hash: "abc123de" (igual que antes)
├─ Busca: Data/feature_cache/simple_abc123de_train_X.npy
├─ EXISTE
│
├─ [CACHE HIT]
├─ Carga desde disco (fast)
│
└─ Tiempo: ~0.5s (I/O de 200 MB)
```

### **Setup 3: CNN diferente, nueva sesión**

```
PS entrena CNN E2E, guarda nuevos pesos
PS carga CNN (pesos actualizados)
Worker recibe CNN_WEIGHTS
│
├─ Calcula hash: "def456ab" (diferente que Setup 1)
├─ Busca: Data/feature_cache/simple_def456ab_train_X.npy
├─ NO existe (pesos nuevos)
│
├─ [CACHE MISS]
├─ Extrae features con CNN nueva
├─ Guarda en caché con nuevo hash
│
└─ Tiempo: ~45s (extracción + I/O)
```

---

## ⚡ Impacto de rendimiento

### **Sin caché**

```
Setup (primera vez): ~60s (extracción siempre)
Setup (sesión 2):     ~60s (extracción siempre)
Setup (sesión 3):     ~60s (extracción siempre)

Total 3 sesiones: ~180s
```

### **Con caché**

```
Setup 1: ~60s (extracción + guardado)
Setup 2: ~0.5s (CACHE HIT)
Setup 3: ~0.5s (CACHE HIT, si pesos no cambiaron)

Total 3 sesiones: ~61s
───────────────────────────────────
Mejora: 180s → 61s = 3x más rápido
```

**Observación**: El caché es útil para validación/demostración rápida.

---

## 🔍 Validación de caché

### **Checks implementados**

```python
# [1] Archivo existe
if not os.path.exists(cache_X_path):
    → CACHE MISS

# [2] Carga sin error (no corrupto)
try:
    X_feat = np.load(cache_X_path, allow_pickle=False)
except:
    → CACHE CORRUPT

# [3] Shape exacto
expected_n = len(X)          # 50000
expected_dim = 512
if X_feat.shape != (expected_n, expected_dim):
    → SHAPE INVALID
```

### **Escenarios de corrupción**

| Escenario | Efecto | Manejo |
|-----------|--------|--------|
| Discodiscadenado mid-write | Archivo incompleto | load() falla → re-extract |
| Modificación manual de .npy | Datos basura | load() siempre falla → re-extract |
| Cambio X input shape | Shape no coincide | Shape check → re-extract |
| numpy versión diferente | Lectura falla | load() maneja automático |

---

## 📝 Logging del caché

### **Mensajes tipificados**

```
[CACHE HIT][TRAIN]     → Features cargados instantáneamente
[CACHE HIT][TEST]      → Features de prueba desde caché
[CACHE MISS][TRAIN]    → Extrayendo features…
[CACHE MISS][TEST]     → Extrayendo features de test…
[CACHE CORRUPT][TRAIN] → Error leyendo caché, regenerando…
[CACHE CORRUPT][TEST]  → Error leyendo caché, regenerando…
[SHAPE INVALID][TRAIN] → Shape mismatch, regenerando…
[SHAPE INVALID][TEST]  → Shape mismatch, regenerando…
[CACHE SAVE][TRAIN]    → Features guardados en caché (45.2s)
[CACHE SAVE][TEST]     → Features de test guardados en caché (5.1s)
[CACHE SAVE ERROR]     → No se guardó caché (pero continuando…)
```

---

## 🎯 Invalidación de caché

### **Cuándo se invalida**

```
Razón                               Acción
─────────────────────────────────────────────────────────
CNN arch cambió (simple → resnet)   Hash diferente → new cache dir
CNN weights actualizados            Hash diferente → new cache dir
X_raw shape cambió                  Shape validation falla → re-extract
Y_raw cambió                        [No validado] → confusión potencial
split cambió (train ↔ test)         Clave diferente → separate files
```

### **Invariantes**

```
✓ Si hash(CNN_weights[t1]) == hash(CNN_weights[t2])
  Entonces features[t1] == features[t2]
  (determinístico, reproducible)

✓ Si hash != hash'
  Entonces features ≠ features'
  (o muy improbable, p < 10^-8)

✓ Una vez guardado, no se modifica
  (append-only de facto)
```

---

## 🚀 Optimizaciones futuras

### **1. Incremental caching**

```python
# En lugar de guardar 50000 imganes de golpe,
# guardar por batches:

batch_size = 5000
for batch_idx in range(0, 50000, batch_size):
    X_batch_feat = extract_batch(batch_idx, batch_idx + batch_size)
    np.save(f"{cache_key}_batch_{batch_idx}.npy", X_batch_feat)

# Ventaja: si falla mid-extract, reutilizar lo guardado
# Desventaja: más archivos, I/O más granular
```

### **2. Compresión con .npz**

```python
# .npz = zip de múltiples .npy

# Actual:
np.save("train_X.npy", X_feat)  # 200 MB
np.save("train_Y.npy", Y)       # 0.2 MB

# Optimizado:
np.savez("train.npz", X=X_feat, Y=Y)  # 200 MB (sin compresión)
np.savez_compressed("train.npz", X=X_feat, Y=Y)  # 50 MB (con zlib)
```

### **3. Caché distribuido**

```
En lugar de caché local de cada Worker,
usar caché compartido (NFS, S3):

   Workers (GPU farms)
        ↓
    read/write → Shared Storage (NFS)
                        ↓
                  cached features

Ventaja: un Worker extrae, otros reutilizan inmediatamente
Desventaja: latencia red, colisiones de I/O
```

---

## 📣 Cuándo deshabilitar caché

```python
# En modo E2E, quizá NOT guarde caché después de cada época
# porque los pesos cambian:

if training_mode == "end_to_end":
    # CNN pesos varían en cada época
    # Caché sería inútil (nunca se reutiliza)
    save_to_cache = False
else:
    # Precomputed: CNN fija
    # Caché amortiza el costo inicial
    save_to_cache = True
```

---

**Documento**: `docs/07_caching_system.md`  
**Última actualización**: 2026-03-27  
**Nivel**: Avanzado
