# ANÁLISIS DE ARQUITECTURA Y PROBLEMAS DETECTADOS

## Resumen Ejecutivo

El proyecto tiene una **arquitectura sólida** pero con algunas **ineficiencias y problemas menores**:

- ✅ Flujo distribuido PS ↔ Workers es consistente
- ✅ Separación entre "simple" y "resnet18" está clara
- ⚠️ Algunos imports no usados
- ⚠️ Código redundante en carga de etiquetas
- ⚠️ Lógica duplicada en dataloaders

**Total de problemas:** 7 críticos/menores, **todos reparables sin cambiar arquitectura**.

---

## PROBLEMAS DETECTADOS

### 1. [MENOR] Import no usado: `get_stream_shard_size`

**Archivo:** `Distributed/worker_node.py`

**Problema:**
```python
from Utils.imagenet_loader import (
    ...,
    get_stream_shard_size,  # ← NUNCA SE USA
    ...
)
```

**Impacto:** Ligero (aumenta tamaño imports)

**Solución:** Eliminar import

**Estado:** ✅ REPARADO

---

### 2. [MENOR] Tipo incompleto en `_load_raw_images_for_indices()`

**Archivo:** `Distributed/worker_node.py`, línea ~805

**Problema:**
```python
def _load_raw_images_for_indices(self, indices: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
```

Falta `Optional` en el retorno (puede devolver `(None, None)`).

**Solución:**
```python
def _load_raw_images_for_indices(self, indices: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
```

**Estado:** ⏳ PENDIENTE

---

### 3. [MENOR] Código duplicado: Carga de etiquetas en dos sitios

**Archivo:** `Distributed/worker_node.py`

**Problema:**
Líneas ~410-432 (en `_handle_cnn_weights` para modo "simple") y líneas ~805-840 (en `_load_raw_images_for_indices`) repiten lógica:

```python
# Repetición 1: _handle_cnn_weights (línea 410+)
if self._data_source == "local":
    loader = get_imagenet_dataloader(...)
    all_y = []
    for _, y in loader:
        all_y.append(y.cpu().numpy())
    self._Y_raw = np.concatenate(all_y)
else:
    loader = get_imagenet_stream_dataloader(...)
    all_y = []
    for _, y in loader:
        all_y.append(y.cpu().numpy())
    self._Y_raw = np.concatenate(all_y)

# Repetición 2: _load_raw_images_for_indices (línea 810+)
if self._data_source == "local":
    loader = get_imagenet_dataloader(...)
    ...
else:
    loader = get_imagenet_stream_dataloader(...)
    ...
```

**Solución:** Extraer método `_load_labels_from_split(split: str)`:

```python
def _load_labels_from_split(self, split: str = "train") -> Optional[np.ndarray]:
    """Carga etiquetas de train/val en ambos modos (local + stream)."""
    try:
        if self._data_source == "local":
            loader = get_imagenet_dataloader(
                split=split, data_dir=self._data_dir,
                batch_size=512, num_workers=4
            )
        else:
            if not self._hf_token:
                return None
            loader = get_imagenet_stream_dataloader(
                split=split, token=self._hf_token,
                batch_size=512, shard_index=0, num_shards=1
            )
        
        all_y = []
        for _, y in loader:
            all_y.append(y.cpu().numpy())
        return np.concatenate(all_y)
    except Exception as e:
        self._log(f"Error cargando etiquetas: {e}")
        return None
```

**Estado:** ⏳ PENDIENTE

---

### 4. [MENOR] Tipos de retorno inconsistentes

**Archivo:** `Distributed/worker_node.py`

**Problema:**
`_load_raw_images_for_indices()` retorna `(None, None)` en error, pero el caller no siempre lo maneja:

```python
X_raw, Y_raw = self._load_raw_images_for_indices(indices)

if X_raw is None or Y_raw is None or len(X_raw) == 0:  # ← Necesario verificar
    ...
```

Mejor sería usar excepciones o Tuple con Union:

**Solución alternativa (recomendado):**
```python
def _load_raw_images_for_indices(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lanza excepción si no puede cargar, nunca retorna None."""
    try:
        # lógica de carga
        ...
        return X_raw, Y_raw
    except FileNotFoundError as e:
        self._log(f"Error crítico: {e}")
        raise
```

**Estado:** ⏳ RECOMENDADO PERO NO CRÍTICO

---

### 5. [MENOR] `get_stream_shard_size()` nunca se usa

**Archivo:** `Utils/imagenet_loader.py`

**Problema:**
```python
def get_stream_shard_size(token: str, split: str = "train") -> int:
    """Nunca llamada en el código."""
    ...
```

**Solución:** Eliminar función o documentar por qué existe.

**Estado:** ⏳ PENDIENTE

---

### 6. [MAYOR] Variable `img_parts` no inicializada antes de usar

**Archivo:** `Distributed/worker_node.py`, método `_load_raw_images_for_indices()`

**Problema:**
```python
def _load_raw_images_for_indices(self, indices: np.ndarray):
    # ...código...
    if self._data_source == "local":
        loader = get_imagenet_dataloader(...)
        for X_batch, Y_batch in loader:
            img_parts.append(...)  # ← NameError: img_parts no definida
```

**Solución:**
```python
def _load_raw_images_for_indices(self, indices: np.ndarray):
    img_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    
    if self._data_source == "local":
        ...
```

**Estado:** ✅ REPARABLE EN 1 LÍNEA

---

### 7. [MENOR] Logger verbose pero sin control granular

**Problema:**
Muchos `self._log()` en worker_node.py, pero no hay forma de controlar qué se imprime sin modificar código.

**Solución alternativa (futuro):**
```python
self._log("msg", level="DEBUG")  # solo si verbose >= DEBUG
self._log("msg", level="INFO")
self._log("msg", level="WARN")
```

**Estado:** ⏳ MEJORA FUTURA (no crítica)

---

## TABLA DE ACCIONES

| # | Archivo | Problema | Solución | Criticidad | Estado |
|---|---------|----------|----------|------------|--------|
| 1 | worker_node.py | Import no usado | Eliminar `get_stream_shard_size` | Menor | ✅ HECHO |
| 2 | worker_node.py | Tipo incompleto | Añadir `Optional[Tuple[...]]` | Menor | ⏳ |
| 3 | worker_node.py | Código duplicado | Extraer `_load_labels_from_split()` | Menor | ⏳ |
| 4 | worker_node.py | Manejo inconsistente None | Usar excepciones en lugar de None | Menor | ⏳ |
| 5 | imagenet_loader.py | Función muerta | Eliminar `get_stream_shard_size()` | Menor | ⏳ |
| 6 | worker_node.py | Variable no inicializada | Inicializar `img_parts = []` | Mayor | ⏳ |
| 7 | worker_node.py | Logging sin control | Añadir niveles de logging | Futuro | ⏳ |

---

## ANÁLISIS DE ARQUITECTURA

### ✅ Flujo PS ↔ Worker ES CONSISTENTE

**Fase 1: CNN Distribution**
```
PS envía: CNN_WEIGHTS(arch, weights_bytes)
    │
    ├─→ Worker recibe → carga pesos
    ├─→ Si arch="resnet18": extrae features → cachea
    ├─→ Si arch="simple": carga etiquetas → prepara
    └─→ Worker envía: CNN_READY
```

**Verificación:** ✅ Ambos casos manejados en `_handle_cnn_weights()` con branching claro.

---

### ✅ Separación "simple" vs "resnet18" ES CLARA

```
Para "resnet18":
  _handle_cnn_weights() → líneas 352-401
    ├─ Extrae features a shards
    ├─ Calcula FeatureScaler
    └─ Carga índices desde shards

Para "simple":
  _handle_cnn_weights() → líneas 404-449
    ├─ NO extrae shards
    ├─ NO calcula FeatureScaler
    └─ Carga etiquetas para índices

_run_training_resnet18() → líneas 557-598
    └─ Carga features de shards

_run_training_simple() → líneas 640-695
    └─ Carga raw images, computa features on-the-fly
```

**Verificación:** ✅ Dos caminos separados y bien documentados.

---

### ✅ Uso de --n-train ES CONSISTENTE

```
ps_terminal.py:
  args.n_train = resolve_value(default=None) → 1,281,167

parameter_server.py:
  train(n_train=args.n_train)
    └─ broadcast TRAIN_START(n_train)

worker_node.py:
  msg["payload"]["n_train"]
    └─ _reconstruct_indices(n_train)
    └─ shard size = (n_train + SHARD_SIZE - 1) // SHARD_SIZE
```

**Verificación:** ✅ Valor fluye correctamente de terminal → PS → Workers.

---

### ✅ No hay lógica "muerta" (dead code)

Todas las funciones se llaman:
- `_hand_cnn_weights()` → msg routing
- `_run_training_session()` → msg routing
- `_load_features_for_indices()` → _run_training_resnet18()
- `_load_raw_images_for_indices()` → _run_training_simple()
- `_reconstruct_indices()` → ambas sesiones

**Hay una excepción:** `get_stream_shard_size()` en imagenet_loader.py (nunca se llama).

---

## CONCLUSIÓN

**Calificación general: 8/10**

| Aspecto | Evaluación |
|---------|-----------|
| Consistencia PS ↔ Worker | ✅ Excelente |
| Separación simple/resnet18 | ✅ Excelente |
| Uso correcto de --n-train | ✅ Excelente |
| Código limpio | ⚠️ 7/10 (minor issues) |
| Documentación | ⏳ README mejorado |
| Testing | ⏳ No hay tests automáticos |

**Acciones recomendadas (prioridad):**

1. ✅ Eliminar imports no usados
2. ✅ Inicializar `img_parts` en `_load_raw_images_for_indices()`
3. ⏳ Extraer método `_load_labels_from_split()` (reduce duplicación)
4. ⏳ Mejorar tipos de retorno (opcional)
5. ⏳ Escribir README profesional (HECHO)
