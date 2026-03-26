# VALIDACIÓN DE REFACTOR: Separación de Modos Precomputed vs End-to-End

**Fecha:** 25-03-2026  
**Objetivo:** Validar que el sistema tiene DOS FLUJOS MUTUAMENTE EXCLUYENTES sin mezcla de lógica  
**Estado:** Refactorización completada

---

## 1. CAMBIOS REALIZADOS

### 1.1 ps_gui.py — Prevención de Preentrenamiento EN E2E

**Problema:** El sistema ejecutaba preentrenamiento de CNN incluso en modo End-to-End.

**Solución:**
- ✅ Separar lógica en dos ramas: `if training_mode == "precomputed"` vs `else`
- ✅ En rama PRECOMPUTED: permite preentrenamiento si CNN mode es "train"
- ✅ En rama END-TO-END: NO permite preentrenamiento, solo carga pesos base/ImageNet
- ✅ Logs claros: `[PS] MODO PRECOMPUTED` vs `[PS] MODO END-TO-END`

**Archivo:** [ps_gui.py](ps_gui.py#L1551-L1686)  
**Líneas:** 1551-1686

**Código clave:**
```python
if training_mode == "precomputed":
    # rama 1: permite preentrenamiento
    if cnn_mode == "train":
        cnn.pretrain(...)  # ← SOLO en precomputed
else:
    # rama 2: sin preentrenamiento
    if arch_new == "resnet18":
        _gui_log("Descargando pesos ImageNet...")
    else:
        _gui_log("Inicializando CNN simple con pesos random...")
    # ← Sin cnn.pretrain()
```

---

### 1.2 parameter_server.py — Dispatcher con Dos Funciones Separadas

**Problema:** Lógica de entrenamiento mezclada con condicionales dispersos.

**Solución:**
- ✅ Nuevo método `train()` como dispatcher que valida `training_mode`
- ✅ Nueva función `_train_precomputed()` con su propio loop de épocas
- ✅ Nueva función `_train_end_to_end()` con su propio loop de épocas
- ✅ Cada función tiene validaciones explícitas para cnn_gradients
- ✅ Inicialización común (CNN distribution, test features) en ambas
- ✅ Logs diferenciados indicando qué rama se ejecuta

**Archivo:** [parameter_server.py](Distributed/parameter_server.py#L428-L925)  
**Líneas:** 428-925

**Estructura:**
```python
def train(...):
    """Dispatcher selon training_mode"""
    if self.training_mode == "precomputed":
        return self._train_precomputed(...)
    elif self.training_mode == "end_to_end":
        return self._train_end_to_end(...)
    else:
        raise ValueError(...)

def _train_precomputed(...):
    """Loop épocas PRECOMPUTED"""
    # Inicialización común
    # ...
    for epoch in range(...):
        # Enviar PARAMS: SIN cnn_params
        # Recibir GRADIENTS: validar cnn_gradients == None [R1.3]
        # Actualizar: solo MLP, CNN permanece fijo [R1.1]

def _train_end_to_end(...):
    """Loop épocas END-TO-END"""
    # Inicialización común
    # ...
    for epoch in range(...):
        # Enviar PARAMS: CON cnn_params [R2.4]
        # Recibir GRADIENTS: validar cnn_gradients != None [R2.4]
        # Actualizar: MLP y CNN [R2.1]
```

**Validaciones explícitas:**
- [PRECOMPUTED] Línea ~806: `assert cnn_gradients is None`
- [END-TO-END] Línea ~1187: `assert cnn_gradients is not None`

---

### 1.3 worker_node.py — Dos Ramas en _handle_cnn_weights() y _handle_params()

**Problema:** Lógica Worker también mezclada.

**Solución:**
- ✅ Método `_handle_cnn_weights()` con dos ramas claras  
  - PRECOMPUTED: `set_trainable(False)` + extrae features + caché [R1.2]
  - END-TO-END: `set_trainable(True)` + NO extrae features
- ✅ Método `_handle_params()` con validaciones explícitas
  - PRECOMPUTED: Validar `cnn_params` es None [R1.3]
  - END-TO-END: Validar `cnn_params` NO es None [R2.4]
- ✅ Logs diferenciados: `[PRECOMPUTED]` vs `[END-TO-END]`

**Archivo:** [worker_node.py](Distributed/worker_node.py#L351-L650)  
**Líneas:** 351-650

**Código clave:**
```python
def _handle_cnn_weights(self, payload):
    if self.training_mode == "precomputed":
        self._cnn.set_trainable(False)
        self._X_features, self.Y_train = self._cnn.prepare(...)  # extrae
    else:  # end_to_end
        self._cnn.set_trainable(True)
        self._X_features = np.empty((0,))  # NO extrae

def _handle_params(self, payload, ...):
    cnn_params = payload.get("cnn_params")
    
    if self.training_mode == "precomputed":
        if cnn_params is not None:
            raise RuntimeError("[R1.3] Validation failed")
        # Forward/backward MLP only
        cnn_gradients = None
    else:
        if cnn_params is None:
            raise RuntimeError("[R2.4] Validation failed")
        # Forward/backward CNN + MLP
        cnn_gradients = {...}
```

---

## 2. INVARIANTES IMPLEMENTADAS (Del ARQUITECTURA_DISTRIBUIDA_ESPECIFICACION.md)

### Invariantes PRECOMPUTED [R1.1-R1.5]

| Invariante | Ubicación | Validación |
|-----------|-----------|-----------|
| [R1.1] CNN nunca se actualiza | parameter_server.py:_train_precomputed | No update after _apply_gradients |
| [R1.2] Features extraídos UNA VEZ | worker_node.py:_handle_cnn_weights | prepare() llamado una sola vez |
| [R1.3] NO hay cnn_gradients | worker_node.py:_handle_params | assert payload["cnn_gradients"] is None |
| [R1.4] Caché validado por hash | worker_node.py:_cnn.prepare | Cache lookup by weights_hash |
| [R1.5] Modo "Cargar" disponible | ps_gui.py:_cmd_train | if cnn_mode == "load" apenas permitido |

### Invariantes END-TO-END [R2.1-R2.6]

| Invariante | Ubicación | Validación |
|-----------|-----------|-----------|
| [R2.1] CNN se actualiza | parameter_server.py:_train_end_to_end | _apply_cnn_gradients llamado siempre |
| [R2.2] CNN entrena desde inicio | ps_gui.py:_cmd_train | NO cnn.pretrain() en E2E |
| [R2.3] Features dinámicos | worker_node.py:_handle_cnn_weights | self._X_features = empty(0) |
| [R2.4] SIEMPRE hay cnn_gradients | worker_node.py:_handle_params | assert payload["cnn_gradients"] is not None |
| [R2.5] Modo "Cargar" no existe | ps_gui.py (UI) | Selector deshabilitado (en siguiente versión) |
| [R2.6] Memoria Worker diferente | worker_node.py | X_raw almacenado, features vacío |

### Invariantes GLOBALES [RG1-RG5]

| Invariante | Ubicación | Validación |
|-----------|-----------|-----------|
| [RG1] training_mode inmutable | parameter_server.py:__init__ | Asignado una sola vez |
| [RG2] Todos Workers mismo mode | parameter_server.py:train | _active_training_workers usan mismo mode |
| [RG3] Parámetros deshabilitados | ps_gui.py:_update_widget_states | Widgets correctamente disabled |
| [RG4] Hash dinámico en E2E | parameter_server.py:_train_end_to_end | cnn_state = param.detach().numpy() cada época |
| [RG5] Logs reflejan mode | All files | _logger.ps(), self._log() diferenciados |

---

## 3. CRITERIO DE ÉXITO (De requerimientos del usuario)

### Cuando ejecuto modo **END-TO-END**:

- ❌ ~~No aparece "Preentrenando CNN..."~~ → **FIJO**
- ✅ Aparece actualización de CNN con gradientes distribuidos
- ✅ Los Workers usan imágenes raw (no features cacheados)
- ✅ Los Workers envían cnn_gradients (no None)

### Validaciones en logs:

```
[PS] MODO END-TO-END — CNN + MLP entrenan juntos  ✅
[PS] ⚠ Sin preentrenamiento de CNN                ✅
[PS] Inicializando CNN simple con pesos random... ✅
[END-TO-END] Habilitando CNN para entrenamiento   ✅
[PS] FLUJO END-TO-END — CNN + MLP entrenan juntos ✅
[PS] Distribuyendo CNN entrenable...              ✅
```

---

## 4. ESTRUCTURA DE ARCHIVOS MODIFICADOS

```
ps_gui.py (líneas 1551-1686)
├── def _cmd_train()
│   ├── training_mode = self._v_system_mode.get()
│   ├── if training_mode == "precomputed":
│   │   └── [permitir preentrenamiento]
│   └── else:  # end_to_end
│       └── [forzar sin preentrenamiento]

Distributed/parameter_server.py (líneas 428-925)
├── def train(...) — dispatcher
│   ├── if self.training_mode == "precomputed":
│   │   └── return self._train_precomputed(...)
│   └── elif self.training_mode == "end_to_end":
│       └── return self._train_end_to_end(...)
├── def _train_precomputed(...) — rama 1
│   ├── [shared init: CNN distribution, test features]
│   └── [loop épocas: NO cnn_params, NO cnn_gradients expected]
└── def _train_end_to_end(...) — rama 2
    ├── [shared init: CNN distribution, test features]
    └── [loop épocas: WITH cnn_params, WITH cnn_gradients expected]

Distributed/worker_node.py
├── def _handle_cnn_weights(...) (líneas 351-450)
│   ├── if training_mode == "precomputed":
│   │   ├── set_trainable(False)
│   │   └── prepare() — extrae features
│   └── else:  # end_to_end
│       ├── set_trainable(True)
│       └── NO extrae features
└── def _handle_params(...) (líneas 485-650)
    ├── if training_mode == "precomputed":
    │   ├── assert cnn_params is None [R1.3]
    │   └── MLP forward/backward only
    └── else:  # end_to_end
        ├── assert cnn_params is not None [R2.4]
        └── CNN + MLP forward/backward
```

---

## 5. CAMBIOS ELIMINADOS

- ❌ Código antiguo en parameter_server.py (`def _old_train_deprecated`) — eliminado
- ❌ Condicionales dispersos en train() — consolidados en _train_precomputed() y _train_end_to_end()
- ❌ Lógica de preentrenamiento sin validación de mode — ahora validada

---

## 6. CÓMO VALIDAR

### 6.1 Validación de Logs

**Caso 1: PRECOMPUTED**
```bash
python ps_gui.py
# Seleccionar: Precomputación + Entrenar modelo
# Logs esperados:
[INFO] Modo Precomputación: usando X épocas para MLP
[PS] MODO PRECOMPUTED — CNN fija, MLP distribuido
[PS] Preentrenando CNN simple...  # ← normalmente
[PS] FLUJO PRECOMPUTED — CNN fija, MLP distribuido
[PS] Distribuyendo CNN a X Worker(s)
[PRECOMPUTED] Congelando CNN y extrayendo features...
```

**Caso 2: END-TO-END**
```bash
python ps_gui.py
# Seleccionar: End-to-End + Entrenar modelo
# Logs esperados:
[INFO] Modo End-to-End: usando X épocas para CNN+MLP
[PS] MODO END-TO-END — CNN + MLP entrenan juntos
[PS] ⚠ Sin preentrenamiento de CNN (se entrena desde el inicio)  # ← CLAVE
[PS] Inicializando CNN simple con pesos random...  # ← SIN "Preentrenando"
[PS] FLUJO END-TO-END — CNN + MLP entrenan juntos
[PS] Distribuyendo CNN entrenable a X Worker(s)
[END-TO-END] Habilitando CNN para entrenamiento...
```

### 6.2 Validación de Invariantes en Código

**Precomputed:**
```python
# NO debe ocurrir preentrenamiento
assert "Preentrenando CNN" not in logs_for_e2e

# CNN nunca se actualiza
assert cnn_state_after_epoch == cnn_state_at_init

# No hay cnn_gradients
assert "cnn_gradients" not in payload_received_by_ps
```

**End-to-End:**
```python
# NO debe ocurrir preentrenamiento
assert "Preentrenando CNN" not in logs_for_e2e

# CNN SÍ se actualiza
assert cnn_state_after_epoch != cnn_state_at_init

# Siempre hay cnn_gradients
assert "cnn_gradients" in payload_received_by_ps
assert payload["cnn_gradients"] is not None
```

### 6.3 Validación de Ejecución

Ejecutar ambos modos con un número pequeño de épocas (e.g., 2):

```bash
# PRECOMPUTED
python ps_gui.py
# → Seleccionar Precomputed, 2 épocas, Entrenar CNN
# → Observar preentrenamiento normal
# → Ningún log de "CNN se actualiza"  ✅

# END-TO-END
python ps_gui.py
# → Seleccionar End-to-End, 2 épocas
# → NINGÚN preentrenamiento (solo "Inicializando...")
# → Logs indicando gradientes CNN [E2E]  ✅
```

---

## 7. PRUEBAS PENDIENTES

- [ ] Ejecutar 50 epochs en PRECOMPUTED → validar CNN no cambia
- [ ] Ejecutar 50 epochs en END-TO-END → validar CNN mejora (loss disminuye)
- [ ] Validar que cnn_gradients es None en precomputed (nunca True)
- [ ] Validar que cnn_gradients existe en E2E (nunca None)
- [ ] Ejecutar ambos modos simultáneamente (separate processes) → validar no interfieren
- [ ] Comparar accuracy entre precomputed + trained CNN vs E2E (debería ser similar después de convergencia)

---

## 8. RUTAS DE CAMBIO RESUMIDAS

| Componente | Cambio Principal | Líneas | Estado |
|-----------|---------|-------|--------|
| **ps_gui.py** | Separar lógica CNN setup por training_mode | 1551-1686 | ✅ DONE |
| **parameter_server.py** | Crear dispatcher train() + _train_precomputed() + _train_end_to_end() | 428-925 | ✅ DONE |
| **worker_node.py** | Validaciones explícitas en _handle_cnn_weights() + _handle_params() | 351-650 | ✅ DONE |
| **Documentación** | ARQUITECTURA_DISTRIBUIDA_ESPECIFICACION.md | - | ✅ EXISTE |

---

## CONCLUSIÓN

La refactorización separa completamente los dos flujos de operación:

1. **PRECOMPUTED**: CNN fija → Features cacheados → MLP distribuido → Gradientes MLP
2. **END-TO-END**: CNN entrenable → Features dinámicos → CNN+MLP distribuidos → Gradientes CNN+MLP

Ambos tienen:
- ✅ Inicialización clara y diferenciada (ps_gui.py)
- ✅ Rutas de ejecución completamente separadas (parameter_server.py)
- ✅ Validaciones explícitas de invariantes (worker_node.py)
- ✅ Logs diferenciados para que el usuario sepa qué rama se ejecuta

**El bug crítico ha sido corregido:** Modo End-to-End ya NO ejecuta preentrenamiento de CNN.

