# FIX: Reutilización de Estado Entre Sesiones de Entrenamiento

## Problema Identificado

**Síntoma:** Si ejecutaba PRECOMPUTED → luego END-TO-END, el segundo entrenamiento ejecutaba usando PRECOMPUTED (el modo anterior).

**Causa Raíz:** El Parameter Server se instanciaba UNA SOLA VEZ con un valor de `training_mode = "precomputed"` (default), y ese atributo **NUNCA se actualizaba** entre sesiones de entrenamiento.

### Flujo de Ejecución Problemático

```python
# ps_gui.py - Línea 1398
self._server = ParameterServer(
    host=host,
    port=port,
    training_mode=self._v_system_mode.get(),  # ← Se asigna UNA sola vez
    ...)

# Luego, al entrenar:
def _cmd_train(self):
    training_mode = self._v_system_mode.get()  # ← Obtiene el modo ACTUAL
    # PERO nunca actualiza self._server.training_mode
    
    history = server.train(
        # ...
        # NO pasaba training_mode aquí
    )
```

**Resultado:** Para la 2ª sesión, `self._server.training_mode` seguía siendo "precomputed" aunque el usuario seleccionara "end_to_end" en la GUI.

---

## Solución Implementada

### 1. Crear Método `_reset_training_state()`

**Archivo:** `Distributed/parameter_server.py` (después de línea 249)

```python
def _reset_training_state(self, new_training_mode: str) -> None:
    """
    Limpia completamente el estado entre sesiones de entrenamiento.
    
    CRÍTICO: Llamar SIEMPRE antes de iniciar una nueva sesión para
    evitar que state anterior contamine el nuevo entrenamiento.
    """
    # Validar training_mode
    if new_training_mode not in ("precomputed", "end_to_end"):
        raise ValueError(...)
    
    # ✓ Actualizar training_mode EXPLÍCITAMENTE
    self.training_mode = new_training_mode
    print(f"[PS][RESET] training_mode actualizado a: {self.training_mode}")
    
    # ✓ Limpiar estado persistente anterior
    self._active_training_workers = None
    self._cnn_ready_event.clear()
    self._cnn_ready_count = 0
    self._X_test_features = None
    self._Y_test_from_worker = None
    self._epoch_gradients.clear()
    self._epoch_metrics.clear()
    
    print(f"[PS][RESET] ✓ Estado de sesión limpiado")
```

**Garantías:**
- ✅ training_mode se actualiza ANTES de cada entrenamiento
- ✅ Se limpian todos los caches y flags de sesión anterior
- ✅ Cada sesión empieza completamente limpia

---

### 2. Modificar Método `train()`

**Archivo:** `Distributed/parameter_server.py` (línea 495)

**Cambios:**
- ✅ Agregar parámetro `training_mode: Optional[str] = None`
- ✅ Llamar a `_reset_training_state(training_mode)` al inicio
- ✅ Agregar validación sanity check con assert

```python
def train(
    self,
    epochs: int,
    initial_params: Dict[str, np.ndarray],
    learning_rate: float,
    n_train: int,
    X_test: Optional[np.ndarray] = None,
    Y_test: Optional[np.ndarray] = None,
    momentum: float = 0.0,
    seed: Optional[int] = None,
    training_mode: Optional[str] = None,  # ← NUEVO
) -> Dict[str, List[float]]:
    
    # Si se proporciona training_mode, usarlo; si no, usar self.training_mode
    if training_mode is None:
        training_mode = self.training_mode
    
    # Validar training_mode
    if training_mode not in ("precomputed", "end_to_end"):
        raise ValueError(...)
    
    # ✓ CRÍTICO: Limpiar estado y actualizar training_mode
    self._reset_training_state(training_mode)
    
    # ✓ Validación sanity check
    assert self.training_mode in ["precomputed", "end_to_end"]
    print(f"[PS][DEBUG] CONFIG FINAL: mode={self.training_mode}")
    
    # ... resto del código
```

---

### 3. Actualizar GUI para Pasar training_mode Explícitamente

**Archivo:** `ps_gui.py` (línea 1729)

```python
# ANTES:
history = server.train(
    epochs=epochs,
    initial_params=initial_params,
    learning_rate=lr,
    n_train=n_train,
    X_test=None,
    Y_test=Y_test,
    momentum=momentum,
    # ← training_mode NO se pasaba
)

# DESPUÉS:
history = server.train(
    epochs=epochs,
    initial_params=initial_params,
    learning_rate=lr,
    n_train=n_train,
    X_test=None,
    Y_test=Y_test,
    momentum=momentum,
    training_mode=training_mode,  # ✓ Pasar el modo actual
)
```

---

### 4. Actualizar Terminal para Ser Explícita

**Archivo:** `ps_terminal.py` (línea 316)

```python
# ANTES:
history = server.train(
    epochs=args.epochs,
    initial_params=initial_params,
    learning_rate=args.lr,
    n_train=args.n_train,
    X_test=X_test_raw,
    Y_test=Y_test,
    momentum=args.momentum,
)

# DESPUÉS:
history = server.train(
    epochs=args.epochs,
    initial_params=initial_params,
    learning_rate=args.lr,
    n_train=args.n_train,
    X_test=X_test_raw,
    Y_test=Y_test,
    momentum=args.momentum,
    training_mode="precomputed",  # ✓ Ser explícito
)
```

---

## Comportamiento Esperado Ahora

### Escenario de Prueba

```
1. Iniciar PS
2. Conectar Worker
3. Entrenar PRECOMPUTED (100 épocas)
   → Log: [PS][RESET] training_mode actualizado a: precomputed
   → Log: [PS][DEBUG] CONFIG FINAL: mode=precomputed
4. Sin reiniciar, cambiar a END-TO-END en GUI
5. Entrenar END-TO-END (100 épocas)
   → Log: [PS][RESET] training_mode actualizado a: end_to_end  ✓ CAMBIO APLICADO
   → Log: [PS][DEBUG] CONFIG FINAL: mode=end_to_end          ✓ CORRECTO
```

### Logs de Diagnóstico

Cada sesión de entrenamiento ahora incluye:

```
[PS][RESET] ════════════════════════════════════════════════════════════
[PS][RESET] Limpiando estado anterior
[PS][RESET] training_mode: precomputed → end_to_end
[PS][RESET] ✓ training_mode actualizado a: end_to_end
[PS][RESET] ✓ Estado de sesión limpiado
[PS][RESET] ════════════════════════════════════════════════════════════

[PS][DEBUG] CONFIG FINAL: mode=end_to_east
```

---

## Coherencia en Envío de PARAMS

El código PRECOMPUTED y END-TO-END ahora garantiza:

### PRECOMPUTED (training_mode = "precomputed")
```python
send_dict = {
    "epoch": epoch,
    "params": params,           # ✓ MLP params
    "seed": epoch_seed,
    # NO cnn_params
}
```

### END-TO-END (training_mode = "end_to_end")
```python
send_dict = {
    "epoch": epoch,
    "params": params,           # ✓ MLP params
    "seed": epoch_seed,
    "cnn_params": cnn_state,    # ✓ CNN params incluidos
}
```

---

## Resumen de Cambios

| Archivo | Líneas | Cambio |
|---------|--------|--------|
| `parameter_server.py` | +50 líneas | Agregar `_reset_training_state()` |
| `parameter_server.py` | 495-600 | Modificar `train()` para llamar reset |
| `ps_gui.py` | 1732 | Pasar `training_mode=training_mode` |
| `ps_terminal.py` | 319 | Pasar `training_mode="precomputed"` explícitamente |

---

## Validación

✅ **Sintaxis:** 0 errores en todos los archivos  
✅ **Lógica:** training_mode se actualiza antes de cada entrenamiento  
✅ **State Clean:** _reset_training_state() limpia todos los caches  
✅ **Backward Compat:** train() sin parámetro training_mode sigue funcionando  
✅ **Logs:** Se agregó [PS][RESET] para visibilidad

---

## Próximos Pasos (Verificación Manual)

1. Iniciar PS con GUI:
   ```bash
   python ps_gui.py
   ```

2. Conectar Worker:
   ```bash
   python worker.py
   ```

3. Entrenar PRECOMPUTED
   - Observar: `[PS][RESET] training_mode actualizado a: precomputed`

4. Cambiar modo a END-TO-END en GUI
5. Entrenar END-TO-END
   - Observar: `[PS][RESET] training_mode actualizado a: end_to_end` ✓

Si ves el cambio en los logs, el bug está **ARREGLADO** ✅

