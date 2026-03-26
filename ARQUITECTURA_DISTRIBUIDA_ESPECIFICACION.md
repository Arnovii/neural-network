# ESPECIFICACIÓN: Arquitectura Distribuida — Modos Precomputed vs End-to-End

## Versión: 1.0
**Última actualización:** 25-03-2026  
**Estado:** Definición Técnica (sin implementación)  
**Propósito:** Guía clara para refactorización correcta del sistema

---

## 1. DEFINICIÓN CONCEPTUAL

### 1.1 Modo PRECOMPUTED

**Concepto principal:**
La CNN es un **extractor de características fijo**. El MLP es el único modelo que se entrena de forma distribuida.

**Características del modo:**
- La CNN **nunca se actualiza** durante el entrenamiento distribuido
- Los pesos CNN son idénticos en todos los Workers
- Los features son **precalculados una sola vez** por Worker (en RAM)
- El MLP es el único componente que se entrena con Algoritmo de Diego

**Flujo conceptual:**
```
Imágenes raw
    ↓
CNN (pesos FIJOS del PS)
    ↓
Features extraídos (almacenados en RAM del Worker)
    ↓
MLP (único que se entrena, pesos distribuidos por PS)
    ↓
Gradientes del MLP → PS
```

**Origen de la CNN:**
- Puede ser **cargada desde disco** (modo "Cargar modelo")
- Puede ser **preentrenada localmente** antes de la sesión distribuida (modo "Entrenar modelo")
- El PS la distribuye a los Workers

**Validez de este modo:**
- Válido tanto en `mode=train` (preentrenar) como en `mode=load` (cargar)
- La sesión distribuida solo comienza después de que la CNN esté fija

---

### 1.2 Modo END-TO-END

**Concepto principal:**
La CNN y el MLP forman un **único modelo entrenado conjuntamente**. No existe extracción de features fija.

**Características del modo:**
- La CNN **se actualiza durante el entrenamiento distribuido**
- Los gradientes fluyen desde el MLP hacia el MLP, y luego hacia la CNN
- Los features se **calculan dinámicamente** en cada época
- No existe caché de features — se usa imagen raw

**Flujo conceptual:**
```
Imágenes raw (cada época)
    ↓
CNN (pesos ENTRENABLE, actualizados cada época)
    ↓
Features dinámicos (no cacheados)
    ↓
MLP (también entrenable)
    ↓
Gradientes CNN + Gradientes MLP → PS
```

**Origen de la CNN:**
- La CNN comienza con ciertos pesos (random o ImageNet)
- **No existe preentrenamiento separado** — se entrena de extremo a extremo desde el inicio
- El PS distribuye la CNN inicial, luego la actualiza de forma centralizada

**Restricción:** Mode debe ser "Entrenar modelo" (no tiene sentido "Cargar modelo" sin features fijos)

---

## 2. FLUJO COMPLETO DEL ENTRENAMIENTO

### 2.1 Fase de Inicialización

#### PRECOMPUTED

**PS (Parameter Server):**
1. Se inicia el PS con `training_mode="precomputed"`
2. Se configura la CNN (parámetro: `cnn` o `arch`)
3. PS.set_cnn(cnn) — establece la CNN que se distribuirá
4. Internamente:
   - Si la CNN es `arch="simple"`: opcionalmente preentrenada localmente
   - Los pesos se guardan
   - El hash de pesos se calcula para el caché de features

**Worker (cuando arranca):**
1. Se conecta al PS, recibe su ID
2. Entra en bucle de espera (no hace nada hasta TRAIN_START)
3. Lleva CIFAR-10 sin procesar: X_raw (50000, 3, 32, 32), Y_raw

**PS → Worker (CNN_WEIGHTS):**
1. PS envía al Worker: arch + pesos CNN
2. Worker recibe y reconstruye la CNN con esos pesos
3. Worker congelará la CNN: `cnn.set_trainable(False)`

**Worker (extracción de features):**
1. Extrae los 50000 features de una sola vez
   - Input: X_raw (50000, 3, 32, 32)
   - CNN congelada
   - Output: cached X_features (50000, FEATURE_DIM=512)
2. Guarda en RAM para todas las épocas
3. Envía CNN_READY al PS

**Datos de prueba (si existen):**
- PS pide a un Worker que extraiga features de prueba
- Worker envía: X_test_features + Y_test
- PS almacena para evaluación global

---

#### END-TO-END

**PS (Parameter Server):**
1. Se inicia el PS con `training_mode="end_to_end"`
2. Se configura la CNN (parámetro: `cnn` o `arch`)
3. PS.set_cnn(cnn) — establece la CNN inicial
4. Internamente:
   - CNN no se preentrenará — se distribuye tal cual
   - Hash de pesos se calcula

**Worker (cuando arranca):**
1. Se conecta al PS, recibe su ID
2. Entra en bucle de espera
3. Lleva CIFAR-10 sin procesar: X_raw (50000, 3, 32, 32), Y_raw

**PS → Worker (CNN_WEIGHTS):**
1. PS envía: arch + pesos CNN inicial
2. Worker recibe y reconstruye la CNN con esos pesos
3. Worker habilitará la CNN: `cnn.set_trainable(True)`

**Worker (preparación):**
1. **NO extrae features de toda la imagen**
2. Guarda X_raw tal cual en memoria (sin procesar)
3. Prepara índices por clase estratificados
4. Envía CNN_READY al PS

**Diferencia clave:**
- NO hay extracción inicial de features
- Cada época, el forward pasa X_raw a través de CNN

---

### 2.2 Flujo de Entrenamiento (por época)

#### PRECOMPUTED

**PS envía (PARAMS):**
```python
{
    "epoch": epoch,
    "params": params_mlp,       # W1, b1, W2, b2, W3, b3
    "seed": rng_seed,           # para partición estratificada
    "cnn_params": None          # No existe en precomputed
}
```

**Worker recibe (PARAMS) y computa:**
1. Reconstruye índices locales using `seed` + `n_workers` + `worker_rank`
2. Obtiene training set partition
3. Extrae features del batch:
   ```
   X_batch = X_features[indices]      # ← De caché
   Y_batch = Y_train[indices]
   ```
4. Forward solo MLP:
   ```cpp
   F_batch (features) → MLP forward → gradients_mlp
   cnn_gradients = None
   ```
5. Calcula:
   - gradients_mlp
   - loss
   - accuracy
6. Envía GRADIENTS:
   ```python
   {
       "epoch": epoch,
       "gradients": gradients_mlp,     # Gradientes solo MLP
       "loss": loss,
       "accuracy": accuracy,
       "cnn_gradients": None           # Nunca existe
   }
   ```

**PS recibe gradientes de todos los Workers:**
1. Promedia gradients_mlp de todos los Workers
2. Actualiza parámetros MLP:
   ```python
   params_mlp = SGD_update(params_mlp, avg_gradients, lr, momentum)
   ```
3. **CNN permanece sin cambios** en PS ← CRÍTICO

**Evaluación global (opcional):**
- PS usa X_test_features (cacheados) + MLP actual
- Calcula accuracy/loss de prueba

---

#### END-TO-END

**PS envía (PARAMS):**
```python
{
    "epoch": epoch,
    "params": params_mlp,       # W1, b1, W2, b2, W3, b3
    "seed": rng_seed,
    "cnn_params": cnn_state     # Estado actual de la CNN (dict[name] = array)
}
```

**Worker recibe (PARAMS) y computa:**
1. Reconstruye índices locales
2. Obtiene training set partition
3. Selecciona batch de imágenes raw:
   ```python
   X_batch = X_raw[indices]        # ← Imágenes raw, NO features cacheadas
   Y_batch = Y_raw[indices]
   ```
4. Forward CNN (con gradientes habilitados):
   ```python
   features_torch = CNN.forward(X_batch_torch)    # PyTorch
   ```
5. Forward MLP + Backward MLP para obtener ∇features:
   ```python
   dX_features, mlp_grads = MLP_backward(features, Y_batch)
   ```
6. Backward CNN usando ∇features:
   ```python
   CNN.backward(dX_features)       # Calcula gradientes CNN
   ```
7. Calcula:
   - gradients_mlp
   - gradients_cnn (dict[layer_name] = array)
   - loss
   - accuracy
8. Envía GRADIENTS:
   ```python
   {
       "epoch": epoch,
       "gradients": gradients_mlp,         # Gradientes MLP
       "loss": loss,
       "accuracy": accuracy,
       "cnn_gradients": gradients_cnn      # Gradientes CNN (dict)
   }
   ```

**PS recibe gradientes de todos los Workers:**
1. Promedia tanto gradients_mlp como gradients_cnn
2. Actualiza parámetros MLP:
   ```python
   params_mlp = SGD_update(params_mlp, avg_gradients_mlp, lr, momentum)
   ```
3. Actualiza parámetros CNN (NEW):
   ```python
   cnn_state = SGD_update_cnn(cnn_state, avg_gradients_cnn, lr)
   ```
4. Distribuyó los pesos CNN actualizados en la siguiente época

**Evaluación global (opcional):**
- PS usa X_test raw + CNN actual + MLP actual
- CNN extrae features on-the-fly
- Calcula accuracy/loss de prueba

---

### 2.3 Comunicación Parameter Server ↔ Worker

#### Mensajes (independientes del modo)

| Mensaje | Origen | Destino | Payload | Notas |
|---------|--------|---------|---------|-------|
| READY | Worker | PS | {} | Worker listo pero no tiene ID todavía |
| WORKER_ID | PS | Worker | {"id": int} | PS asigna ID único |
| CNN_WEIGHTS | PS | Worker | {"arch": str, "weights_bytes": bytes} | CNN inicial del PS |
| CNN_READY | Worker | PS | {} | Worker confirmó carga de CNN |
| TRAIN_START | PS | Worker | {"epochs": int, "n_train": int, "n_workers": int, "worker_rank": int} | Sesión comienza |
| PARAMS | PS | Worker | {"epoch": int, "params": dict, "seed": int, ["cnn_params": dict]} | Clave "cnn_params" solo en E2E |
| GRADIENTS | Worker | PS | {"epoch": int, "gradients": dict, "loss": float, "accuracy": float, ["cnn_gradients": dict]} | Clave "cnn_gradients" solo en E2E |
| STOP | PS | Worker | {} | Worker cierra conexión limpiamente |

---

## 3. DIFERENCIAS CLAVE (Tabla Comparativa)

| Aspecto | PRECOMPUTED | END-TO-END |
|--------|----------|-----------|
| **CNN se entrena** | ❌ NO (congelada) | ✅ SÍ (entrenable) |
| **MLP se entrena** | ✅ SÍ (distribuido) | ✅ SÍ (distribuido) |
| **Features precacheados** | ✅ SÍ (en RAM Worker) | ❌ NO |
| **Features se actualizan** | ❌ NUNCA | ✅ Cada época |
| **Datos de entrada al MLP** | Features (F: 512 dims) | Features dinámico (calculado cada época) |
| **Datos de entrada al entrenamiento Worker** | Features cacheados | Imágenes raw |
| **Gradientes enviados al PS** | Gradientes MLP | Gradientes MLP + Gradientes CNN |
| **PS actualiza** | solo MLP | MLP + CNN |
| **Caché de features en disco** | ✅ SÍ (reutilizable) | ❌ NO (innecesario) |
| **Memoria del Worker** | ~512 * 50000 = 26 MB + overhead | ~3 * 3 * 32 * 32 * 50000 = 1.5 GB raw (o dynamic batches) |
| **Tiempo de setup (CNN_READY)** | ~30-60s (extracción features) | <1s (solo carga pesos) |
| **Tiempo por época** | ~1-5s (MLP NumPy) | ~5-30s (CNN forward/backward PyTorch + MLP) |
| **Preentrenamiento CNN** | ✅ Sí (antes de distribuido) | ❌ NO (desde random/ImageNet) |
| **Arquitectura CNN válida** | simple + resnet18 | simple + resnet18 |
| **Aplicable desde UI "Cargar"** | ✅ (cargar CNN preentrenada) | ❌ (no tiene sentido sin features) |
| **Aplicable desde UI "Entrenar"** | ✅ (preentrenar + distribuido) | ✅ (optimizar E2E) |

---

## 4. PARÁMETROS POR MODO

### 4.1 Parámetros que existen en AMBOS modos

| Parámetro | Tipo | Rango | Propósito |
|-----------|------|-------|----------|
| **learning_rate** | float | 0.0001 - 10.0 | Tasa de actualización para SGD (MLP y CNN si E2E) |
| **momentum** | float | 0.0 - 0.99 | Coeficiente momentum para SGD |
| **seed** | int o None | - | Semilla para partición estratificada del batch |
| **hidden1** | int | 32 - 1024 | Neuronas capa oculta 1 del MLP |
| **hidden2** | int | 32 - 512 | Neuronas capa oculta 2 del MLP |
| **n_workers** | int | ≥1 | Número de Workers en la sesión (automático) |

### 4.2 Parámetros específicos PRECOMPUTED

| Parámetro | Tipo | Rango | Propósito |
|-----------|------|-------|----------|
| **epochs_mlp** | int | 50 - 1000 | Épocas de entrenamiento MLP distribuido |
| **n_train_mlp** | int | 100 - 50000 | Ejemplos de entrenamiento MLP por época total |
| **cnn_arch** | str | "simple" \| "resnet18" | Arquitectura CNN (preentrenada antes de distribuido) |
| **cnn_pretrain_epochs** | int | 0 - 50 | Épocas preentrenamiento CNN local (si arch="simple" y no cargada) |
| **cnn_pretrain_lr** | float | 1e-4 - 0.1 | Learning rate preentrenamiento CNN |
| **cnn_pretrain_samples** | int | 100 - 50000 | Muestras para preentrenamiento CNN local |

### 4.3 Parámetros específicos END-TO-END

| Parámetro | Tipo | Rango | Propósito |
|-----------|------|-------|----------|
| **epochs_e2e** | int | 50 - 1000 | Épocas entrenamiento CNN + MLP distribuido |
| **n_train_e2e** | int | 100 - 50000 | Ejemplos de entrenamiento total por época |
| **cnn_arch** | str | "simple" \| "resnet18" | Arquitectura CNN (no preentrenada, se entrena desde el inicio) |

### 4.4 Parámetros que NO tienen sentido en cada modo

**NO USAR en PRECOMPUTED:**
- ❌ gradientes CNN (no se calculan)
- ❌ actualizar pesos CNN (CNN congelada)
- ❌ features dinámicos (features cacheados)

**NO USAR en END-TO-END:**
- ❌ features cacheados / precalculados (se calculan cada época)
- ❌ caché en disco de features (innecesario)
- ❌ preentrenamiento CNN local (se entrena distribuido desde el inicio)

---

## 5. IMPLICACIONES PARA LA INTERFAZ UI (ps_gui.py)

### 5.1 Controles que SIEMPRE están activos (ambos modos)

- **Neuronas ocultas 1 y 2**
- **Learning rate**
- **Momentum**
- **Semilla MLP para reproducibilidad**

### 5.2 Controles visibles SOLO en PRECOMPUTED

| Control | Activo cuando | Significado |
|---------|--|----------|
| **Épocas MLP** | Siempre | Entrenar MLP distribuido |
| **Ejemplos entrenamiento MLP** | Siempre | N ejemplos por sesión MLP |
| **Selector Cargar/Entrenar CNN** | Siempre | Modo del CNN |
| **Épocas CNN** | CNN en modo "Entrenar" | Solo si preentrenamos CNN |
| **LR CNN** | CNN en modo "Entrenar" | Solo si preentrenamos CNN |
| **Muestras CNN** | CNN en modo "Entrenar" | Solo si preentrenamos CNN |

### 5.3 Controles visibles SOLO en END-TO-END

| Control | Activo cuando | Significado |
|---------|--|----------|
| **Épocas E2E** | Siempre | Entrenar CNN + MLP distribuido |
| **Ejemplos entrenamiento global** | Siempre | N ejemplos por sesión distribuida |
| **CNN siempre en modo "Entrenar"** | Solo existe este modo | No existe "Cargar modelo" en E2E |

### 5.4 Regla de activación rigurosa

```
SI modo_sistema == "precomputed":
    - Épocas MLP: ENABLED
    - Ejemplos MLP: ENABLED
    - Épocas E2E: DISABLED
    - Ejemplos E2E: DISABLED
    - Selector CNN Cargar/Entrenar: ENABLED
    - Controles CNN Entrenar (épocas, LR, muestras): DISABLED si CNN.mode="load"
                                                       ENABLED si CNN.mode="train"
    
SI modo_sistema == "end_to_end":
    - Épocas MLP: DISABLED
    - Ejemplos MLP: DISABLED
    - Épocas E2E: ENABLED
    - Ejemplos E2E: ENABLED
    - Selector CNN Cargar/Entrenar: DISABLED (solo "Entrenar")
    - Controles CNN: DISABLED (CNN siempre se entrena desde el inicio)
```

---

## 6. IMPLICACIONES PARA EL CÓDIGO ACTUAL

### 6.1 Puntos de separación lógica requerida

#### En Parameter Server (parameter_server.py)

**Punto 1: train() — Inicialización de CNN**
```
Decisión: ¿training_mode == "precomputed"?
- Si SÍ:  CNN se distribuye congelada, features se extraen en Workers
- Si NO:  CNN se distribuye entrenable, no se precalculan features
```

**Punto 2: train() — Estructura de PARAMS**
```
Decisión: ¿training_mode == "end_to_end"?
- Si SÍ:  Incluir "cnn_params" en el payload
- Si NO:  No incluir "cnn_params"
```

**Punto 3: train() — Procesamiento de GRADIENTS**
```
Decisión: ¿training_mode == "end_to_end"?
- Si SÍ:  Esperar y procesar "cnn_gradients" de cada Worker
- Si NO:  Ignorar cualquier "cnn_gradients" (será None)
```

**Punto 4: train() — Actualización de parámetros**
```
Decisión: ¿training_mode == "end_to_end"?
- Si SÍ:  Actualizar both params_mlp AND cnn_state
- Si NO:  Actualizar solo params_mlp
```

#### En Worker (worker_node.py)

**Punto 1: _handle_cnn_weights() — Preparación CNN**
```
Decisión: ¿training_mode == "precomputed"?
- Si SÍ:  set_trainable(False), extrae features, cachea   
- Si NO:  set_trainable(True), NO extrae, usa X_raw
```

**Punto 2: _handle_params() — Forward/Backward**
```
Decisión: ¿training_mode == "precomputed"?
- Si SÍ:  X = cached features;  backward MLP only
- Si NO:  X = raw; backward CNN → backward MLP
```

**Punto 3: _handle_params() — Graduales CNN**
```
Decisión: ¿training_mode == "end_to_end"?
- Si SÍ:  Calcular gradientes CNN, incluir en payload
- Si NO:  cnn_gradients = None
```

#### En GUI (ps_gui.py)

**Punto 1: _on_system_mode_change()**
```
Decisión: cambiar modo_sistema de UI
- Actualizar visibilidad/estado de controles según tabla 5.4
```

**Punto 2: _on_cnn_mode_change()**
```
Decisión: cambiar CNN mode "load" ↔ "train" (solo precomputed)
- Actualizar visibilidad/estado de controles CNN
```

**Punto 3: _cmd_train()**
```
Decisión: iniciar sesión de entrenamiento
- Leer training_mode
- Seleccionar epochs según modo
- Seleccionar n_train según modo
- Pasar training_mode al PS
```

### 6.2 Código que mezcla ambos modos (REQUIERE SEPARACIÓN)

**PROBLEMA 1:** En Worker._handle_params()

Actual: Mezcla "precomputed" y "end_to_end" en una sola rama

```python
# ACTUAL (problematic pseudocode):
if training_mode == "precomputed":
    # ... pero el else hace también ...
    # features, backward, etc. sin distinguir claramente
```

Debería ser:
```python
# CORRECTO:
if training_mode == "precomputed":
    X_batch = self._X_features[indices]
    backward_mlp_only()
else:  # end_to_end
    X_batch = self._X_raw[indices]
    backward_cnn_then_mlp()
```

**PROBLEMA 2:** En ParameterServer.train()

Actual: Calcula cnn_gradients y cnn_params sin verificar mode

```python
# ACTUAL:
cnn_gradients = payload.get("cnn_gradients")  # Siempre None en precomputed
```

Debería ser:
```python
# CORRECTO:
if self.training_mode == "end_to_end":
    cnn_gradients = payload["cnn_gradients"]
    avg_cnn_grad = aggregate(cnn_gradients)
    self.update_cnn(avg_cnn_grad)
else:
    # No esperar ni procesar gradientes CNN
    pass
```

**PROBLEMA 3:** En GUI._cmd_train()

Actual: Selecciona épocas sin validar consistencia con modo

```python
# ACTUAL:
if training_mode == "precomputed":
    epochs = self._v_epochs.get()  # Bien
else:
    epochs = self._v_e2e_epochs.get()  # Bien
# PERO: no se valida que los otros se hayan deshabilitado en la UI
```

Debería ser:
```python
# CORRECTO:
# El UI state DEBE garantizar que está disabled/enabled correctamente
# Validar en _cmd_train() como fail-safe:
if training_mode == "precomputed":
    epochs = self._v_epochs.get()
    n_train = self._v_n_train_mlp.get()
    assert epochs > 0 and n_train > 0
else:
    epochs = self._v_e2e_epochs.get()
    n_train = self._v_n_train_e2e.get()
    assert epochs > 0 and n_train > 0
```

### 6.3 Funciones que requieren actualización

| Función | Archivo | Cambios requeridos |
|---------|---------|-------------------|
| ParameterServer.train() | parameter_server.py | Separar lógica por training_mode en cada época |
| ParameterServer._handle_cnn_weight() | parameter_server.py | Validar que training_mode es coherente |
| WorkerNode._handle_cnn_weights() | worker_node.py | Separar precomputed vs end_to_end en preparación |
| WorkerNode._handle_params() | worker_node.py | Separar forward/backward por training_mode |
| DistributedPSApp._cmd_train() | ps_gui.py | Usar params según training_mode, validar state |
| DistributedPSApp._on_system_mode_change() | ps_gui.py | Actualizar UI state de forma centralizada ✓ (ya hecho) |

---

## 7. REGLAS ESTRICTAS (Invariantes del Sistema)

### 7.1 Reglas PRECOMPUTED

**[R1.1]** CNN nunca se actualiza en ParameterServer durante train()
- Los pesos CNN permanecen idénticos desde el inicio hasta el final de todas las épocas
- self._cnn debe ser inmutable durante la sesión de entrenamiento

**[R1.2]** Features de entrenamiento se calculan UNA SOLA VEZ por Worker
- En _handle_cnn_weights(), NO en cada época
- Se almacenan en self._X_features (RAM)
- Se indexan por época, nunca se recalculan

**[R1.3]** NO debe existir "cnn_gradients" en ningún payload
- Si payload.get("cnn_gradients") != None en precomputed, ERROR
- El Worker NUNCA calcula gradientes CNN
- El PS NUNCA espera gradientes CNN

**[R1.4]** Caché de features en disco es validado
- Hash de pesos debe coincidir con archivo de caché
- Si no coincide, features se re-extraen

**[R1.5]** Modo de UI "Cargar modelo" solo existe en precomputed
- End-to-end no ofrece selector "Cargar/Entrenar" (solo "Entrenar")

---

### 7.2 Reglas END-TO-END

**[R2.1]** CNN se actualiza en cada época en ParameterServer
- cnn_state se modifica después de agregar gradientes
- Distribuyó en PARAMS de siguiente época

**[R2.2]** CNN se entrena desde el inicio
- NO existe preentrenamiento local (cnn_pretrain_epochs = 0)
- CNN comienza random o ImageNet, se optimiza distribuida desde la época 1

**[R2.3]** Features se calculan dinámicamente en cada época
- NO se precachean en disco
- Worker guarda X_raw en memoria (imagen nativa)
- Cada época: forward X_batch → features → MLP

**[R2.4]** Siempre existen "cnn_gradients" en payload
- Si payload.get("cnn_gradients") == None en end_to_end, ERROR
- El Worker SIEMPRE calcula gradientes CNN
- El PS SIEMPRE espera y procesa gradientes CNN

**[R2.5]** Selector "Cargar modelo" no existe en la UI
- Sistema fuerza modo "Entrenar modelo"
- No tiene sentido cargar features fijos sin CNN entrenable

**[R2.6]** Memoria Worker es diferente
- Debe almacenar X_raw completo (50000, 3, 32, 32) o procesarlo por batch
- NO X_features cacheados

---

### 7.3 Reglas GLOBALES (Ambos modos)

**[RG1]** training_mode es inmutable durante una sesión de entrenamiento
- Se establece en ParameterServer.__init__() y no cambia hasta shutdown()
- La UI puede cambiar entre sesiones, pero no durante una

**[RG2]** Todos los Workers usan el mismo training_mode
- El PS no mezcla Workers en diferentes modos
- El _active_training_workers todos operan con el mismo training_mode

**[RG3]** Parámetros desabilitados en la UI NO afectan el entrenamiento
- Si un Worker recibe training_mode pero la UI deshabilitó un valor, fallar
- Validar en _cmd_train() antes de iniciar

**[RG4]** Hash de pesos CNN
- En precomputed: hash es INMUTABLE (para caché de features)
- En end_to_end: hash cambia cada época (CNN se actualiza)

**[RG5]** Logs y feedback deben reflejar el mode actual
- Mensajes del PS deben indicar "Precomputed" vs "End-to-End"
- Mensajes del Worker deben indicar qué se calcifica

---

## 8. TABLA RESUMIDA: DECISIONES CRÍTICAS POR COMPONENTE

| Componente | Decisión | Si PRECOMPUTED | Si END-TO-END |
|-----------|----------|---|---|
| **PS.train()** | CNN se actualiza | NO | SÍ |
| **PS.train()** | Parámetro CNN en PARAMS | NO | SÍ |
| **PS.train()** | Procesa cnn_gradients | NO | SÍ |
| **PS.train()** | Features precalculados | SÍ | NO |
| **Worker._handle_cnn_weights()** | set_trainable | False | True |
| **Worker._handle_cnn_weights()** | Extrae features | SÍ | NO |
| **Worker._handle_params()** | Entrada X | Cached features | Raw images |
| **Worker._handle_params()** | Backward CNN | NO | SÍ |
| **Worker payload** | Incluye cnn_gradients | NO | SÍ |
| **UI.PRECOMPUTED** | Visible | Épocas MLP, Cargar/Entrenar CNN | NO |
| **UI.END_TO_END** | Visible | Épocas E2E, Entrenar CNN (solo) | SÍ |

---

## 9. VALIDACIÓN: CHECKLIST PARA REFACTORIZACIÓN

### 9.1 Parameter Server Class

- [ ] ParameterServer.__init__() almacena training_mode y NO cambia durante la sesión
- [ ] train() separado en ramas claras: if training_mode == "precomputed" vs else
- [ ] Rama precomputed: NO actualiza CNN, envía CNN (fijo) en CNN_WEIGHTS
- [ ] Rama end_to_end: actualiza CNN después de agregar gradientes, envía cnn_params en PARAMS
- [ ] _handle_params() valida presencia/ausencia de cnn_gradients según mode
- [ ] Logs indican claramente el mode operativo

### 9.2 Worker Class

- [ ] WorkerNode.__init__() almacena training_mode, coherente con PS
- [ ] _handle_cnn_weights() tiene dos ramas claras: precomputed vs end_to_end
- [ ] Rama precomputed: set_trainable(False), extrae features, caché
- [ ] Rama end_to_end: set_trainable(True), NO extrae features, guarda X_raw
- [ ] _handle_params() diferencia X según mode: features vs raw
- [ ] _handle_params() calcula cnn_gradients solo si end_to_end
- [ ] Payload GRADIENTS incluye cnn_gradients si end_to_end, omite si precomputed
- [ ] Logs indican claramente la rama ejecutada

### 9.3 UI (ps_gui.py)

- [ ] _update_widget_states() valida mode del sistema y modo CNN
- [ ] Controles PRECOMPUTED deshabilitados cuando modo=end_to_end
- [ ] Controles END_TO_END deshabilitados cuando modo=precomputed
- [ ] Selector Cargar/Entrenar CNN deshabilitado en end_to_end
- [ ] _cmd_train() valida que los valores no sean contradictorios con el mode
- [ ] Logs reflejan mode y decisiones tomadas

### 9.4 Protocol / Mensajes

- [ ] PARAMS NEVER envía "cnn_params" en precomputed
- [ ] PARAMS ALWAYS envía "cnn_params" en end_to_end
- [ ] GRADIENTS NEVER envía "cnn_gradients" en precomputed
- [ ] GRADIENTS ALWAYS envía "cnn_gradients" en end_to_end

---

## 10. CASOS DE USO VÁLIDOS E INVÁLIDOS

### Caso de uso VÁLIDO: Precomputed

```
1. UI: Seleccionar "Precomputación (CNN fija + MLP distribuido)"
2. UI: Seleccionar "Entrenar modelo" para CNN
3. UI: Configurar: épocas_cnn=10, lr_cnn=0.001, muestras_cnn=10000
4. PS inicia: cnn.pretrain(10 épocas)
5. UI: Seleccionar "Iniciar entrenamiento" con épocas_mlp=200, lr=0.01
6. PS distribuye CNN congelada a Workers
7. Workers calculan X_features en caché
8. Entrenar 200 épocas: solo MLP se actualiza
✓ VÁLIDO
```

### Caso de uso VÁLIDO: End-to-End

```
1. UI: Seleccionar "End-to-End (CNN + MLP se entrenan juntos)"
2. UI: Selector Cargar/Entrenar DESHABILITADO (forzado a "Entrenar")
3. UI: Configurar: épocas_e2e=100, lr=0.001
4. PS inicia: CNN con pesos random o ImageNet
5. PS distribuye CNN entrenable a Workers
6. Workers NO calculan X_features (guardación X_raw)
7. Entrenar 100 épocas: CNN + MLP se actualizan juntos
✓ VÁLIDO
```

### Caso de uso INVÁLIDO: Mezcla

```
1. UI: "Precomputed"
2. UI: Intentar acceder a "épocas_e2e" slider
❌ INVÁLIDO: Debe estar deshabilitado
```

---

## 11. NOTAS FINALES

### 11.1 Principios de diseño

1. **Claridad sobre sutileza:** Dos ramas claramente separadas, no un único código con condicionales dispersos
2. **Fail-fast:** Validar presencia/ausencia de cnn_gradients, cnn_params en cada punto de entrada
3. **Simetría:** Si precomputed NO envía X, end_to_end SIEMPRE envía en la misma forma
4. **UI refleja lógica:** Lo que se desactiva en la UI corresponde a lo que no se ejecuta en el código

### 11.2 Métricas de éxito post-refactorización

- [ ] training_mode es una variable de decisión clara en 5+ puntos del código
- [ ] Cambiar training_mode en PS.__init__() causa cambios predecibles en todo el flujo
- [ ] No hay código "muerto" (else branches que nunca se ejecutan)
- [ ] Logs claramente indican qué rama se ejecutó
- [ ] Tests pueden ser separado por mode sin condicionales

### 11.3 Próximos pasos (después de esta especificación)

1. Refactorizar ParameterServer.train() en dos métodos: _train_precomputed() y _train_end_to_end()
2. Refactorizar WorkerNode._handle_params() de la misma forma
3. Crear tests que fuerzen cada mode y verifiquen las reglas estrictas
4. Documentar en README.md los dos modos con ejemplos operativos

---

**FIN DE ESPECIFICACIÓN**

