# 4. MODOS DE ENTRENAMIENTO: PRECOMPUTED vs END-TO-END

## 📊 Comparación rápida

| Aspecto | PRECOMPUTED | END-TO-END |
|--------|-------------|-----------|
| **CNN** | Congelada (sin gradientes) | Entrenable (con gradientes) |
| **Features** | Pre-extraidos, cacheados | Calculados cada época |
| **Setup** | ~45s (extracción) | ~2s (solo cargar) |
| **Época típica** | ~2s (NumPy MLP) | ~25s (CNN forward+backward) |
| **Tamaño PARAMS** | ~60 KB | ~50 MB (CNN + MLP) |
| **Tamaño GRADIENTS** | ~60 KB | ~50 MB (CNN + MLP) |
| **Mejor para** | Validación rápida, CNN buena | Ajuste fino, investigación |
| **Requiere GPU** | No (NumPy) | Recomendable (CNN costosa) |

---

## 🔍 Flujo PRECOMPUTED en detalle

### **Idea clave**

"La CNN es un extractor de features congelado. Una vez que tenemos los features, solo necesitamos entrenar el MLP lineal."

```
┌─────────────────────────────────────────────────────┐
│                PRECOMPUTED PIPELINE                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│ SETUP (UNA VEZ):                                    │
│  Raw Images (50K) ──CNN (frozen)──> Features (50K) │
│         Cachear features                            │
│                                                     │
│ CADA ÉPOCA:                                         │
│  Feature[índices] ──MLP──> Logits ──Loss            │
│         ∇MLP → PS                                   │
│                                                     │
│ CNN nunca se entrena.                               │
│ Solo gradientes MLP se comunican.                   │
└─────────────────────────────────────────────────────┘
```

### **Código del Worker (rama PRECOMPUTED)**

```python
# Cuando recibe CNN_WEIGHTS:
def _handle_cnn_weights(self, payload):
    if self.training_mode == "precomputed":
        # [1] Congelar CNN
        self._cnn.set_trainable(False)  # requires_grad = False
        
        # [2] Extraer features de TODOS los datos de train
        #     Esto es correcto porque CNN no cambia
        X_features, Y = self._load_features_with_cache(
            self._X_raw,  # 50,000 images
            self._Y_raw,
            arch=self._cnn.arch,
            batch_size=2048
        )  # Cache hit: si no cambió CNN hash, reutiliza (~0.5s)
           # Cache miss: extrae ahora (~30-60s)
        
        self._X_features = X_features  # Guardar en memoria
        
        # [3] Confirmar
        send_message(self._sock, MsgType.CNN_READY, ...)

# Cuando recibe PARAMS (cada época):
def _handle_params(self, payload):
    if self.training_mode == "precomputed":
        # [1] Reconstruir índices usando seed
        indices = self._reconstruct_indices(
            payload["seed"],
            n_train=50000,
            n_workers=2,
            worker_rank=0
        )  # Resultado: ~25,000 índices
        
        # [2] Indexar features (no recalcular CNN)
        X_batch = self._X_features[indices]  # (25K, 512)
        Y_batch = self.Y_train[indices]       # (25K,)
        
        # [3] Forward + Backward MLP únicamente
        gradients, loss, acc = forward_and_gradients(
            self.mlp_params,
            X_batch,  # Features ya calculados
            Y_batch
        )
        
        # [4] Enviar gradientes al PS
        send_message(self._sock, MsgType.GRADIENTS, {
            "gradients": gradients,    # MLP gradients only
            "cnn_gradients": None      # CNN no se entrena
        })
```

### **Mensajes en PRECOMPUTED**

```
PS → Worker: PARAMS
├─ epoch: int
├─ params: {W1, b1, W2, b2, W3, b3}  (MLP only)
├─ seed: int (for reconstruction)
└─ cnn_params: None  ←─ IMPORTANTE: ausente

Worker → PS: GRADIENTS
├─ gradients: {dW1, db1, dW2, db2, dW3, db3}  (MLP only)
└─ cnn_gradients: None  ←─ IMPORTANTE
```

### **Caché en PRECOMPUTED**

```
Setup:
  CNN weights: W₀ (inicial)
  Hash(W₀) = "abc123def456"
  
  Calcula: X_train_features = CNN_simple(X_raw)
  Cachea:  Data/feature_cache/
           simple_abc123def456_train_X.npy  (200 MB)
           simple_abc123def456_train_Y.npy  (200 KB)

La próxima vez que se creen features con los MISMOS pesos:
  Hash match → CACHE HIT (0.5s)
  Hash mismatch → CACHE MISS (30-60s)
```

### **Flujo temporal**

```
T=0s:       Worker se conecta
T=1s:       Recibe CNN_WEIGHTS
T=2-50s:    Extrae features (o carga caché)
T=50s:      CNN_READY
T=51-60s:   Espera TRAIN_START
T=60s:      TRAIN_START recibido
─────────── EPOCH 0 ──────────
T=60-62s:   PARAMS → calcula gradientes → GRADIENTS
─────────── EPOCH 1 ──────────
T=62-64s:   PARAMS → calcula gradientes → GRADIENTS
─ … ─
─────────── EPOCH 9 ──────────
T=78-80s:   PARAMS → calcula gradientes → GRADIENTS
T=80s:      STOP

Total: ~80 segundos para 10 épocas
```

---

## 🚀 Flujo END-TO-END en detalle

### **Idea clave**

"Entrenar tanto la CNN como el MLP. Features se recalculan cada época con pesos CNN actualizados."

```
┌─────────────────────────────────────────────────────┐
│              END-TO-END PIPELINE                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│ SETUP:                                              │
│  Carga CNN (pesos iniciales pero requires_grad=True)│
│  NO extrae features (se calcularán under de época)  │
│  Guarda images raw en RAM                           │
│                                                     │
│ CADA ÉPOCA:                                         │
│  Raw Images ──CNN (with gradients)──> Features      │
│              ──MLP──> Logits ──Loss                 │
│         ∇CNN + ∇MLP → PS                            │
│                                                     │
│ CNN se actualiza cada época.                        │
│ Gradientes CNN + MLP se comunican (costoso).        │
└─────────────────────────────────────────────────────┘
```

### **Código del Worker (rama END-TO-END)**

```python
# Cuando recibe CNN_WEIGHTS:
def _handle_cnn_weights(self, payload):
    if self.training_mode == "end_to_end":
        # [1] Habilitar CNN para gradientes
        self._cnn.set_trainable(True)  # requires_grad = True
        
        # [2] NO extraer features
        #     Guardar images raw — se procesarán cada época
        self._X_features = np.empty((0,), dtype=np.float32)  # placeholder
        self._X_raw = X_raw  # Mantener en RAM
        
        # [3] Confirmar
        send_message(self._sock, MsgType.CNN_READY, ...)

# Cuando recibe PARAMS (cada época):
def _handle_params(self, payload):
    if self.training_mode == "end_to_end":
        # [1] VALIDACIÓN: debe haber cnn_params
        if payload.get("cnn_params") is None:
            raise RuntimeError("E2E mode requires cnn_params in PARAMS")
        
        # [2] Cargar CNN con pesos actualizados (desde PS)
        self._cnn.load_weights_from_bytes(payload["cnn_params"])
        
        # [3] Reconstruir índices
        indices = self._reconstruct_indices(
            payload["seed"],
            n_train=50000,
            n_workers=2,
            worker_rank=0
        )
        
        # [4] FORWARD CNN + MLP (on-the-fly)
        X_batch_raw = self._X_raw[indices]  # (25K, 3, 32, 32)
        Y_batch = self.Y_train[indices]
        
        X_batch_features = self._cnn.forward(X_batch_raw)  # (25K, 512)
        
        logits, cnn_hidden_states = mlp_forward(
            self.mlp_params,
            X_batch_features
        )
        
        loss = cross_entropy_loss(logits, Y_batch)
        
        # [5] BACKWARD MLP
        mlp_grads = mlp_backward(loss, self.mlp_params, ...)
        
        # [6] BACKWARD CNN (through MLP)
        cnn_grads = cnn_backward(
            loss,
            X_batch_raw,
            cnn_hidden_states,
            ...
        )
        
        # [7] Enviar AMBOS gradientes al PS
        send_message(self._sock, MsgType.GRADIENTS, {
            "gradients": mlp_grads,
            "cnn_gradients": cnn_grads  # ← IMPORTANTE: presente
        })
```

### **Mensajes en END-TO-END**

```
PS → Worker: PARAMS
├─ epoch: int
├─ params: {W1, b1, W2, b2, W3, b3}  (MLP)
├─ seed: int (for reconstruction)
└─ cnn_params: <bytes>  ←─ IMPORTANTE: pesos CNN serializados

Worker → PS: GRADIENTS
├─ gradients: {dW1, db1, dW2, db2, dW3, db3}  (MLP)
└─ cnn_gradients: {layer1_weight, layer2_weight, ...}  ←─ IMPORTANTE
```

### **Caché en END-TO-END**

```
NO HAY CACHÉ de features.
Cada época, se calcula:
  X_batch_raw = cargar desde RAM
  X_batch_features = CNN.forward(X_batch_raw)  # Recalcular
  
Las imágenes raw están almacenadas en RAM (costoso, ~150 MB para 50K CIFAR-10).
```

### **Flujo temporal**

```
T=0s:       Worker se conecta
T=1s:       Recibe CNN_WEIGHTS
T=2s:       Habilita CNN (no extrae features)
T=2s:       CNN_READY
T=3-60s:    Espera TRAIN_START
T=60s:      TRAIN_START recibido
─────────── EPOCH 0 ──────────
T=60-85s:   PARAMS → CNN forward → MLP forward
            MLP backward → CNN backward → GRADIENTS
─────────── EPOCH 1 ──────────
T=85-110s:  PARAMS → CNN forward → MLP forward
            MLP backward → CNN backward → GRADIENTS
─ … ─
─────────── EPOCH 9 ──────────
T=285-310s: PARAMS → CNN forward → MLP forward
            MLP backward → CNN backward → GRADIENTS
T=310s:     STOP

Total: ~310 segundos para 10 épocas
         (10x más lento que PRECOMPUTED)
```

---

## ⚡ Por qué END-TO-END es 10x más lentO

| Operación | PRECOMPUTED | E2E | Razón |
|-----------|-------------|-----|-------|
| Load features | 0.1s (indexar RAM) | 0.1s | igual |
| Forward CNN | 0s (ya hecho) | **3-5s** | CNN ejecución + backward tracking |
| Forward MLP | 0.1s | 0.1s | igual |
| Backward MLP | 0.1s | 0.1s | igual |
| Backward CNN | 0s (no se entrena) | **3-5s** | Propagación gradientes a través 18 capas |
| Serializar gradientes | 0.01s | **1-2s** | MLP 60KB vs CNN 50MB |
| **Total época** | **~0.3-0.5s** | **~7-13s** | Cálculo CNN domina |

---

## 🎯 Cuándo usar cada modo

### **PRECOMPUTED**
- ✅ Tienes una CNN buena preentrenada (ResNet + ImageNet)
- ✅ Quieres entrenar rápido (demostración, prototipo)
- ✅ Datos de prueba limitados (no quieres ajustar CNN)
- ✅ CPU-only environment (sin GPU)

**Ejemplo**: "Necesito verificar que el algoritmo de Diego funciona."

### **END-TO-END**
- ✅ Quieres ajustar la CNN a tu dataset (CIFAR-10)
- ✅ Tienes datos de entrenamiento abundantes (50K+)
- ✅ Tienes GPU disponible
- ✅ Investigación: quieres máxima precisión

**Ejemplo**: "Necesito optimizar tanto CNN como MLP para mi caso de uso."

---

## 🚨 Bug conocido: E2E test_acc collapse (SOLUCIONADO)

### **Problema (antes del fix)**

En modo END-TO-END, si `X_test_raw` no se pasaba al Worker, el PS no podía extraer features de test → se enviaban logits erráticos → test_acc colapsaba a ~10% (chance level).

```
PS                          Worker 0
 │                               │
 ├─ REQUEST_TEST_FEATURES ──────►│
 │                               │
 │                               ├─ X_test_raw is None!
 │                               ├─ No puedo extraer features
 │                               │
 │ ◄─────────────────────────────┤ (timeout)
 │
 PS intenta evaluar con features None
 → eval() retorna garbage
 → test_acc ≈ 10%
```

### **Solución**

En [ps_gui.py línea 1734](ps_gui.py#L1734):
```python
# Condicionalmente pasar X_test_raw al Worker si training_mode es E2E
x_test_for_train = X_test_raw if training_mode == "end_to_end" else None
```

También se deshabilitó cache loading en E2E (pues los pesos CNN cambian).

### **Verificación**

Después del fix, test_acc converge normalmente (~97-98%).

---

## 📝 Puntos críticos para explicar oralmente

1. **"En PRECOMPUTED, la CNN es una constante."**
   > Una vez que la extenuamos, es como si dijéramos: "Cada imagen = vector de 512 números". El MLP solo aprende a clasificar esos 512 números.

2. **"La diferencia de velocidad en E2E es pura CNN."**
   > El MLP toma ≈0.1s. La CNN toma ≈5-10s. El factor 10x viene de la CNN.

3. **"Precomputed es perfecto para validar el algoritmo.*
   > Si algo falla en PRECOMPUTED, es culpa del MLP o de la sincronización. No es la CNN.

4. **"En E2E, ambos componentes cambian cada época."**
   > CNN aprende features mejores. MLP se adapta a esas nuevas features. Más flexible pero más lento.

---

**Documento**: `docs/04_modes_precomputed_vs_e2e.md`  
**Última actualización**: 2026-03-27  
**Nivel**: Intermedio
