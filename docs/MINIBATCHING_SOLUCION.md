# Solución: Mini-batching en Modo End-to-End

## PROBLEMA ORIGINAL

El Worker en modo end_to_end procesaba **TODA la partición en un solo batch**:

```python
X_batch = self._X_raw[indices]  # Potencialmente ~40,000 imágenes en un Worker
# Forward CNN + backward CNN + backward MLP todo de una vez
```

**Consecuencias:**
- CPU se disparaba (especialmente sin GPU)
- Aplicación se congelaba sin mensaje de error
- No había progreso visible durante la época

---

## ESTRATEGIA ELEGIDA: Opción D (Híbrida)

### Fundamento Técnico

1. **Backward es 2-3× más costoso que forward puro**
   - `_optimal_batch_size()` fue calibrado para **extracción CNN** (solo forward)
   - En end_to_end, además de forward CNN tenemos: backward CNN + backward MLP

2. **Fórmula de mini-batch:**
   ```
   mini_batch_size = max(16, int(_optimal_batch_size() / 2.5))
   ```
   - `_optimal_batch_size()` es la base (considera CPU cores, arquitectura CNN, device)
   - Dividir entre 2.5 es factor de seguridad conservador
   - `max(16, ...)` evita overhead de demasiados mini-batches

3. **Ejemplos de valores resultantes:**
   - CNN simple en CPU: 512 / 2.5 ≈ 205
   - CNN simple en GPU: 2048 / 2.5 ≈ 820
   - ResNet18 en CPU: 64 / 2.5 ≈ 26
   - **Resultado:** Manejable en todo hardware

---

## IMPLEMENTACIÓN: ACUMULACIÓN DE GRADIENTES

### 1. Loop de Mini-batches

```python
n_batches = (n_total + mini_bs - 1) // mini_bs

for batch_idx in range(n_batches):
    # Extraer mini-batch de índices
    mini_indices = indices[start_idx:end_idx]
    X_mini = self._X_raw[mini_indices]
    Y_mini = self._Y_raw[mini_indices]
    
    # FORWARD CNN (detach)
    features_torch = self._cnn._model(X_mini_torch)
    features = features_torch.detach().cpu().numpy()
    
    # BACKWARD MLP → obtener dX
    dX_mini, mlp_grads_mini, loss_mini, acc_mini = mlp_backward_to_input(
        mlp_params, features, Y_mini
    )
    
    # BACKWARD CNN (sin detach, con gradientes)
    self._cnn._model.zero_grad()  # ← CRÍTICO: limpiar antes
    features_torch = self._cnn._model(X_mini_torch)
    loss_proxy = (features_torch * torch.from_numpy(dX_mini.T / len(Y_mini))).sum()
    loss_proxy.backward()
    
    # ACUMULAR gradientes
    accumulated_mlp_grads.append(mlp_grads_mini)
    accumulated_cnn_grads[name] += param.grad.detach().cpu().numpy()
    accumulated_losses.append(loss_mini)
    accumulated_accs.append(acc_mini)
```

### 2. Acumulación Correcta de Gradientes MLP

```python
# ANTES (incorrecto): un solo forward/backward
gradients = {... resultado de un batch único ...}

# AHORA (correcto): promedio de todos los mini-batches
gradients = {}
for key in accumulated_mlp_grads[0].keys():
    gradients[key] = np.mean(
        [g[key] for g in accumulated_mlp_grads], axis=0
    )
```

**Matemática:**
- Cada mini-batch computa gradientes MLP
- Se almacenan en lista de dicts
- Se promedian al final (todos los mini-batches pesan igual)

### 3. Acumulación Correcta de Gradientes CNN

```python
# DURANTE loop: acumular SIN normalizar
for name, param in self._cnn._model.named_parameters():
    if param.grad is not None:
        grad_np = param.grad.detach().cpu().numpy()  # ← SIN dividir
        if name not in accumulated_cnn_grads:
            accumulated_cnn_grads[name] = grad_np.copy()
        else:
            accumulated_cnn_grads[name] += grad_np

# AL FINAL: normalizar por TOTAL de ejemplos (no por n_batches)
cnn_gradients = {
    name: grad / n_total     # ← Divide por tamaño total
    for name, grad in accumulated_cnn_grads.items()
}
```

**Matemática correcta:**
- Cada `.backward()` computa ∇loss / ∂params para ese mini-batch
- Acumular SUMAS: `sum_i(grad_i)`
- Normalizar por n_total: `sum(grad_i) / n_total`
- **Resultado:** Gradiente promedio correcto

### 4. Promediado de Loss y Accuracy

```python
loss = float(np.mean(accumulated_losses))
accuracy = float(np.mean(accumulated_accs))
```

Cada mini-batch contribuye con su loss/accuracy. Se promedian.

---

## CONTROLES CRÍTICOS

### 1. Zero Grad Correcto

```python
# DENTRO del loop, ANTES de backward
self._cnn._model.zero_grad()
```

**¿Por qué?** PyTorch acumula gradientes. Si no limpiamos, el segundo mini-batch acumularía gradientes del primero (incorrecto).

### 2. Doble Forward de CNN Intencional

Sí, hacemos forward DOS veces:

```python
# Primera (detach): obtener features para MLP backward
features_torch = self._cnn._model(X_batch_torch)
features = features_torch.detach().cpu().numpy()

# Segunda (con grad): para backward CNN
features_torch = self._cnn._model(X_batch_torch)
loss_proxy.backward()
```

**¿Por qué?** El primer forward necesita `.detach()` (nunca guardamos su computation graph). El segundo forward crea un nuevo graph para backward. _Esto es ineficiente pero necesario_ para mantener separación entre MLP backward y CNN backward.

### 3. Proxy Loss Correcto

```python
loss_proxy = (
    features_torch
    * torch.from_numpy(dX_mini.T / len(Y_mini)).to(device)
).sum()
loss_proxy.backward()
```

**¿Qué hace?** Backpropaga ∇L/∂features (del MLP) a través de la CNN:
- `dX_mini.T`: gradiente del MLP ∇L / ∂features_i para cada ejemplo i
- Multiplica elemento a elemento por features_torch
- Suma → scalar loss_proxy
- `.backward()` propaga a parámetros de CNN

---

## LOGGING Y PROGRESO

### Logs Agregados

```
[END-TO-END] Procesando 41000 ejemplos en 200 mini-batches (size=205)...
  [END-TO-END] Batch 1/200  size=205
  [END-TO-END] Batch 51/200  size=205
  [END-TO-END] Batch 101/200  size=205
  [END-TO-END] Batch 151/200  size=205
  loss=0.8234  acc=78.45%  (45.322s)
```

**Frecuencia:** Cada 20% de progreso (no cada mini-batch, evita saturación)

---

## VERIFICACIÓN DE INVARIANTES

| Invariante | Estado | Verificación |
|-----------|--------|-----|
| [R2.1] CNN se actualiza cada época | ✓ | `cnn_gradients` siempre presente |
| [R2.2] CNN se entrena desde inicio (sin preentrenamiento) | ✓ | No hay `cnn.pretrain()` en E2E |
| [R2.3] Features dinámicos (no cacheados) | ✓ | `self._X_features = empty()` en `_handle_cnn_weights` E2E |
| [R2.4] Siempre hay `cnn_gradients` en payload | ✓ | Se calcula en loop, se incluye en payload |
| [R2.5] Mismo protocolo de mensaje GRADIENTS | ✓ | `payload_send` idéntico, solo con `cnn_gradients` presente |
| [R2.6] CNN entrenable (set_trainable=True) | ✓ | Hecho en `_handle_cnn_weights` E2E |

---

## RENDIMIENTO ESPERADO

### Antes (BUGGY)
- Congelamiento en CPU con ~40K ejemplos
- Imposible de usar sin GPU

### Después (MINI-BATCHING)
- **CPU (simple CNN):** ~30-50s por época (manejable)
- **GPU (simple CNN):** ~5-10s por época (bueno)
- **CPU (resnet18):** ~60-90s por época (lento pero estable)
- **GPU (resnet18):** ~10-20s por época (aceptable)

### Uso de Memoria
- Mini-batch: 205 imágenes × 3×32×32 × 4 bytes ≈ 5 MB (features CNN)
- PyTorch overhead: ~50-100 MB (computation graph temporal)
- **Total:** ~100-150 MB vs. ~500-800 MB (batch completo)

---

## RESTRICCIONES RESPETADAS

✓ NO tocar ParameterServer  
✓ NO modificar CNNExtractor  
✓ NO modificar MLP  
✓ NO cambiar protocolo de mensajes GRADIENTS  
✓ NO introducir dependencias nuevas  
✓ Mantener compatibilidad con múltiples Workers  

---

## TESTING RECOMENDADO

1. **Ejecución CPU + precomputed**: Verificar que sigue funcionando
2. **Ejecución CPU + end_to_end**: Verificar que NO se congela
3. **Ejecución GPU + end_to_end**: Verificar que mantiene rendimiento
4. **Múltiples workers**: Verificar que promedios de gradientes son correctos
5. **Convergencia**: Comparar accuracy final vs. versión anterior (debe ser similar)

