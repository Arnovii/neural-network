# Fix: "does not require grad and does not have a grad_fn"

## Diagnosis

**Error:** `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

**Root Cause:** En modo precomputed, la CNN se inicializa con `requires_grad=False` para todos sus parámetros (línea 196-198 en `Model/cnn_extractor.py`):
```python
for param in self._cnn._model.parameters():
    param.requires_grad_(False)  ← Esto previene backprop
self._cnn._model.eval()          ← Modo inferencia
```

Cuando cambias a modo END-TO-END, necesitas entrenar AMBOS CNN + MLP conjuntamente. Pero aunque envuelvas el forward en `torch.enable_grad()`, PyTorch NO crea un computation graph válido si los parámetros del modelo tienen `requires_grad=False`.

**Por qué el autograd falla:**
```python
# Input: X_mini_torch (requires_grad=False)
# CNN parámetros: requires_grad=False ← PROBLEMA
# Contexto: torch.enable_grad() activado

output = cnn(input)  # output.requires_grad = False ← Sin grad_fn
loss = output.sum()  # loss.requires_grad = False
loss.backward()      # ❌ ERROR: no hay grad_fn
```

## Solution

### 1. **Habilitar gradientes al inicio del flujo END-TO-END** (línea 625-629)

```python
# ✓ CRÍTICO: Habilitar gradientes en la CNN para entrenamiento
self._cnn._model.train()                    # BatchNorm diferenciable
for param in self._cnn._model.parameters(): # Parámetros entrenables
    param.requires_grad_(True)
```

**Por qué funciona:**
- `train()`: Activa Batch Normalization diferenciable
- `requires_grad_(True)`: Los parámetros ahora participan en backprop
- Dentro de `torch.enable_grad()`, el forward ahora genera un computation graph válido

### 2. **Remover línea redundante** (antes línea 689)

```python
# ❌ REMOVIO: X_mini_torch.requires_grad_(False)
# Ya no es necesario — los parámetros del modelo ahora tienen requires_grad=True
```

### 3. **Restaurar CNN a estado inicial** (después del loop, línea 745-753)

```python
# Al final del flujo END-TO-END
self._cnn._model.eval()                  # Modo inferencia
for param in self._cnn._model.parameters():
    param.requires_grad_(False)          # Vuelve a state inicial
```

**Por qué restaurar:**
- Garantiza que las próximas épocas precomputed no entrenen la CNN (invariante [R1.3])
- Las operaciones forward son más eficientes en eval() mode
- El estado es reproducible entre épocas y workers

## Testing

**Paso 1: Ejecutar Worker**
```bash
python worker.py
```

**Paso 2: Ejecutar Parameter Server en otra terminal**
```bash
python ps_terminal.py
```

**Paso 3: Configurar sesión END-TO-END**
```
1. Seleccionar "end_to_end" como training mode
2. Ejecutar: 1 epoch
3. Observe: El worker debe procesar todos los mini-batches sin error
```

**Expected Output:**
```
[W0] TRAIN_START — 1 épocas  n_train=50000  rank=0/1  mode=end_to_end
[W0] Época 1 — 50000 ejemplos
[W0] [END-TO-END] Procesando 50000 ejemplos en 163 mini-batches (size=307)...
[W0]   [END-TO-END] Batch 1/163  size=307
[W0]   [END-TO-END] Batch 33/163  size=307
[W0]   [END-TO-END] Batch 66/163  size=307
[W0]   [END-TO-END] Batch 99/163  size=307
[W0]   [END-TO-END] Batch 132/163  size=307
[W0]  loss=XXXX.XXXX  acc=XX.XX%  (XX.XXXs)
[W0] GRADIENTS enviados
```

**Success Indicators:**
✅ No RuntimeError sobre grad_fn
✅ All mini-batches complete
✅ Final loss/accuracy printed  
✅ Gradients sent successfully

## Files Modified

| File | Line | Change |
|------|------|--------|
| `Distributed/worker_node.py` | 625-629 | Enable CNN gradients at start of END-TO-END |
| `Distributed/worker_node.py` | 689 | Remove redundant `requires_grad_(False)` |
| `Distributed/worker_node.py` | 745-753 | Restore CNN eval mode and disable gradients |

## Invariants Maintained

✅ **[R1.3]** Precomputed mode never trains CNN (CNN restored to eval/requires_grad=False)
✅ **[R2.1]** END-TO-END mode always trains CNN (enabled during E2E)
✅ **[R2.4]** CNN parameters included in END-TO-END gradients (now with valid computation graph)
✅ **Protocol**: No changes (backward compatible)

## Key Insight

PyTorch's `torch.enable_grad()` context manager enables operations to be traced for backprop,
but it doesn't automatically make parameters trainable. Those must have `requires_grad=True`
BEFORE entering the context. This is the critical distinction:

- `torch.enable_grad()`: Enables tracing (builds computation graph)
- `param.requires_grad_(True)`: Makes parameter trainable (participates in backprop)

**Both must be true for backprop to work.**
