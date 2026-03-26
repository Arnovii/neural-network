# Resumen Ejecutivo: Mini-batching End-to-End

## ¿QUÉ SE CAMBIÓ?

### Ubicación
`Distributed/worker_node.py` → Método `_handle_params()` → Rama `end_to_end`

### Cambio Principal
**Antes:** 
```python
X_batch = self._X_raw[indices]  # Todo el batch
# Forward + backward de toda la partición de una vez
```

**Después:**
```python
mini_bs = max(16, int(_optimal_batch_size() / 2.5))
# Loop sobre mini-batches de tamaño adaptativo
# Forward/backward + acumulación de gradientes en cada mini-batch
```

---

## ¿POR QUÉ FUNCIONA?

1. **Batch size dinámico:** Usa `_optimal_batch_size()` (ya testeado) dividido entre 2.5
   - CPU simple: ~200-500 ejemplos por mini-batch
   - GPU: ~800-2000 ejemplos
   - ResNet18: ~25-128 ejemplos (más pequeño, arquitectura pesada)

2. **Factor 2.5:** Backward es 2-3× más caro que forward
   - Forward CNN extraction (original): 512-2048
   - Forward + Backward CNN+MLP (E2E): 512/2.5 ≈ 205

3. **Acumulación correcta:**
   - MLP gradients: promedio de mini-batches
   - CNN gradients: suma / total de ejemplos
   - Loss/Accuracy: promedio de mini-batches

---

## RESULTADOS ESPERADOS

### Antes (CONGELAMIENTO)
```
[W1] [END-TO-END] Forward/backward CNN+MLP...
[Se congela por 2-5 minutos]
```

### Después (PROGRESO VISIBLE)
```
[W1] [END-TO-END] Procesando 41000 ejemplos en 200 mini-batches (size=205)...
[W1]   [END-TO-END] Batch 1/200  size=205
[W1]   [END-TO-END] Batch 51/200  size=205
[W1]   [END-TO-END] Batch 101/200  size=205
[W1]   [END-TO-END] Batch 151/200  size=205
[W1]   loss=0.8234  acc=78.45%  (45.322s)
```

### Métricas
| Métrica | Antes | Después |
|---------|-------|---------|
| CPU (simple) | ❌ CONGELADO | ✅ 30-50s/epoch |
| GPU (simple) | ✅ 5-10s | ✅ 5-10s (igual) |
| CPU (resnet18) | ❌ CONGELADO | ✅ 60-90s/epoch |
| Memoria pico | 500-800 MB | 100-150 MB |

---

## VERIFICACIÓN RÁPIDA

### Logs a Buscar
✅ `[END-TO-END] Procesando X ejemplos en Y mini-batches`  
✅ Progreso cada 20% (Batch 1/200, Batch 51/200, ...)  
✅ NO congelamiento en CPU  
✅ `loss=...  acc=...` al final  

### Logs que NO Deberían Aparecer
❌ `RuntimeError` (error de validación)  
❌ Congelamiento sin mensajes  
❌ Errores de memoria  

### Cómo Probar
```bash
# Terminal 1: Parameter Server
python ps_gui.py

# Terminal 2: Worker
python worker.py

# Seleccionar en GUI:
# - Modo: End-to-End
# - Arquitectura: simple (para prueba rápida)
# Clic en "ENTRENAR"

# ESPERADO: CPU trabaja, progreso visible, ~30-50s por epoch
```

---

## INVARIANTES PRESERVADAS

✓ [R2.1] CNN actualizada cada época  
✓ [R2.2] Sin preentrenamiento en E2E  
✓ [R2.3] Features dinámicos (no cacheados)  
✓ [R2.4] `cnn_gradients` siempre en payload  
✓ [R2.5] Protocolo GRADIENTS idéntico  
✓ [R2.6] CNN entrenable  

---

## IMPACTO EN OTROS COMPONENTES

| Componente | Impacto |
|-----------|--------|
| ParameterServer | Ninguno (solo recibe GRADIENTS) |
| Precomputed mode | Ninguno (rama diferente) |
| MLP/CNNExtractor | Ninguno (no se modifican) |
| Protocolo | Ninguno (payload idéntico) |
| Multiple Workers | Ninguno (cada Worker mini-batching independiente) |

---

## PUNTO TÉCNICO CRÍTICO

Mini-batching requiere acumulación cuidadosa:

```python
# CORRECTO para CNN:
for batch_idx in range(n_batches):
    self._cnn._model.zero_grad()  # ← ANTES de backward
    loss_proxy.backward()
    accumulated_cnn_grads += param.grad  # ← Acumular
# Normalizar al final por n_total (no n_batches)
cnn_gradients = accumulated_cnn_grads / n_total
```

```python
# INCORRECTO:
accumulated_cnn_grads += param.grad / len(Y_mini)  # ← Divide entrada
# Luego normalizar por n_batches ← Doble normalización incorrecta
```

La implementación usa la versión CORRECTA.

---

## DOCUMENTACIÓN COMPLETA

Ver `MINIBATCHING_SOLUCION.md` para:
- Análisis completo de estrategias
- Matemática de acumulación
- Detalle de controles críticos
- Testing recomendado
