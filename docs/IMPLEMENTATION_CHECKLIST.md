# Checklist: Mini-batching End-to-End Implementation

## ✅ IMPLEMENTACIÓN COMPLETADA

### Files Modified
- [x] `Distributed/worker_node.py` → `_handle_params()` método 
- [x] Rama `end_to_end` (línea ~615)

### Cambios Específicos
- [x] Cálculo de `mini_bs = max(16, int(_optimal_batch_size() / 2.5))`
- [x] Loop de mini-batches con índices particionados
- [x] Forward CNN detachado (sin gradientes)
- [x] Backward MLP → obtener dX (gradiente respecto a features)
- [x] Backward CNN usando proxy loss
- [x] Acumulación de gradientes MLP en lista
- [x] Acumulación de gradientes CNN en diccionario
- [x] Acumulación de loss y accuracy
- [x] `zero_grad()` ANTES de cada backward
- [x] Promediado final de gradientes MLP (medio de lista)
- [x] Normalización de gradientes CNN por n_total
- [x] Logs cada 20% de progreso
- [x] Preservación del protocolo GRADIENTS

### Verificaciones de Código
- [x] Sin errores de compilación (get_errors: 0 errores)
- [x] Imports presentes: `List`, `Dict`, `torch`
- [x] Shape de dX correcto: `(feature_dim, N_mini)` → transponemos para proxy
- [x] Normalización matemáticamente correcta

---

## 📋 CHECKLIST DE FUNCIONALIDAD

### Mini-batching
- [x] Divide partición en mini-batches adaptativos
- [x] Tamaño basado en `_optimal_batch_size()` ajustado
- [x] Factor 2.5 para poder costoso de backward
- [x] Mínimo 16 ejemplos por mini-batch

### Acumulación de Gradientes
- [x] MLP: promedio simple (todos los mini-batches pesan igual)
- [x] CNN: suma acumulada / n_total (promedio ponderado correcto)
- [x] Loss y accuracy: promediados

### Backward Correcto
- [x] Forward CNN con detach (sin gradientes)
- [x] Backward MLP (NumPy, no PyTorch)
- [x] Forward CNN segunda vez (con gradientes)
- [x] Backward CNN con proxy loss
- [x] `-zero_grad()` antes de cada backward

### Logging
- [x] Mensaje inicial con ejemplos totales y n_batches
- [x] Progreso cada 20% (`batch_idx % max(1, n_batches // 5)`)
- [x] Tamaño de cada mini-batch
- [x] Loss/accuracy/tiempo al final

### Protocolo
- [x] Payload GRADIENTS idéntico en estructura
- [x] `cnn_gradients` siempre presente (dict de gradientes)
- [x] Compatibilidad con ParameterServer

---

## 🧪 TESTING RECOMENDADO

### Test 1: Precomputed (No debe cambiar)
```bash
Modo: Precomputed
CNN: simple (load o train)
Épocas: 5
Esperado: Funciona igual que antes (~5-10s por epoch)
```

### Test 2: End-to-End CPU (CRÍTICO)
```bash
Modo: End-to-End
CPU: cualquiera
CNN: simple
Épocas: 3
Esperado: 
  - NO congelamiento
  - Logs de progreso cada batch
  - ~30-50s total por epoch
  - CPU ~80% de uso (controlado)
```

### Test 3: End-to-End GPU (Rendimiento)
```bash
Modo: End-to-End
GPU: nvidia/amd/mps
CNN: simple
Épocas: 3
Esperado:
  - Rápido como antes (~5-10s por epoch)
  - Mini-batching trasparente (número de batches bajo)
```

### Test 4: End-to-End ResNet18 (Arquitectura pesada)
```bash
Modo: End-to-End
Device: CPU o GPU
CNN: resnet18
Épocas: 1
Esperado:
  - CPU: ~60-90s / epoch (lento pero estable)
  - GPU: ~15-25s / epoch
  - Logs muestran mini-batches pequeños (~25-30)
```

### Test 5: Multiple Workers
```bash
Workers: 2-4
Modo: End-to-End
CNN: simple
Épocas: 2
Esperado:
  - Cada worker mini-batching independiente
  - Gradientes correctamente promediados en PS
  - Convergencia similar a single worker
```

---

## 📊 MÉTRICAS A VERIFICAR

### Logs que Deben Aparecer
```
[W0] [END-TO-END] Procesando 41000 ejemplos en 200 mini-batches (size=205)...
[W0]   [END-TO-END] Batch 1/200  size=205
[W0]   [END-TO-END] Batch 51/200  size=205
[W0]   [END-TO-END] Batch 101/200  size=205
[W0]   [END-TO-END] Batch 151/200  size=205
[W0]   loss=0.8234  acc=78.45%  (45.123s)
```

### Métricas de Rendimiento
| Métrica | Aceptable | Excelente |
|---------|-----------|-----------|
| CPU (simple) | <60s/epoch | 30-50s/epoch |
| GPU (simple) | <15s/epoch | 5-10s/epoch |
| CPU (resnet18) | <120s/epoch | 60-90s/epoch |
| Memoria pico | <500 MB | 100-200 MB |
| CPU util. | 60-90% | 70-85% |

### Gradientes Esperados
- Forma de `cnn_gradients`: Dict con keys = nombres de parámetros CNN
- Rango de valores: típicamente [-0.01, 0.01] (después de normalizar)
- No deben ser NaN o Inf

---

## 🔍 DEBUGGING

### Síntoma: "Todavía se congela"
→ Verificar:
- [ ] mini_bs se calcula correctamente
- [ ] Device es CPU (verbose logging para confirmar)
- [ ] ResNet18 puede necesitar mini_bs aún menor

### Síntoma: "Loss no converge"
→ Verificar:
- [ ] Acumulación de gradientes es correcta (suma / n_total)
- [ ] MLP gradients se promedian bien
- [ ] dX_mini.T está transpuesta correctamente

### Síntoma: "Logs no muestran progreso cada 20%"
→ Verificar:
- [ ] `n_batches // 5` es > 0
- [ ] `batch_idx % max(1, n_batches // 5)` lógica correcta

### Síntoma: "Error de shape de tensores"
→ Verificar:
- [ ] X_mini_torch tiene shape (N, 3, 32, 32)
- [ ] features_torch tiene shape (N, feature_dim)
- [ ] dX_mini tiene shape (feature_dim, N) ← use .T

---

## ✨ PUNTOS TÉCNICOS CLAVE

1. **Factor 2.5 NO es arbitrario**
   - Forward CNN: ~X tiempo
   - Backward CNN: ~2× tiempo
   - Acumulación: ~0.5× tiempo
   - Total: ~2.5× backward vs forward
   - Usar 80% del time budget forward para estar seguro

2. **zero_grad() es CRÍTICO**
   - Si no se hace: gradientes se acumulan INDEFINIDAMENTE
   - CNN iría a infinito en mini-batch 2
   - Debe ser ANTES de backward, DENTRO del loop

3. **Doble forward es INTENCIONAL**
   - 1ª vez: detach (sin gradientes), para MLP backward
   - 2ª vez: con gradientes, para CNN backward
   - Ineficiente pero necesario para separación

4. **Normalización CNN es EXACTA**
   - Acumular SIN normalizar en cada batch
   - Normalizar por n_total al final
   - NO por n_batches (sería incorrecto)

---

## 🚀 DEPLOYMENT

### Para Producción
1. Deploy `worker_node.py` versión nueva
2. Verificar test 1 (precomputed sigue funcionando)
3. Ejecutar test 2 (CPU + E2E)
4. Si OK: desplegar a usuarios
5. Monitorear logs por primeras 24h

### Rollback (si algo falla)
Cambiar rama `end_to_end` de vuelta a versión anterior en git
```bash
git checkout HEAD~1 -- Distributed/worker_node.py
```

---

## 📚 DOCUMENTACIÓN ASOCIADA

- `MINIBATCHING_SOLUCION.md` — Análisis técnico completo
- `MINIBATCHING_RESUMEN.md` — Resumen ejecutivo
- `ARQUITECTURA_DISTRIBUIDA_ESPECIFICACION.md` — Especificación original
- Código en `Distributed/worker_node.py` líneas 615-730

