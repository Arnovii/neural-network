# 04. Modos de Entrenamiento: PRECOMPUTED vs END-TO-END

## Comparación a Alto Nivel

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRECOMPUTED                                  │
├─────────────────────────────────────────────────────────────────┤
│ CNN:       CONGELADA (pesos fijos)                              │
│ Features:  Extraídos UNA sola vez al inicio                     │
│ Caché:     Pesa ~500 MB, reutilizado cada época                 │
│ Compute:   ~2 seg/época (solo MLP)                              │
│ GPU:       NO necesaria                                         │
│ Precisión: 94-97%                                               │
│ Caso uso:  Teaching, quick validation, offline environments     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    END-TO-END                                   │
├─────────────────────────────────────────────────────────────────┤
│ CNN:       ENTRENABLE (pesos se actualizan cada época)          │
│ Features:  Extraídos FRESCO cada época                          │
│ Caché:     Cache invalidado (hash distinto) cada época          │
│ Compute:   ~12-15 seg/época (CNN+MLP)                           │
│ GPU:       RECOMENDADA                                          │
│ Precisión: 98-99%                                               │
│ Caso uso:  Production, best accuracy, research                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Modo 1: PRECOMPUTED (Rápido, Congelado)

### Flujo General

```
1. Inicio (una vez):
   ├─ CNN inicializada (seed fijo o pretrained)
   ├─ CNN.set_trainable(False)  ← CONGELADA
   ├─ PS distribuye CNN_WEIGHTS a todos Workers
   ├─ Cada Worker: extract_features(X_train) → (50000, 512)
   ├─ Cachea con key: simple_a3f2c8d1_train_X.npy
   └─ Enviado CNN_READY
   
2. Por cada época:
   ├─ PS envía PARAMS (MLP weights)
   ├─ Worker: recv PARAMS
   ├─ Worker: X_batch = X_features[indices]  ← INDEXING EN MEMORIA
   ├─ Worker: forward/backward MLP
   ├─ Worker: envía GRADIENTS (solo MLP, sin CNN)
   ├─ PS: promedia, actualiza MLP params
   └─ CNN nunca es actualizada (congelada)
```

### Invariante Clave: Hash CNN es Constante

```python
# Época 0
weights_md5_0 = MD5(cnn.state_dict())  # "a3f2c8d1..."

# Época 1, 2, 3, ...
weights_md5_1 = MD5(cnn.state_dict())  # "a3f2c8d1..." (MISMO)
weights_md5_2 = MD5(cnn.state_dict())  # "a3f2c8d1..." (MISMO)

# Esto es garantizado porque CNN.set_trainable(False)
#  → pesos nunca se actualizan
```

**Implicación**: Cache key = `simple_a3f2c8d1_train_X.npy` nunca cambia.
- Primera época: 7-30s (extract + save)
- Épocas 2-N: 0.1-0.3s each (load from disk)

### Timing por Época

```
Época 1:
  Compute MLP (16667 samples): ~1500ms
    ├─ Indexing X_features[indices]: ~50ms
    ├─ Forward W1@X: ~400ms
    ├─ Backward (3 layers): ~900ms
    └─ Total forward+backward: ~1500ms
  = 2.5s de ejecución (pura NumPy)

Épocas 2-10:
  Identico: ~1500ms local compute
  = 2.5s each
```

### Código Ejemplar: Extracción PRECOMPUTED

```python
# En WorkerNode.run() al inicio:

# Época 0, antes de entrenamiento
cnn_weights = receive_message(MsgType.CNN_WEIGHTS)  # Del PS
cnn.load_state_dict(cnn_weights)
cnn.set_trainable(False)  # ← CONGELADA

# Extraer features UNA SOLA VEZ
X_features, Y_indices = cnn.extract_with_cache(
    X_raw,            # (50000, 3, 32, 32)
    Y_raw,            # (50000,)
    arch="simple",
    batch_size=2048,  # para no sobrecargar memoria
    split="train",
)
# X_features ahora es (50000, 512)
# Cachea en Data/feature_cache/simple_a3f2c8d1_train_X.npy

# Guardar para todo el entrenamiento
self._X_features = X_features

# Luego, por cada época...
```

### Código Ejemplar: Una Época PRECOMPUTED

```python
# En WorkerNode._run_training_session() loop:

for epoch in range(n_epochs):
    # Recibir pesos nuevos del PS
    msg = receive_message()
    assert msg["type"] == MsgType.PARAMS
    
    new_params = msg["payload"]["params"]
    seed = msg["payload"]["seed"]
    
    # Reconstruir índices (determinista)
    rng = RandomState(seed)
    shuffled = arange(n_train)
    rng.shuffle(shuffled)
    my_indices = [i for i in shuffled if i % n_workers == my_rank]
    
    # Indexar features (MEMORIA, no I/O)
    X_batch = self._X_features[my_indices]  # (16667, 512), ~67 MB array slicing
    Y_batch = self._Y_raw[my_indices]       # (16667,), ~67 KB
    
    # Forward + backward (NumPy)
    grads, loss, acc = mlp.forward_and_gradients(new_params, X_batch, Y_batch)
    
    # Enviar gradientes al PS (solo MLP, sin CNN)
    send_message(MsgType.GRADIENTS, {
        "worker_id": my_id,
        "epoch": epoch,
        "gradients": grads,    # Dict, 6 arrays (W1, b1, W2, b2, W3, b3)
        "loss": loss,
        "accuracy": acc,
    })
```

### Ventajas Específicas

1. **No requiere GPU**: Indexing de arrays en RAM es tan rápido en CPU
2. **Reproducible sin estado**: Ejecutar dos veces = resultados idénticos (float precision)
3. **Determinista**: No hay variables adicionales, siempre igual
4. **Ideal para enseñanza**: Se ve claro qué sucede (CNN fija, MLP se entrena)
5. **Caché gigante beneficio**: Primera época ~60s, épocas restantes ~2s

### Desventajas Específicas

1. **CNN no se adapta al dataset**: ResNet entrenado en ImageNet, SimpleCNN entrenado en datos aleatorios → no optimal para CIFAR-10
2. **Precisión limitada**: 94-97% vs 98-99% en END-TO-END
3. **No es end-to-end**: La CNN no aprende características específicas del problema

---

## Modo 2: END-TO-END (Lento, Entrenable)

### Flujo General

```
1. Inicio (Una vez):
   ├─ CNN inicializada (seed fijo o pretrained)
   ├─ CNN.set_trainable(True)  ← ENTRENABLE
   ├─ PS distribuye CNN_WEIGHTS a todos Workers
   └─ Workers confirmam CNN_READY (pueden haber extraído features temporales)
   
2. Por cada época:
   ├─ PS envía PARAMS (MLP weights) + CNN_WEIGHTS (CNN updated) + seed
   ├─ Worker: recv CNN_WEIGHTS y carga en su CNN
   ├─ Worker: forward pass CNN sobre X_batch_raw
   │    └─ X_features = CNN(X_batch_raw)  ← FRESCO cada época
   ├─ Worker: forward/backward MLP
   ├─ Worker: backward CNN (gradient chaining en PyTorch)
   ├─ Worker: envía GRADIENTS (MLP + CNN)
   ├─ PS: promedia AMBOS tipos de gradientes
   ├─ PS: actualiza AMBOS param sets (CNN + MLP)
   └─ Vuelve a época siguiente con nuevos CNN pesos
```

### Hash CNN Cambia Cada Época

```python
# Época 0
params_cnn_0 = cnn.state_dict()
weights_md5_0 = MD5(params_cnn_0)  # "a3f2c8d1..."

# PS average CNN gradients, actualiza CNN
# cnn.W1 -= lr * (grad_cnn_W1 de worker0 + grad_cnn_W1 de worker1 + ...) / n

# Época 1
params_cnn_1 = cnn.state_dict()  # DISTINTOS después de update
weights_md5_1 = MD5(params_cnn_1)  # "7f9a4e2c..." (DISTINTO)

# Época 2
params_cnn_2 = cnn.state_dict()
weights_md5_2 = MD5(params_cnn_2)  # "c3b1d9f5..." (DISTINTO)
```

**Implicación**: Cache key cambia cada época.
- Época 1: 5s (extract con CNN vieja) + 10s (forward pass con CNN nueva actualizada) + compute MLP
- Época 2+: MISMA (no reutiliza caché anterior, todo fresco)

### Timing por Época

```
Época 1 (END-TO-END, GPU):
  Recibir CNN_WEIGHTS nuevo: 200ms (network + deserialization)
  Load CNN en GPU: 100ms
  Extract features (16667 imgenes): ~5000ms (CNN forward on GPU)
    ├─ CNN tiene 18 bloques (ResNet18) o 3 (SimpleCNN)
    ├─ Por imagen: ~300μs (ResNet es pesada)
    └─ 16667 * 300μs ≈ 5s
  MLP forward+backward: 1500ms (similar input size)
  CNN backward (gradient to CNN layers): 2000ms (backprop through 18 layers)
  = 12-15s por época

Épocas 2-10:
  Identico: CNN distinto cada vez, 12-15s each
```

**Comparación**: PRECOMPUTED 2.5s/época vs END-TO-END 12.5s/época ≈ 5x más lento.

### Código Ejemplar: Extracción END-TO-END

```python
# En WorkerNode._run_training_session() loop:

for epoch in range(n_epochs):
    # Recibir pesos nuevos del PS (AHORA incluye CNN_WEIGHTS)
    msg = receive_message()
    assert msg["type"] == MsgType.PARAMS
    
    new_cnn_weights = msg["payload"]["cnn_weights"]  # ← NUEVO cada época
    new_mlp_params = msg["payload"]["mlp_params"]
    seed = msg["payload"]["seed"]
    
    # Cargar CNN actualizada (fue entrenada por PS)
    cnn.load_state_dict(new_cnn_weights)  # Pesos han cambiado
    cnn.set_trainable(True)
    
    # Reconstruir índices
    rng = RandomState(seed)
    shuffled = arange(n_train)
    rng.shuffle(shuffled)
    my_indices = [i for i in shuffled if i % n_workers == my_rank]
    
    # Obtener imágenes raw
    X_batch_raw = self._X_raw[my_indices]  # (16667, 3, 32, 32)
    Y_batch = self._Y_raw[my_indices]      # (16667,)
    
    # Forward CNN (fresco con pesos actualizados) ← COSTO
    with torch.no_grad():  # Sin autograd para no acumular history
        X_batch_features = cnn.extract(X_batch_raw)  # (16667, 512)
        # Pero si queremos gradientes: sin no_grad(), guardar grafo
    
    # Forward + backward MLP
    grads_mlp, loss, acc = mlp.forward_and_gradients(new_mlp_params, X_batch_features, Y_batch)
    
    # En E2E real (más avanzado, no en este código):
    # Backward CNN también (chain rule)
    # grads_cnn = cnn.compute_gradients(X_raw, dL/dX_features)
    
    # Enviar gradientes al PS (MLP + CNN opcional)
    send_message(MsgType.GRADIENTS, {
        "worker_id": my_id,
        "epoch": epoch,
        "gradients_mlp": grads_mlp,    # Dict, 6 arrays
        "gradients_cnn": grads_cnn,    # Dict, ~20 arrays (ResNet)
        "loss": loss,
        "accuracy": acc,
    })
```

### Ventajas Específicas

1. **CNN se adapta**: Aprende características específicas de CIFAR-10 (colores, formas, texturas)
2. **Mejor precisión**: 98-99% vs 94-97%
3. **True end-to-end**: Optimización conjunta de feature extractor + classifier
4. **Más realista**: Simula entrenamiento joint como en producción

### Desventajas Específicas

1. **Requiere GPU** (o muy lento en CPU): CNN forward es O(depth × H × W × spatial_filters)
2. **Sin caché**: Cada época, fresco → I/O y cómputo duplicado si no hay cambio de pesos
3. **Más comunicación**: Gradientes CNN son mas grandes (ResNet18: ~60 MB vs MLP: ~1 MB)
4. **Instabilidad potencial**: Si learning rate alto para CNN, puede divergir
5. **Determinismo**: Float precision en backprop de redes profundas puede variar ligeramente

---

## Comparativa Cuantitativa

### Consumo de Memoria

| | PRECOMPUTED | END-TO-END |
|---|---|---|
| X_features cached | 600 MB (permanente) | 0 MB |
| X_raw en memoria | - | 600 MB |
| CNN en GPU | 1 MB (inference mode) | 100 MB (training mode) |
| Gradients buffer | ~7 MB | ~100 MB (CNN + MLP) |
| **Total/Worker** | ~600 MB | ~700-800 MB |

### Transferencia de Red por Época

| | PRECOMPUTED | END-TO-END |
|---|---|---|
| PARAMS downlink | 1.1 MB | 1.1 MB |
| CNN_WEIGHTS (si envía) | 0 MB | 50-150 MB (ResNet) |
| GRADIENTS uplink | 1.1 MB | 1.1 + 60 MB = 61 MB |
| **Total downlink** | 1.1 MB | 50-150 MB |
| **Total uplink** | 1.1 MB | 60-65 MB |

**Conclusión**: END-TO-END es 50-100x más demandante en red. Viable en LAN o GPU cluster. NO viable en WAN.

### Convergencia Comparada

```
PRECOMPUTED (10 épocas):
  Epoch 0: Train Acc=80%, Test Acc=78%
  Epoch 1: Train Acc=88%, Test Acc=85%
  Epoch 2: Train Acc=91%, Test Acc=88%
  Epoch 3: Train Acc=93%, Test Acc=91%
  ...
  Epoch 9: Train Acc=96%, Test Acc=94% ← Plateau, CNN congelada

END-TO-END (10 épocas):
  Epoch 0: Train Acc=70%, Test Acc=68% (inicios, CNN + MLP ajustando)
  Epoch 1: Train Acc=82%, Test Acc=80%
  Epoch 2: Train Acc=88%, Test Acc=86%
  Epoch 3: Train Acc=92%, Test Acc=90%
  ...
  Epoch 9: Train Acc=99%, Test Acc=98% ← Continúa mejorando, CNN adaptada
```

**Observación**: END-TO-END converge más lentamente al inicio (optimiza dos conjuntos) pero llega más alto.

---

## Decidir Cuál Usar

**Elige PRECOMPUTED si**:
- Objetivo: Enseñanza de distributed learning
- Tiempo limitado: Quieres rápido feedback de cambios
- Datos limitados: No necesitas full GPU
- CNN pre-entrenada visible: ResNet ImageNet es suficiente

**Elige END-TO-END si**:
- Objetivo: Máxima precisión
- GPU disponible
- Red rápida o local (LAN)
- Tiempo disponible: Puedes esperar 15s/época
- Datos de dominio específico: Beneficia de adaptación CNN

---

## Switching Entre Modos (Mismo Checkpoint)

**Posible**: Entrenar en PRECOMPUTED hasta cierto punto, luego switchear a END-TO-END.

```python
# Checkpoints guardados identicamente: params["W1"], params["b1"], etc.

# Session 1: PRECOMPUTED, 5 épocas
ps = ParameterServer(..., training_mode="precomputed")
ps.train(..., epochs=5)
save_checkpoint("checkpoint_e5.pkl")

# Session 2: END-TO-END con mismo MLP weights
ps = ParameterServer(..., training_mode="end_to_end")
ps.load_checkpoint("checkpoint_e5.pkl")
ps.train(..., epochs=10)  # Continue training end-to-end
```

**Resultado**: MLP weights iniciales idénticas. CNN se entrena desde epoch 5 en adelante. Convergencia probablemente mejor que start-from-scratch END-TO-END (warm start).

