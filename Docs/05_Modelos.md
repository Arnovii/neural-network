# Modelos: CNN Extractor y MLP Classifier

## CNN Extractor: Arquitecturas Soportadas

### ResNet-18 (Recomendado para Producción)

**Archivo**: `Model/cnn_extractor.py` → `_build("resnet18", pretrained=True)`

```python
import torchvision.models as tvm

model = tvm.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Identity()  # Remove classification head
```

**Especificaciones**:

| Atributo | Valor |
|---|---|
| Feature dimension | 512 |
| Parámetros totales | 11.7M (89% en capas convolicionales) |
| Tamaño state_dict | ~45 MB en float32 |
| Pesos | ImageNet1K_V1 (preentrenados) |
| Velocidad (CPU) | ~100ms per batch (64 imágenes) |
| Velocidad (GPU) | ~10-15ms per batch |

**Arquitectura**:
```
Input: (N, 3, 224, 224)
  ↓
conv1 + BN + ReLU + MaxPool
  ↓
res_layer2 (64 ch)  – 2 bloques residuales
res_layer3 (128 ch) – 2 bloques residuales
res_layer4 (256 ch) – 2 bloques residuales
res_layer5 (512 ch) – 2 bloques residuales
  ↓
AdaptiveAvgPool(1,1)
  ↓
Output: (N, 512)
```

**Ventajas**:
- ✅ Pesos preentrenados → features relevantes desde inicio
- ✅ Convergencia rápida (transfer learning)
- ✅ Balance complejidad/velocidad
- ✅ Well-established en la comunidad

**Desventajas**:
- ⚠️ Descarga ~45 MB de pesos en primera ejecución
- ⚠️ SGD puro sin momentum en backprop hace convergencia lenta

**Estado en el Sistema**:
- ❌ **CONGELADA (requires_grad=False)**: No recibe gradientes en backward
- ❌ Solo forward pass para extracción de features
- ✅ Se resincroniza con CNN global (PS) cada REQUEST_PARAMS con pesos promediados
- ✅ Dinámica: CNN global se entrena distribuida via SimpleCNN + Async-FedAvg

### SimpleCNN (Para Experimentación)

**Archivo**: `Model/cnn_extractor.py` → `_SimpleCNN`

```python
class _SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        
        self.features = nn.Sequential(
            _block(3, 64),           # (N,64,112,112)
            _block(64, 128),         # (N,128,56,56)
            _block(128, 256),        # (N,256,28,28)
            nn.AdaptiveAvgPool2d((1, 1)),  # (N,256,1,1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
        )
```

**Especificaciones**:

| Atributo | Valor |
|---|---|
| Feature dimension | 512 |
| Parámetros totales | 1.58M |
| Tamaño state_dict | ~6 MB en float32 |
| Pesos | Random (Kaiming init) |
| Velocidad (CPU) | ~12ms per batch |
| Velocidad (GPU) | ~2ms per batch |

**Ventajas**:
- ✅ Muy rápida (3-5x más rápida que ResNet-18)
- ✅ Pequeña (7x menos parámetros)
- ✅ Sin dependencias externas
- ✅ Ideal para testing/debugging

**Desventajas**:
- ⚠️ Sin preentrenamiento → convergencia lenta
- ⚠️ Features iniciales aleatorias → ruido puro primeros centenares de batches
- ⚠️ Peor generalización
- ⚠️ SGD puro sin momentum + resincronización = convergencia muy lenta

**Estado en el Sistema**:
- ✅ **ENTRENABLE (requires_grad=True)**: Recibe gradientes en backward
- ✅ CNN se actualiza localmente con SGD
- ⚠️ Cambios locales NO PERSISTEN: se resincroniza con CNN global (PS) cada REQUEST_PARAMS
- ⚠️ Dinámica: CNN local se descarta periódicamente y se sobrescribe con promedio global
- ⚠️ **NO RECOMENDADA**: Features aleatorias + resincronización = learning inestable

---

## Interfaz Pública: CNNExtractor

```python
cnn = CNNExtractor(
    arch="resnet18",           # "resnet18" o "simple"
    pretrained=True,           # Si True: ImageNet1K_V1
    device="cuda",             # Device de PyTorch
    seed=None                  # None = aleatorio, int = reproducible
)
```

### Métodos Clave

#### Forward Pass (Extracción de Features)

```python
# En PS (evaluación)
features = cnn._model(images)  # (N, 512)

# En Workers (training loop)
with torch.inference_mode():
    features = cnn._model(images)
```

#### Serialización TCP

```python
# Obtener pesos para enviar
weights_bytes = cnn._get_weights_bytes()  # Retorna bytes (torch.save format)

# Cargar pesos recibidos
cnn.load_weights_from_bytes(weights_bytes)
```

**Proceso internamente**:
```
state_dict → BytesIO buffer → torch.save() → bytes
          ↓
bytes → BytesIO buffer → torch.load() → state_dict
```


---

## MLP Classifier: Arquitectura y Entrenamiento

**Archivo**: `Model/mlp_pytorch.py`

```python
class MLPPyTorch(nn.Module):
    def __init__(self, feature_dim, hidden1, hidden2, n_classes=1000):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_classes)
        self.relu = nn.ReLU()
        self._init_weights()
```

### Especificaciones

| Componente | Dimensión | Parámetros | Tamaño |
|---|---|---|---|
| fc1 | (512, 1024) | 524K | 2 MB |
| fc2 | (1024, 512) | 524K | 2 MB |
| fc3 | (512, 1000) | 512K | 2 MB |
| **Total** | - | 1.56M | 6 MB |

### Inicialización: Kaiming Uniform (He)

```python
def _init_weights(self):
    for layer in (self.fc1, self.fc2, self.fc3):
        nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(layer.bias)
```

**¿Por qué He Initialization?**

```
ReLU activation: Esperado E[h] = 0

Si usamos random normal:
  → 50% de las activaciones se ponen a cero (ReLU negatives)
  → Varianza decrece por layers

He init garantiza:
  → Var(h_i) ≈ constante por layer
  → Primeros logits tienen distribución razonable
  → Loss ≈ log(1000) ≈ 6.9 (no 0, no NaN)
  → Accuracy ≈ 0.1% (aleatorio)
```

**Comparación**:

| Método | Loss Initial | Acc Initial | Estabilidad |
|---|---|---|---|
| Ceros | ∞ | - | ✗ Colapso |
| Random Uniform | 6.5-7.5 | 0-0.5% | ⚠️ Inestable |
| Xavier | 6.8-7.0 | 0.1-0.2% | ✓ OK |
| **Kaiming (He)** | **6.9** | **0.1%** | **✓ Óptimo** |

---

## Interacción CNN ↔ MLP

### Flujo E2E Simplificado (SimpleCNN)

```
Input Images: (64, 3, 224, 224)
       ↓
┌──────────────────────────────┐
│ CNN.forward() (SimpleCNN)    │  Extrae features (entrenable)
│ (se entrena localmente,      │  Backward durante accum_steps
│  se resincroniza globalmente)│  Cambios locales no persisten
└──────┬───────────────────────┘
       │
Features: (64, 512)
       ↓
┌──────────────────┐
│ MLP.forward()    │  Clasifica features
│ (entrenamientos) │  Se optimizan parámetros
│ (resincronizados)│  Se resincroniza del PS
└──────┬───────────┘
       │
Logits: (64, 1000)
       ↓
   CrossEntropyLoss
       ↓
Loss: scalar (≈6.9-0.1 durante entrenamiento)
       ↓
  Backward()
       ↓
Gradients en MLP params (fc1, fc2, fc3) ✓ SE COMPUTAN Y SE USAN
Gradients en CNN params (120+ capas) ✓ SE COMPUTAN Y SE USAN (SimpleCNN)
```

### Flujo MLP-Only (ResNet-18)

```
Input Images: (64, 3, 224, 224)
       ↓
┌──────────────────────────────┐
│ CNN.forward() (ResNet-18)    │  Extrae features (congelada)
│ (congelada, requires_grad=F) │  Solo forward, sin gradientes
│ (se resincroniza globalmente)│  
└──────┬───────────────────────┘
       │
Features: (64, 512)
       ↓
┌──────────────────┐
│ MLP.forward()    │  Clasifica features
│ (entrenamientos) │  Se optimizan parámetros
│ (resincronizados)│  Se resincroniza del PS
└──────┬───────────┘
       │
Logits: (64, 1000)
       ↓
   CrossEntropyLoss
       ↓
Loss: scalar (≈6.9-0.1 durante entrenamiento)
       ↓
  Backward()
       ↓
Gradients en MLP params (fc1, fc2, fc3) ✓ SE COMPUTAN Y SE USAN
Gradients en CNN params (120+ capas) ✗ NO se computan (congelada)
```

### Dinámicas de Entrenamiento CNN: Local vs Global (SimpleCNN)

**SimpleCNN SÍ se entrena en el código, pero con dinámicas especiales:**

| Aspecto | Realidad |
|---|---|
| **Backward E2E** | Gradientes llegan a CNN (120+ capas de SimpleCNN) |
| **SGD local** | CNN se actualiza: `cnn_param.data -= lr * cnn_param.grad` |
| **Duración** | Cambios CNN locales duran accum_steps batches (ej: 5 batches) |
| **Resincronización** | Cada REQUEST_PARAMS, CNN local se SOBRESCRIBE con CNN global del PS |
| **Persistencia** | CNN cambios locales se DESCARTAN cuando sincroniza (NO persisten) |
| **Global** | PS recibe CNN de cada Worker, la promedia con Async-FedAvg → CNN global SÍ aprende |
| **Comunicación** | CNN 6MB + MLP 4.5MB = ~11MB total intercambiados (AMBAS se sincronizan) |

**Dinámica Especial SimpleCNN**:
- A nivel **local**: CNN aparenta estar congelada (cambios no persisten entre ciclos)
- A nivel **global**: CNN entrena su entrenamiento distribuido vía Async-FedAvg
- **Efecto**: CNN global converge lentamente (no hay momentum persistente a nivel local)

### Dinámicas de Entrenamiento CNN: ResNet-18 (Congelada)

**ResNet-18 NO se entrena - está permanentemente congelada:**

| Aspecto | Realidad |
|---|---|
| **Backward** | Gradientes se computan pero NO se propaguen a CNN (grad_fn interrumpido) |
| **SGD local** | CNN NO se actualiza (requires_grad=False) |
| **Congelación** | Permanente desde __init__, nunca cambia |
| **Resincronización** | CNN se recibe del PS pero no cambia (porque nunca cambió localmente) |
| **Persistencia** | No hay cambios locales que persistir |
| **Global** | PS solo actualiza MLP, CNN permanece estática (no hay Async-FedAvg de CNN para ResNet-18) |
| **Comunicación** | CNN 45MB + MLP 4.5MB = ~50MB total intercambiados (ambas se sincronizan por completitud) |

**Dinámica Especial ResNet-18**:
- CNN congelada permanentemente → transfer learning puro
- Solo MLP se entrena → clasificador aprendible sobre features fijas
- **Efecto**: Convergencia más rápida que SimpleCNN (features preentrenadas), pero limitada (CNN no se adapta)

---

## Cómo se Entrena Realmente (SimpleCNN)

```
Forward Pass:
  X → CNN → features (512D)
  features → MLP → logits (1000D)
  logits → Loss

Backward Pass:
  dLoss/d(logits) → MLP gradient ✓ (se computa y se ACTUALIZA MLP)
  d(logits)/d(features) → gradient ✓ (se propaga a CNN)
  d(features)/dCNN_weights → CNN gradient ✓ (se computa y se ACTUALIZA CNN localmente)

SGD Local (ambas redes SE ACTUALIZAN):
  mlp.parameters() -= lr * mlp.grad  ✓ MLP SE ACTUALIZA LOCALMENTE
  cnn.parameters() -= lr * cnn.grad  ✓ CNN SE ACTUALIZA LOCALMENTE

Sincronización (ambas se RESINCRODNIZAN del PS):
  mlp_global = recv(PARAMS).mlp_state  → MLP se SOBRESCRIBE
  cnn_global = recv(PARAMS).cnn_state  → CNN se SOBRESCRIBE (cambios locales se pierden)
```

## Cómo se Entrena Realmente (ResNet-18)

```
Forward Pass:
  X → CNN (requires_grad=False) → features (512D)
  features → MLP → logits (1000D)
  logits → Loss

Backward Pass:
  dLoss/d(logits) → MLP gradient ✓ (se computa y se ACTUALIZA MLP)
  d(logits)/d(features) → gradient ✓ (se propaga, pero CNN congelada)
  d(features)/dCNN_weights → CNN gradient ✗ NO se computa (congelada, no participa en backward)

SGD Local (solo MLP SE ACTUALIZA):
  mlp.parameters() -= lr * mlp.grad  ✓ MLP SE ACTUALIZA LOCALMENTE
  cnn.parameters() ← NO SE ACTUALIZAN (congelada)

Sincronización (solo MLP se RESINCRONIZA del PS):
  mlp_global = recv(PARAMS).mlp_state  → MLP se SOBRESCRIBE
  cnn_global = recv(PARAMS).cnn_state  → CNN se SOBRESCRIBE pero no ha cambiado (fue congelada localmente)
```

### Estado Dict Intercambiado

```python
# En MLPPyTorch.state_dict_numpy()
{
    "fc1.weight": (1024, 512) array,
    "fc1.bias": (1024,) array,
    "fc2.weight": (512, 1024) array,
    "fc2.bias": (512,) array,
    "fc3.weight": (1000, 512) array,
    "fc3.bias": (1000,) array,
}

# Formato exacto de PyTorch → compatible con .load_state_dict()
```

---

## Ejemplo: Forward + Backward en un Batch

```python
# Datos
X = torch.randn(64, 3, 224, 224)  # Imágenes
Y = torch.randint(0, 1000, (64,))  # Labels

# Forward E2E
features = cnn._model(X)           # (64, 512)
logits = mlp(features)             # (64, 1000)
loss = F.cross_entropy(logits, Y)  # scalar ≈ 6.9

# Backward
loss.backward()

# Gradientes generados:
#   mlp.fc1.weight.grad: (1024, 512)
#   mlp.fc1.bias.grad: (1024,)
#   mlp.fc2.weight.grad: (512, 1024)
#   mlp.fc2.bias.grad: (512,)
#   mlp.fc3.weight.grad: (1000, 512)
#   mlp.fc3.bias.grad: (1000,)
#
#   cnn._model[...].weight.grad: (many) ← SÍ se USAN para actualizar CNN

# SGD local (AMBAS redes se actualizan)
lr = 0.001
with torch.no_grad():
    for name, param in mlp.named_parameters():
        param.data -= lr * param.grad  # Update MLP ✓

    for param in cnn._model.parameters():
        if param.grad is not None:
            param.data -= lr * param.grad  # Update CNN ✓ (cambios duran accum_steps batches)

# Resultado: MLP tiene pesos nuevos, CNN tiene pesos nuevos (localmente)
# Cambios CNN persisten EN ESTE CICLO (accum_steps batches), pero se descartan
# después de enviar UPDATES y recibir REQUEST_PARAMS (se resincroniza con CNN global del PS)
```

---

## Configuración Flexible

### Arquitectura MLP Ajustable

```python
# Default (balanced)
MLP(512, 1024, 512, 1000)

# Pequeña (rápida)
MLP(512, 512, 256, 1000)

# Grande (más expresiva)
MLP(512, 2048, 1024, 1000)

# Muy pequeña (tiny)
MLP(512, 256, 128, 1000)
```

Cambiar en:
- `ps_imagenet.py --hidden1 2048 --hidden2 1024`
- `ps_gui_imagenet.py` → GUI input fields
- `worker_imagenet.py --hidden1 2048 --hidden2 1024` (debe coincidir con PS)

### Impacto en Performance

| Config | Params | Train Time/Iter | Memory | Accuracy Potential |
|---|---|---|---|---|
| (512,256,128) | 0.4M | 5ms | 20MB | Medium |
| (512,1024,512) | 1.6M | 20ms | 50MB | Good |
| (512,2048,1024) | 4.1M | 50ms | 150MB | VeryGood |

