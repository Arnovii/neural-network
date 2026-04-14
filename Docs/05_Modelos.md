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

### SIMPLE CNN (ResNet-18 desde Cero)

**Archivo**: `Model/cnn_extractor.py` → `_ResNet18FromScratch` + `_BasicBlock`

La SIMPLE CNN es una arquitectura **ResNet-18 estándar construida desde cero** (sin pesos preentrenados). Utiliza bloques residuales básicos (_BasicBlock) organizados en 4 capas progresivas con skip connections fuertes, optimizada para entrenamiento E2E en Async-SGD distribuido con staleness.

```python
class _BasicBlock(nn.Module):
    """Bloque residual estándar de ResNet: Conv→BN→ReLU→Conv→BN + shortcut."""
    expansion = 1
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        # Conv(3×3, stride) + BN + ReLU
        # Conv(3×3) + BN (sin ReLU, se aplica após shortcut)
        # Shortcut: identidad o Conv(1×1) si stride > 1 o cambio de canales
        # Zero-init en BN final → bloque inicia como identidad

class _ResNet18FromScratch(nn.Module):
    def __init__(self):
        # Stem: Conv(3→64, 7×7, stride=2) + BN + ReLU + MaxPool(stride=2) → stride 4×
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 4 layers con 2 BasicBlocks cada uno en progresión: 64→128→256→512
        self.layer1 = self._make_layer(_BasicBlock, 64, 64, 2, stride=1)    # 56×56
        self.layer2 = self._make_layer(_BasicBlock, 64, 128, 2, stride=2)   # 28×28
        self.layer3 = self._make_layer(_BasicBlock, 128, 256, 2, stride=2)  # 14×14
        self.layer4 = self._make_layer(_BasicBlock, 256, 512, 2, stride=2)  # 7×7 [stride = 32×]
        
        # GAP + Dropout + Proyección lineal
        self.gap = AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.proj = nn.Linear(512, 512)  # → feature_dim = 512
    
    def _make_layer(self, block, in_ch, out_ch, blocks, stride):
        """Construir capa con N bloques residuales."""
        layers = [block(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(block(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)
```

**Especificaciones**:

| Atributo | Valor |
|---|---|
| Feature dimension | 512 |
| Parámetros totales | **11.7M** |
| Tamaño state_dict | **~45.8 MB** en float32 |
| Pesos | Random (Kaiming Normal init) |
| Velocidad (CPU) | ~100-120ms per batch (64 imágenes) |
| Velocidad (GPU) | ~15-20ms per batch |

**Arquitectura de Bloques Residuales**:

Cada `_BasicBlock(in_ch, out_ch, stride)` tiene (estructura estándar ResNet-18):
```
input
  ├─ Conv(3×3, stride) → BN → ReLU → Conv(3×3) → BN ─┐
  │                                                  (suma)
  └─ Shortcut (identidad o Conv 1×1) ────────────────┤
                                                     ↓
                                                    ReLU
                                                     ↓
                                                   output
```

**Zero-init BN final**: En cada bloque, el último BN inicia con scale=0 → el bloque es aproximadamente identidad al inicio → estabilidad de gradientes en primeros pasos.

**Comparativa: SIMPLE CNN (11.7M) vs Alternativas**:

| Aspecto | SIMPLE CNN (Actual) | SimpleCNN Anterior | ResNet-18 Preentrenada |
|---|---|---|---|
| **Parámetros** | 11.7M | 1.36M | 11.7M |
| **Pesos iniciales** | Random (Kaiming) | Random (Kaiming) | ImageNet1K_V1 |
| **requires_grad** | True (entrenable) | True (entrenable) | False (congelada) |
| **Stride efectivo** | 32× | 32× | 32× |
| **Skip connections** | Sí (_BasicBlock) | Sí (_ResBlockLite) | Sí (standard) |
| **Dropout** | 0.5 | 0.2 | Ninguno |
| **Bloque residual** | Estándar ResNet | Custom ligero | Estándar ResNet |
| **Velocidad CPU** | ~100ms/batch | ~12ms/batch | ~100ms/batch |
| **Capacidad** | Suficiente para ImageNet-1k | Insuficiente (~1.4K params/clase) | Suficiente (~11.7K params/clase) |
| **Convergencia inicialmente** | Lenta (sin pretrain) | Lenta (sin pretrain) | Rápida (transfer learning) |
| **Modo de uso** | E2E training (ambas redes) | E2E training (ambas redes) | MLP-only (CNN congelada) |

**Ventajas de SIMPLE CNN**:
- ✅ Arquitectura validada ResNet-18 standard → reproducible
- ✅ Suficiente capacidad para ImageNet-1k (11.7M params = 11.7K params/clase)
- ✅ Entrenable E2E desde cero → no requiere dataset preentrenamiento
- ✅ Skip connections robustos → manejan bien staleness en Async-SGD
- ✅ Comparable en velocidad a ResNet-18 preentrenada
- ✅ Ideal para testing/debugging y entrenamiento distribuido

**Desventajas de SIMPLE CNN vs ResNet-18 preentrenada**:
- ⚠️ Sin preentrenamiento → convergencia **muy lenta inicialmente** (primeros miles de batches)
- ⚠️ Features iniciales aleatorias = efectivamente ruido en primeros steps
- ⚠️ Peor generalización en validación (vs transfer learning con ImageNet preentrenada)
- ⚠️ SGD puro sin momentum + resincronización global cada REQUEST_PARAMS → inestable
- ⚠️ **NO RECOMENDADA para producción**

**Crítica en entorno distribuido Async-FedAvg**:
- ⚠️ **Cambios locales NO PERSISTEN**: Se resincroniza con CNN global del PS cada REQUEST_PARAMS
  - Worker entrena CNN localmente por accum_steps → gradientes computados
  - Envía parámetros al PS → son promediados con otros Workers
  - Recibe CNN global nuevamente → parámetros locales se **sobrescriben** con el promedio
  - Efecto: Gradientes de este Worker "se pierden" en comunicación network
- **Dinámica Async-FedAvg**: Cambios locales se descartan, pero gradientes se promedian **globalmente** vía PS
  - CNN global entrena distribuida
  - Convergencia es **comunal**, no local
  - Staleness hace convergencia inestable
- **NO RECOMENDADA para producción**: Inestabilidad inherente de Async-FedAvg E2E + sin pretrain

**Estado en el Sistema**:
- ✅ **ENTRENABLE (requires_grad=True)**: Recibe gradientes en backward
- ✅ CNN se actualiza localmente con SGD cada batch
- ✅ CNN se resincroniza con PS cada REQUEST_PARAMS (ver Async-FedAvg en Docs/00)
- ⚠️ **Dinámica Async-FedAvg**: Cambios locales se pierden, pero gradientes se promedian globalmente

### Forward Pass de SIMPLE CNN: Transformación de Shapes

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """SIMPLE CNN (ResNet-18 desde cero): 224×224 RGB → 512-dimensional feature vector."""
    # Stem: Conv(7×7, stride=2) + BN + ReLU + MaxPool(stride=2) → stride efectivo 4×
    x = self.conv1(x)       # (B,3,224,224) → (B,64,112,112)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.maxpool(x)     # (B,64,112,112) → (B,64,56,56)   [stride 4×]
    
    # Layer1: 2 BasicBlocks(64→64), stride=1 → sin submuestreo espacial
    x = self.layer1(x)      # (B,64,56,56)   → (B,64,56,56)   [identity blocks]
    
    # Layer2: 2 BasicBlocks(64→128), stride=2 en primer bloque
    x = self.layer2(x)      # (B,64,56,56)   → (B,128,28,28)  [stride 2×]
    
    # Layer3: 2 BasicBlocks(128→256), stride=2 en primer bloque
    x = self.layer3(x)      # (B,128,28,28)  → (B,256,14,14)  [stride 2×]
    
    # Layer4: 2 BasicBlocks(256→512), stride=2 en primer bloque
    x = self.layer4(x)      # (B,256,14,14)  → (B,512,7,7)    [stride 2×]
    
    # Global Average Pooling: 7×7 → 1×1
    x = self.gap(x)         # (B,512,7,7)    → (B,512,1,1)    [compute mean]
    x = x.flatten(1)        # (B,512,1,1)    → (B,512)        [reshape]
    
    # Dropout + Proyección lineal
    x = self.dropout(x)     # (B,512)        → (B,512)        [regularización p=0.5]
    return self.proj(x)     # (B,512)        → (B,512)        [identity projection]
```

**Stride efectivo acumulado**:
- Conv(7×7, stride=2): 2×
- MaxPool(stride=2): 2×
  - **Subtotal Stem**: 2 × 2 = 4×
- Layer2 (stride=2): 2×
- Layer3 (stride=2): 2×
- Layer4 (stride=2): 2×
- **Total**: 4 × 2 × 2 × 2 = **32×** (igual que ResNet-18)

**Verificación de spatial reduction**:
- Input: 224×224
- Output feature map: 224 / 32 = 7×7 ✓

**Por qué importa el stride 32×**:
- **Receptive field grande**: Cada neurona en layer4 ve ≈ 1024×1024 píxeles de entrada (receptive field = stride × 3×3 kernel size ≈ 30-50)
- **Verificación matemática**: Layer4 output 7×7 → 224 / 32 = 7 ✓
- **Compresión espacial eficiente**: 224² = 50,176 píxeles → 7² = 49 activaciones = **1024x reduction**
- **Estandarización**: Stride 32× es el estándar de ResNet-18 → compatible con muchas aplicaciones downstream

---

## Interfaz Pública: CNNExtractor

```python
cnn = CNNExtractor(
    arch="resnet18",           # "resnet18" o "simple"
    device="cuda",             # Device de PyTorch
    seed=None                  # None = aleatorio, int = reproducible
)
```

### Métodos Clave

#### Forward Pass (Extracción de Features)

```python
# En PS (extracción de features)
features_batch = cnn._model(image_batch)  # (N, 512)

# En Training Loop (Worker, ResNet-18 preentrenada)
image_batch = ...  # (64, 3, 224, 224) del dataset
features = cnn._model(image_batch)  # CNN congelada → (64, 512)
logits = mlp(features)  # MLP entrenable → (64, 1000)
loss = F.cross_entropy(logits, labels)
loss.backward()  # Solo MLP recibe gradientes (CNN congelada)

# En Training Loop (Worker, SIMPLE CNN)
image_batch = ...  # (64, 3, 224, 224) del dataset
features = cnn._model(image_batch)  # CNN entrenable → (64, 512)
logits = mlp(features)  # MLP entrenable → (64, 1000)
loss = F.cross_entropy(logits, labels)
loss.backward()  # **AMBAS** reciben gradientes (E2E backprop)

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

