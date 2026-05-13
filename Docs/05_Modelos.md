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
- ✅ Dinámica: CNN global se entrena distribuida via SIMPLE CNN + Async-FedAvg

### SIMPLE CNN (ResNet-18 desde Cero)

**Archivo**: `Model/cnn_extractor.py` → `_ResNet18FromScratch` + `_BasicBlock`

La SIMPLE CNN es una arquitectura **ResNet-18 estándar construida desde cero** (sin pesos preentrenados). Utiliza bloques residuales básicos (_BasicBlock) organizados en 4 capas progresivas con skip connections fuertes, optimizada para entrenamiento E2E en Async-SGD distribuido con staleness variable.

**Estructura de _BasicBlock (Bloque Residual Estándar)**:

```python
class _BasicBlock(nn.Module):
    """
    Bloque residual básico estándar de ResNet-18:
    Conv(3×3,stride)→BN→ReLU→Conv(3×3)→BN + shortcut→ReLU
    """
    expansion = 1
    
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        # Rama residual:
        self.conv1 = Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = BatchNorm2d(out_ch)
        self.conv2 = Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = BatchNorm2d(out_ch)
        
        # Shortcut: identidad o Conv(1×1) si stride > 1 o in_ch != out_ch
        self.shortcut = (
            Sequential(
                Conv2d(in_ch, out_ch, 1, stride=stride),
                BatchNorm2d(out_ch)
            ) if stride != 1 or in_ch != out_ch else Identity()
        )
        
        # CLAVE: Zero-init en BN final
        # → El bloque inicia como identidad (residual ≈ 0)
        # → Estabiliza gradientes en las primeras iteraciones
        nn.init.zeros_(self.bn2.weight)
    
    def forward(self, x):
        # Forward residual: Conv→BN→ReLU→Conv→BN
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        # Add shortcut + ReLU final
        return F.relu(out + self.shortcut(x), inplace=True)
```

**Arquitectura completa de _ResNet18FromScratch**:

```python
class _ResNet18FromScratch(nn.Module):
    def __init__(self):
        super().__init__()
        
        # STEM: reducción inicial 224→56 (stride 4×)
        self.stem = Sequential(
            Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(3, stride=2, padding=1)
        )
        
        # 4 LAYERS: cada una con 2 BasicBlocks
        # Progresión de canales: 64 → 128 → 256 → 512
        self.layer1 = self._make_layer(64, 64, 2, stride=1)    # 56×56, identidad
        self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 28×28, stride 2
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 14×14, stride 2
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 7×7, stride 2 → total 32×
        
        # GLOBAL AVERAGE POOLING: 7×7 → 1×1 → (B, 512)
        self.gap = AdaptiveAvgPool2d(1)
        
        # INICIALIZACIÓN: Kaiming Normal para todos los Conv/BN
        self._init_weights()
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        """
        Construir layer con num_blocks BasicBlocks.
        Primer bloque usa stride especificado (para submuestreo).
        Bloques restantes usan stride=1.
        """
        layers = [_BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(_BasicBlock(out_ch, out_ch, stride=1))
        return Sequential(*layers)
    
    def _init_weights(self):
        """Inicializa pesos con Kaiming Normal (He initialization)."""
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """224×224 RGB → 512-dimensional feature vector"""
        x = self.stem(x)       # (B,3,224,224) → (B,64,56,56)
        x = self.layer1(x)     # (B,64,56,56) → (B,64,56,56)
        x = self.layer2(x)     # (B,64,56,56) → (B,128,28,28)
        x = self.layer3(x)     # (B,128,28,28) → (B,256,14,14)
        x = self.layer4(x)     # (B,256,14,14) → (B,512,7,7)
        x = self.gap(x)        # (B,512,7,7) → (B,512,1,1)
        x = x.flatten(1)       # (B,512,1,1) → (B,512)  [DIRECTO, sin Dropout ni Projection]
        return x
```

**Especificaciones**:

| Atributo | Valor |
|---|---|
| Feature dimension | 512 |
| Parámetros totales | **11.2M** |
| Tamaño state_dict | **~44.8 MB** en float32 |
| Pesos | Random (Kaiming Normal init) |
| Velocidad (CPU) | ~100-120ms per batch (64 imágenes) |
| Velocidad (GPU) | ~15-20ms per batch |

**Diseño sin Dropout post-GAP y sin Proyección Linear extra**:

El código actual de SIMPLE CNN **NO incluye**:
1. ❌ `nn.Dropout(0.5)` post-GAP
2. ❌ `nn.Linear(512, 512)` para proyección

**Razones técnicas** (de `cnn_extractor.py` docstring):

```
POR QUÉ SIN DROPOUT POST-GAP:
  Dropout(0.5) post-GAP interfiere con BatchNorm en Async-SGD:
    - BN asume distribución estable de activaciones
    - Dropout la altera estocásticamente → inestabilidad con staleness
    - Con resincronización variable, gradientes ya atenuados por α(s)=1/(1+λ·s)
    - Añadir ruido de Dropout encima introduce inestabilidad numérica innecesaria
  
  Regularización efectiva viene de:
    - weight_decay=1e-4 en SGD (WorkerNode)
    - label_smoothing=0.1 en CrossEntropyLoss
    - Ambos más estables que Dropout en régimen distribuido

POR QUÉ SIN PROYECCIÓN 512→512:
  La proyección Linear(512→512) es semánticamente una identidad aprendida
    - Gap ya produce EXACTAMENTE FEATURE_DIM=512 tras Layer4
    - Proyección agrega ~262K parámetros innecesarios
    - Equipara a ResNet-18 canónico de torchvision (sin fc)
    - Reduce riesgo de colapso de representaciones en primeras épocas E2E
```

**Inicialización: Kaiming Normal (He Initialization)**

Todos los Conv2d y BatchNorm2d se inicializan con Kaiming Normal:

```python
for m in self.modules():
    if isinstance(m, Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, BatchNorm2d):
        nn.init.ones_(m.weight)       # scale = 1
        nn.init.zeros_(m.bias)        # shift = 0
```

**Ventaja de Kaiming Normal**:
- Mantiene varianza de activaciones ~constante entre layers
- Con ReLU: Espected E[h_i] = 0.5 × Var(h_0) para cada layer
- Asegura que logits iniciales ≈ N(0, 1000) → loss ≈ log(1000) ≈ 6.9
- Evita vanishing/exploding gradients en primeras épocas

**Comparativa: SIMPLE CNN (11.2M) vs Alternativas**:

| Aspecto | SIMPLE CNN (Actual) | SIMPLE CNN Anterior | ResNet-18 Preentrenada |
|---|---|---|---|
| **Parámetros** | 11.2M | 1.36M | 11.7M |
| **Pesos iniciales** | Random (Kaiming) | Random (Kaiming) | ImageNet1K_V1 |
| **requires_grad** | True (entrenable) | True (entrenable) | False (congelada) |
| **Stride efectivo** | 32× | 32× | 32× |
| **Skip connections** | Sí (_BasicBlock) | Sí (_ResBlockLite) | Sí (standard) |
| **Dropout post-GAP** | NO (eliminado) | 0.2 | Ninguno |
| **Proyección Linear** | NO (eliminado) | Sí, 512→512 | NO |
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
    """SIMPLE CNN (ResNet-18 desde cero): 224×224 RGB → 512-dimensional feature vector"""
    
    # STEM: Conv(7×7, stride=2) + BN + ReLU + MaxPool(stride=2) → stride efectivo 4×
    x = self.stem(x)       # (B,3,224,224) → (B,64,56,56)   [stride 4×]
    
    # LAYER1: 2 BasicBlocks(64→64), stride=1 → sin submuestreo espacial
    x = self.layer1(x)     # (B,64,56,56)  → (B,64,56,56)   [identity blocks]
    
    # LAYER2: 2 BasicBlocks(64→128), stride=2 en primer bloque
    x = self.layer2(x)     # (B,64,56,56)  → (B,128,28,28)  [stride 2×]
    
    # LAYER3: 2 BasicBlocks(128→256), stride=2 en primer bloque
    x = self.layer3(x)     # (B,128,28,28) → (B,256,14,14)  [stride 2×]
    
    # LAYER4: 2 BasicBlocks(256→512), stride=2 en primer bloque
    x = self.layer4(x)     # (B,256,14,14) → (B,512,7,7)    [stride 2×] → total 32×
    
    # GLOBAL AVERAGE POOLING: 7×7 → 1×1
    x = self.gap(x)        # (B,512,7,7)   → (B,512,1,1)    [mean over spatial]
    
    # FLATTEN: solo reshaping, sin transformaciones no-lineales
    x = x.flatten(1)       # (B,512,1,1)   → (B,512)
    
    # RETORNA FEATURES DIRECTAS (sin Dropout, sin Proyección Linear)
    return x               # (B,512)  ← FEATURE_DIM final
```

**Stride efectivo acumulado**:
- Stem Conv(7×7, stride=2): 2×
- Stem MaxPool(stride=2): 2×
  - **Subtotal Stem**: 2 × 2 = 4×
- Layer2 (stride=2): 2×
- Layer3 (stride=2): 2×
- Layer4 (stride=2): 2×
- **Total**: 4 × 2 × 2 × 2 = **32×** (igual que ResNet-18 preentrenada)

**Verificación de spatial reduction**:
- Input: 224×224
- Output feature map: 224 / 32 = 7×7 ✓
- GAP reduce a: 1×1 ✓
- Output final: (B, 512) ✓

**¿Por qué stride 32× importa?**:
- **Receptive field grande**: Cada neurona en Layer4 ve receptive field global (casi toda la imagen)
- **Compresión espacial eficiente**: 224² = 50,176 píxeles → 49 activaciones (7²) = **~1000x reduction**
- **Estandarización**: Stride 32× es el estándar de ResNet-18 → reproducible y comparable

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

### Arquitectura Mejorada (Sequential con Dropout + Kaiming Init)

El MLP ha sido significativamente mejorado con una arquitectura **Sequential** que incluye **Dropout para regularización**, **inicialización Kaiming correcta para ReLU**, y **eliminación de BatchNorm en la capa de salida**:

```python
class MLPPyTorch(nn.Module):
    def __init__(self, feature_dim, hidden1, hidden2, n_classes=1000):
        super().__init__()
        self.classifier = nn.Sequential(
            # Capa 1: expansión + regularización
            nn.Linear(feature_dim, hidden1),              # (512 → 1024)
            nn.BatchNorm1d(hidden1),                      # Normaliza activaciones
            nn.ReLU(inplace=True),                        # Activación
            nn.Dropout(p=0.4),                            # Regularización
            
            # Capa 2: compresión + regularización
            nn.Linear(hidden1, hidden2),                  # (1024 → 512)
            nn.BatchNorm1d(hidden2),                      # Normaliza activaciones
            nn.ReLU(inplace=True),                        # Activación
            nn.Dropout(p=0.3),                            # Regularización
            
            # Capa 3: clasificación (sin BN)
            nn.Linear(hidden2, n_classes),                # (512 → 1000) logits
        )
        self._init_weights()
    
    def forward(self, x):
        # Dropout se comporta diferente en train vs eval:
        # - model.train(): desactiva neuronas aleatoriamente (regularización)
        # - model.eval(): pasa todas las neuronas (predicción determinista)
        return self.classifier(x)
```

### Especificaciones

| Componente | Dimensión | Parámetros | Tamaño |
|---|---|---|---|
| Linear 1 (classifier.0) | (512, 1024) | 524K | 2 MB |
| BatchNorm1d (classifier.1) | 1024 | 2K | <1 MB |
| Dropout | — | 0 | 0 |
| Linear 2 (classifier.4) | (1024, 512) | 524K | 2 MB |
| BatchNorm1d (classifier.5) | 512 | 1K | <1 MB |
| Dropout | — | 0 | 0 |
| Linear 3 (classifier.8) | (512, 1000) | 512K | 2 MB |
| **Total** | - | **1.56M** | **6 MB** |

### Tres Mejoras Críticas

#### 1. **Dropout para Regularización** (Nuevo)

```python
Dropout(p=0.4) después de primer ReLU
Dropout(p=0.3) después de segundo ReLU
```

**¿Por qué es necesario?**
- Con 1000 clases, el MLP tiende a memorizar sobre features preentrenadas de ResNet-18
- Las features de ResNet-18 congeladas son muy informativas → fácil overfitting
- Dropout desactiva neuronas aleatoriamente durante entrenamiento, forzando redundancia

**Comportamiento crítico**:
- `model.train()`: Dropout activo → desactiva ~40% y ~30% de neuronas
- `model.eval()`: Dropout inactivo → todas las neuronas pasan (determinista)
- ⚠️ **CRÍTICO**: Los Workers DEBEN llamar `mlp.eval()` antes de calcular accuracy
  - Sin esto, Dropout activo durante evaluación degrada las métricas reportadas

#### 2. **Inicialización Kaiming Normal** (Cambio de Xavier)

```python
def _init_weights(self):
    for module in self.classifier.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(module.bias)
```

**¿Por qué cambiar de Xavier a Kaiming?**

| Aspecto | Xavier Uniform | Kaiming Normal (fan_out, relu) |
|---|---|---|
| Diseño | Simétrico (tanh, sigmoid) | Asimétrico (ReLU) |
| Norms esperadas | √(1/fan_in) ≈ 1.0 | √(2/fan_out) ≈ 0.04 |
| Varianza gradientes | Consistente ≈ 1.0 | Consistente ≈ 1.0 |
| Con BatchNorm | Funciona pero subóptimo | **Óptimo: mantiene σ² ≈ 1** |
| Convergencia en Async-SGD | Lenta (gradientes pequeños) | **Rápida (gradientes naturales)** |

**VALIDACIÓN EMPÍRICA**:
- Kaiming + LR=0.001 → convergencia estable, loss inicial ~6.9, accuracy mejora progresivamente ✓
- Sin BatchNorm + Kaiming → divergencia posible (activaciones sin normalizar crecen)
- Con BatchNorm1d + Kaiming → gradientes estables, logits con varianza controlada ✓

#### 3. **Sin BatchNorm en Salida** (Cambio)

```python
# ✗ ANTES:
self.fc3 = nn.Linear(hidden2, n_classes)
self.bn2 = nn.BatchNorm1d(n_classes)  # ← Incorrecto

# ✓ AHORA:
self.fc3 = nn.Linear(hidden2, n_classes)  # Sin BN en salida
```

**¿Por qué eliminar bn2?**
- BatchNorm distorsiona la distribución de logits que CrossEntropyLoss necesita
- CrossEntropyLoss aplica softmax + log-likelihood; assumes logits en distribución natural
- Normalizar los logits degrada la calibración de las probabilidades de salida
- bn2 era "capa fantasma": existía en state_dict pero forward() nunca la usaba

### Mapeo de Claves State Dict: Antiguo vs Sequential

El cambio a Sequential implica un cambio en los nombres de las claves del state_dict.
El PS y Workers soportan AMBOS formatos para compatibilidad con checkpoints antiguos:

| Componente Antiguo | Índice Sequential | Nueva Clave | Soportado | Notas |
|---|---|---|---|---|
| fc1 | 0 | classifier.0.weight/bias | ✓ Ambos | Mismas dimensiones |
| bn0 | 1 | classifier.1.weight/bias/running_* | ✓ Ambos | Buffers de BN |
| ReLU | 2 | (sin parámetros) | — | No almacenado |
| Dropout | 3 | (sin parámetros) | — | No almacenado |
| fc2 | 4 | classifier.4.weight/bias | ✓ Ambos | Mismas dimensiones |
| bn1 | 5 | classifier.5.weight/bias/running_* | ✓ Ambos | Buffers de BN |
| ReLU | 6 | (sin parámetros) | — | No almacenado |
| Dropout | 7 | (sin parámetros) | — | No almacenado |
| fc3 | 8 | classifier.8.weight/bias | ✓ Ambos | Sin BN en salida |
| ~~bn2~~ | — | ~~eliminado~~ | ✓ Descartado | Phantom layer detectado |

**Flujo de compatibilidad**:
1. **PS evaluate()**: Intenta cargar "fc1.weight" → si no existe, intenta "classifier.0.weight"
2. **Worker _sync_mlp()**: Intenta "fc1.weight" → si no existe, intenta "classifier.0.weight"
3. **Worker _load_cnn()**: Acepta AMBOS conjuntos de claves en validación

### Estado del MLP: 16 Keys en state_dict (Sequential)

```python
# Keys con Sequential:
classifier.0.weight, classifier.0.bias        # fc1
classifier.1.weight, classifier.1.bias        # bn0 weight + bias
classifier.1.running_mean                     # bn0 buffer
classifier.1.running_var                      # bn0 buffer
classifier.1.num_batches_tracked              # bn0 buffer
classifier.4.weight, classifier.4.bias        # fc2
classifier.5.weight, classifier.5.bias        # bn1 weight + bias
classifier.5.running_mean                     # bn1 buffer
classifier.5.running_var                      # bn1 buffer
classifier.5.num_batches_tracked              # bn1 buffer
classifier.8.weight, classifier.8.bias        # fc3 (sin BN)
# Total: 16 keys (vs 21 en formato antiguo, bn2 eliminado)
```
- 6 parámetros de Linear: classifier.0.weight, classifier.0.bias, classifier.4.weight, classifier.4.bias, classifier.8.weight, classifier.8.bias (✓ se promedian en PS)
- 4 parámetros de BatchNorm: classifier.1.weight, classifier.1.bias, classifier.5.weight, classifier.5.bias (✓ se promedian en PS)
- 6 buffers de BatchNorm: running_mean, running_var, num_batches_tracked × 2 capas (⊘ NO se promedian, solo se sincronizan)

### ¿Por qué BatchNorm1d en Async-SGD?

**Problema sin BatchNorm1d**:
- Xavier init: pesos ~1.0 → primeros updates pequeños (10× más chicos que con Kaiming)
- LR=0.001 × 1.0 × grad≈0.1 = actualización ≈0.0001 (negligible)
- Con múltiples Workers enviando updates tiny al PS → PS promedia → net zero
- Resultado: **Loss diverge** (7.02 → 7.11) ✗

**Solución con BatchNorm1d**:
- Normaliza activaciones post-fc1/fc2 → E[h]=0, Var[h]≈1
- Mantiene escala de gradientes consistente entre Workers
- Reduce varianza de gradientes locales → menos sensibilidad a staleness variable
- Permite usar LR=0.01 (10× mayor) sin divergencia
- **Resultado**: Loss converge correctamente (7.02 → 7.01, acc 0% → 0.3%) ✓

**Dinámicas de Sincronización**:

```
Worker-Side (training):
  - MLP siempre en train() mode
  - BatchNorm: actualiza running_mean, running_var con momentum=0.1
  - Gradientes se computan para weight y bias de BN
  - Gradientes se computan para fc1, fc2, fc3 weights+bias
  
Parameter-Server (cada REQUEST_PARAMS):
  - Recibe todos 21 keys del state_dict de MLP
  - PROMEDIA (Async-FedAvg con staleness α(s)): 
    * fc1, fc2, fc3 weights+bias (6 keys)
    * bn0, bn1 weight+bias (4 keys)
    Total: 10 keys se promedian
  - NO PROMEDIA (se mantienen intactos):
    * running_mean, running_var, num_batches_tracked (9 keys)
  - Envía: todos 21 keys intactos a Workers
  
Resultado:
  - Parámetros entrenables (fc1-3 + bn0-1 weight/bias) convergen vía Async-FedAvg
  - Estadísticas de BN se sincronizan pero NO se promedian
  - Cada Worker tiene running stats ligeramente diferentes (OK: BN=regularizador adaptivo)
  - Convergencia lineal con número de Workers
```

---

## Interacción CNN ↔ MLP

### Flujo E2E Simplificado (SIMPLE CNN)

```
Input Images: (64, 3, 224, 224)
       ↓
┌──────────────────────────────┐
│ CNN.forward() (SIMPLE CNN)   │  Extrae features (entrenable)
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
Gradients en CNN params (11.2M params) ✓ SE COMPUTAN Y SE USAN (SIMPLE CNN)
```

**⚠️ Nota sobre unidades del Loss:**

El loss reportado está en **Nats** (información natural, usando logaritmo natural). Esto es estándar en PyTorch:
- **Valor típico inicio**: ~6.9-7.0 Nats (predicción uniforme aleatoria sobre 1000 clases)
- **Valor esperado convergencia**: ~0.1-1.0 Nats  
- Para convertir a bits (log₂): Loss_bits = Loss_Nats ÷ 0.693

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
Gradients en CNN params (11.2M params) ✗ NO se computan (congelada)
```

### Dinámicas de Entrenamiento CNN: Local vs Global (SIMPLE CNN)

**SIMPLE CNN SÍ se entrena en el código, pero con dinámicas especiales:**

| Aspecto | Realidad |
|---|---|
| **Backward E2E** | Gradientes llegan a CNN (11.2M parámetros de SIMPLE CNN) |
| **SGD local** | CNN se actualiza: `cnn_param.data -= lr * cnn_param.grad` |
| **Duración** | Cambios CNN locales duran accum_steps batches (ej: 5 batches) |
| **Resincronización** | Cada REQUEST_PARAMS, CNN local se SOBRESCRIBE con CNN global del PS |
| **Persistencia** | CNN cambios locales se DESCARTAN cuando sincroniza (NO persisten) |
| **Global** | PS recibe CNN de cada Worker, la promedia con Async-FedAvg → CNN global SÍ aprende |
| **Comunicación** | CNN 44.8MB + MLP 6MB = ~51MB total intercambiados (AMBAS se sincronizan) |

**Dinámica Especial SIMPLE CNN**:
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
| **Comunicación** | CNN 44.8MB + MLP 6MB = ~51MB total intercambiados (ambas se sincronizan por completitud) |

**Dinámica Especial ResNet-18**:
- CNN congelada permanentemente → transfer learning puro
- Solo MLP se entrena → clasificador aprendible sobre features fijas
- **Efecto**: Convergencia más rápida que SIMPLE CNN (features preentrenadas), pero limitada (CNN no se adapta)

---

## Cómo se Entrena Realmente (SIMPLE CNN)

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

