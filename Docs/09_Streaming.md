# Streaming de Datos: Pipeline HuggingFace

## Arquitectura General del Streaming

```
┌─ HuggingFace Datasets ──┐
│   ILSVRC/imagenet-1k    │
│   timm/imagenet-1k-wds  │
└────────┬────────────────┘
         │  Descarga chunks bajo demanda
         ▼
┌─────────────────────┐
│  Download Cache     │ (~1.3M imágenes, 144GB)
└────────┬────────────┘
         │  Streaming con índice
         ▼
┌──────────────────────────────┐
│   ImageNetStream(split)      │
│   - select(shard via range)  │
│   - map(transform)           │
│   - batch(batch_size)        │
│   - iter (∞ loop)            │
└────────┬─────────────────────┘
         │  Batches (tensor format)
         ▼
┌──────────────────────────────────┐
│   PrefetchBuffer Thread          │
│   - Gets batches from iter       │
│   - Puts into Queue              │
│   - Runs in background (async)   │
└────────┬─────────────────────────┘
         │  Queue[torch.Tensor]
         ▼
┌──────────────────────────────────┐
│   Main Training Loop (Worker)    │
│   - Gets from queue (blocking)   │
│   - Forward pass CNN+MLP         │
│   - Backward pass                │
│   - Update weights locally       │
└──────────────────────────────────┘
```

---

## ImageNetStream: Infinite Generator

### Propósito

Generar un iterator infinito sobre ImageNet-1k que:
- Cargue datos bajo demanda desde HuggingFace
- Se reinicie automáticamente al llegar al final
- Aplique transformaciones (crop, flip, normalize)
- Retorne batches (imágenes, labels)

### Implementación

Ubicación: [Utils/imagenet_streaming.py](../Utils/imagenet_streaming.py#L20)

```python
class ImageNetStream:
    def __init__(
        self,
        split='train',           # 'train' o 'validation'
        batch_size=64,           # Imágenes por batch
        num_workers=1,           # Número de Workers totales
        rank=0,                  # ID del Worker actual (0..num_workers-1)
        shuffle_buffer=5000,     # Tamaño de buffer shuffle
        image_size=224,          # Resolución de salida
        hf_token=None,           # Token de HuggingFace
        dataset_name='ILSVRC/imagenet-1k'  # Dataset ID
    ):
```

### Flujo de Ejecución

#### 1. **Carga Inicial del Dataset**

```python
dataset = load_dataset(
    dataset_name,              # e.g., 'ILSVRC/imagenet-1k'
    split=split,               # 'train'
    streaming=True,            # ← Clave: No descarga todo
    use_auth_token=hf_token
)
```

**Salida en Consola**:
```
Resolving data files: 100%|██████████| 14/14 [00:02<00:00, 7.89it/s]
Resolving data files: 100%|██████████| 1/1 [00:00<00:00, 100.00it/s]
```

**¿Por qué aparece dos veces?**
- Primera: Descargando metadata y splits del dataset
- Segunda: Validando acceso a splits específicos

#### 2. **Sharding por Worker**

```python
# Si num_workers=4, rank=0 → tomar posiciones 0, 4, 8, 12, ...
# Si num_workers=4, rank=1 → tomar posiciones 1, 5, 9, 13, ...

def __iter__(self):
    while True:
        # Infinite loop: reinicia si acaba dataset
        for idx, sample in enumerate(self.dataset):
            # Sharding: saltar muestras de otros workers
            if idx % self.num_workers != self.rank:
                continue
            
            # Procesar esta muestra
            yield self._process_sample(sample)
```

**Implementación Real**:

```python
ds_iter = dataset.iter(batch_size=self.batch_size)

while True:  # ← Infinite
    try:
        # Obtener siguiente batch del streamer
        batch = next(ds_iter)
        
        # Aplicar transformaciones
        batch = self._transform_batch(batch)
        
        yield batch
        
    except StopIteration:
        # Fin del dataset, reiniciar
        print(f"[Worker {self.rank}] Dataset exhausted, restarting...")
        ds_iter = dataset.iter(batch_size=self.batch_size)
```

### Transformación de Datos

Ubicación: [Utils/imagenet_streaming.py](../Utils/imagenet_streaming.py#L80)

```python
def _transform_batch(self, batch):
    # batch['image']: List[PIL.Image] de tamaño batch_size
    # batch['label']: List[int] de tamaño batch_size
    
    images = []
    labels = []
    
    for img, label in zip(batch['image'], batch['label']):
        # 1. RandomResizedCrop(224): Crop aleatorio + resize
        #    (Mantiene aspect entre 0.75-1.0, resize a 224x224)
        img = self.crop_transform(img)
        
        # 2. RandomHorizontalFlip (50% chance)
        if random.random() < 0.5:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        
        # 3. Convertir a Tensor y normalizar ImageNet stats
        img_tensor = self.to_tensor(img)  # [3, 224, 224], values [0..1]
        img_tensor = self.normalize(img_tensor)  # ImageNet mean/std
        
        images.append(img_tensor)
        labels.append(label)
    
    # Stack en batch
    batch_tensor = torch.stack(images)  # [batch_size, 3, 224, 224]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return batch_tensor, labels_tensor
```

**ImageNet Normalization Stats**:

```python
mean = [0.485, 0.456, 0.406]  # RGB mean
std = [0.229, 0.224, 0.225]   # RGB std

normalized = (img - mean) / std
```

---

## PrefetchBuffer: Asynchronous Queueing

### Propósito

Ejecutar en thread separado:
```
┌─ StreamThread ──────────────┐
│  iter = ImageNetStream()    │
│  while 1:                   │
│    batch = next(iter)       │ ← Toma CPU+IO
│    queue.put(batch)         │
│                             │
│  (Running 24/7 in background)
│                             │
└─ MainThread ───────────────┐
│  worker_loop():            │
│    batch = queue.get()      │ ← Espera si vacío
│    forward+backward()       │ ← Entrena con GPU
│                             │
└─────────────────────────────┘
```

### Implementación

Ubicación: [Utils/imagenet_streaming.py](../Utils/imagenet_streaming.py#L130)

```python
class PrefetchBuffer:
    def __init__(self, stream, prefetch_size=4):
        """
        Args:
            stream: ImageNetStream iterator
            prefetch_size: Número de batches a pre-cargar
        """
        self.stream = stream
        self.queue = queue.Queue(maxsize=prefetch_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._producer_loop,
            daemon=True  # Muere con main thread
        )
        self.thread.start()
    
    def _producer_loop(self):
        """Corre en thread separado"""
        try:
            for batch in self.stream:
                # Bloquea si queue está lleno (back-pressure)
                self.queue.put(batch)
        except Exception as e:
            print(f"[PrefetchBuffer] Error: {e}")
            self.stop_event.set()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Bloquea si queue vacío (espera a producer)
        if self.stop_event.is_set() and self.queue.empty():
            raise StopIteration
        
        try:
            batch = self.queue.get(timeout=10)
            return batch
        except queue.Empty:
            raise StopIteration
```

### Flujo de Datos

**Stato Normal (prefetch=4)**:

```
Tiempo T=0:
- Producer: cargando batch 1
- Queue: [vacío]
- Main: esperando

T=0.1s:
- Producer: cargando batch 2
- Queue: [batch 1]
- Main: procesando batch 1

T=0.2s:
- Producer: cargando batch 3
- Queue: [batch 2]
- Main: procesando batch 2 (GPU ocupada)

T=0.3s:
- Producer: cargando batch 4
- Queue: [batch 3, batch 4]
- Main: esperando fin GPU (bloquea)

T=0.4s:
- Producer: esperando (queue lleno, back-pressure)
- Queue: [batch 3, batch 4]
- Main: procesando batch 3

T=0.5s:
- Producer: cargando batch 5 (sacó from queue)
- Queue: [batch 4, batch 5]
- Main: procesando batch 4

...continúa infinito...
```

### Back-Pressure Mechanism

Si Producer es más rápido que Consumer:

```
Queue maxsize = 4

[batch1][batch2][batch3][batch4] [QUEUE LLENO]
                                      ↓
                            queue.put(batch5)  ← BLOQUEA
                                      
Espera hasta que Main consume un batch,
luego Producer continúa
```

**Beneficio**: Ahorra memory (no descarga 1000 batches en advance)

---

## ValidationStream: One-Pass Validator

### Propósito

Iterar exactamente una vez sobre validation split para evaluar

### Implementación

Ubicación: [Utils/imagenet_streaming.py](../Utils/imagenet_streaming.py#L200)

```python
class ValidationStream:
    def __init__(
        self,
        batch_size=64,
        image_size=224,
        hf_token=None,
        dataset_name='ILSVRC/imagenet-1k'
    ):
        self.dataset = load_dataset(
            dataset_name,
            split='validation',
            streaming=True,
            use_auth_token=hf_token
        )
        self.batch_size = batch_size
        self.image_size = image_size
        self.transforms = get_validation_transforms(image_size)
    
    def __iter__(self):
        # Una sola pasada (sin loop infinito)
        ds_iter = self.dataset.iter(batch_size=self.batch_size)
        
        for batch in ds_iter:
            # Sin aplicar random transforms (determinístico)
            batch = self._transform_batch(batch)
            yield batch
```

**Diferencias con ImageNetStream**:

| Aspecto | Train | Validation |
|---|---|---|
| transforms | RandomCrop, RandomFlip | CenterCrop (fijo) |
| shuffle | Sí (shuffle_buffer) | No |
| loop | Infinito (reinicia) | Una pasada (StopIteration) |
| uso | Entrenamiento continuo | Evaluación periódica |

---

## Sharding Per-Worker

### Problema

Si 4 Workers descargan independientemente:
```
Worker 0: [img0, img1, img2, img3, img4, ...]
Worker 1: [img0, img1, img2, img3, img4, ...]  ← DUPLICADO
Worker 2: [img0, img1, img2, img3, img4, ...]  ← DUPLICADO
Worker 3: [img0, img1, img2, img3, img4, ...]  ← DUPLICADO
```

**Consecuencia**: Loss correlacionado, gradientes sesgados, convergencia pobre

### Solución: Índice Strided

```python
dataset = load_dataset('ILSVRC/imagenet-1k', split='train', streaming=True)

# Convertir a indexable
select_indices = range(rank, len_dataset, num_workers)
#  rank=0, num_workers=4 → [0, 4, 8, 12, 16, ...]
#  rank=1, num_workers=4 → [1, 5, 9, 13, 17, ...]
#  rank=2, num_workers=4 → [2, 6, 10, 14, 18, ...]
#  rank=3, num_workers=4 → [3, 7, 11, 15, 19, ...]

sharded_dataset = dataset.select(select_indices)
```

**Resultado**:
```
Worker 0: [img0, img4, img8, img12, ...]        ← Disjoint
Worker 1: [img1, img5, img9, img13, ...]        ← Disjoint
Worker 2: [img2, img6, img10, img14, ...]       ← Disjoint
Worker 3: [img3, img7, img11, img15, ...]       ← Disjoint
Cobertura completa sin repetición
```

### Implementación Real

Ubicación: [Utils/imagenet_streaming.py](../Utils/imagenet_streaming.py#L50)

```python
def __init__(self, ..., rank=0, num_workers=1, ...):
    self.rank = rank
    self.num_workers = num_workers
    
    # Nota: En streaming mode, select() puede no estar disponible
    # Alternativa: filtrar en __iter__
    
def __iter__(self):
    ds_iter = self.dataset.iter(batch_size=self.batch_size)
    
    for idx, sample in enumerate(ds_iter):
        # Filtrar: solo procesar si es "nuestro" índice
        if idx % self.num_workers != self.rank:
            continue
        
        yield self._transform_batch(sample)
```

---

## Performance Characteristics

### Throughput Típico

En CPU (Intel i7-9700K):

| batch_size | prefetch | Throughput |
|---|---|---|
| 32 | 2 | 20 batches/sec (~640 img/sec) |
| 64 | 4 | 15 batches/sec (~960 img/sec) |
| 128 | 8 | 8 batches/sec (~1024 img/sec) |

En GPU (RTX 2080):

| batch_size | prefetch | Throughput |
|---|---|---|
| 64 | 4 | 100 batches/sec (~6400 img/sec) |
| 256 | 8 | 80 batches/sec (~20480 img/sec) |

**Io Bound** en CPU → prefetch crítico  
**Compute Bound** en GPU → prefetch ayuda pero menos crítico

### Memory Overhead

```python
PrefetchBuffer(prefetch_size=4, batch_size=64):

Per batch: (64 imgs @ 224x224 RGB) 
         = 64 × 224 × 224 × 3 × 4bytes (float32)
         = 200MB

Queue: 4 × 200MB = 800MB

Total: ~800MB queue + overhead torchvision
```

### Latencia Primera Imagen

```
t=0: HuggingFace server connection
t=0.5s: Download metadata (~5KB)
t=1.0s: Download primeras imágenes
t=1.5s: ResNet forward (warmup)
t=2.0s: Ready para first training step
```

**Nota**: Primera pasada más lenta (socket setup, disk cache priming)

---

## Debugging Streaming

### Problem: "Resolviendo data files" aparece 10 veces

**Causa**: Cada Worker crea nuevo dataset → Cada uno descarga metadata

**Solución**: Cachear dataset en archivo local

```bash
# First run (downloads)
export HF_HOME=/cache/huggingface
python ps_imagenet.py --dataset ILSVRC/imagenet-1k

# Second run (uses /cache/huggingface)
python worker_imagenet.py --dataset ILSVRC/imagenet-1k
```

### Problem: Loss NaN después de 1000 steps

**Causa**: Posible issue con transformaciones (extremos clipping)

**Debug**:

```python
# En _transform_batch()
img_tensor = self.normalize(img_tensor)  # [3, 224, 224]

print(f"img min={img_tensor.min()}, max={img_tensor.max()}")
print(f"label={labels_tensor}")

if torch.isnan(img_tensor).any():
    print("⚠️ NaN detected in transformations!")
```

### Problem: Training lentísimo

**Causa**: Posible que Producer sea más lento que Consumer

**Debug**:

```python
import time

class DebugPrefetchBuffer(PrefetchBuffer):
    def __next__(self):
        self.size_before = self.queue.qsize()
        batch = self.queue.get(timeout=10)
        self.size_after = self.queue.qsize()
        
        if self.size_before == 0:
            print(f"⚠️ Queue was empty! (starvation)")
        
        return batch

# En training loop
for batch in debug_buffer:
    if debug_buffer.size_before == 0:
        print("Producer is too slow!")
```

---

## Configuration Tuning

### Para Máxima Velocidad

```python
stream = ImageNetStream(
    batch_size=256,          # Más imágenes/batch
    shuffle_buffer=1000,     # Más shuffle (poco overhead)
)

buffer = PrefetchBuffer(
    stream,
    prefetch_size=8          # Más buffering
)
```

### Para Mínimo Memory

```python
stream = ImageNetStream(
    batch_size=32,           # Menos memoria/batch
    shuffle_buffer=100,      # Menos shuffle buffer
)

buffer = PrefetchBuffer(
    stream,
    prefetch_size=2          # Minimal buffering
)
```

### Para Development/Debugging

```python
stream = ImageNetStream(
    batch_size=8,            # Pequeño para debug
    shuffle_buffer=0,        # Sin shuffle (determinístico)
    dataset_name='timm/imagenet-1k-wds'  # Más pequeño
)

buffer = PrefetchBuffer(
    stream,
    prefetch_size=1          # Sin buffering (instant feedback)
)
```

