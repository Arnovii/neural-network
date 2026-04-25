# Hiperparámetros y Configuración

## Hiperparámetros de Entrenamiento

### Learning Rate (η)

**Default**: 0.001  
**Rango**: (0.00001, 1.0)  
**Efecto**: Tamaño del paso en SGD

```
θ_new = θ - η · ∇L
```

**Impacto**:

| LR | Convergencia | Estabilidad | Divergencia |
|---|---|---|---|
| 0.00001 | Muy lenta | ✓ Estable | Nunca (tarda meses) |
| 0.001 | Ideal | ✓ Estable | Raro |
| 0.01 | Rápida | ⚠️ Oscila | A veces |
| 0.1 | Muy rápida | ✗ Diverge | Frecuente |
| 1.0+ | - | ✗ Colapsa | Siempre |

**Recomendación**:
- Comenzar con **0.001** (default safe)
- Si loss cae muy lento → aumentar a 0.005-0.01
- Si loss diverge (NaN) → reducir a 0.0005

### Staleness Lambda (λ)

**Default**: 0.1  
**Rango**: (0.0, 1.0)  
**Efecto**: Factor de corrección de antiguedad de gradientes

```
α(s) = 1 / (1 + λ · s)

donde s = version_actual - version_leída
```

**Interpretación**:

| λ | Comportamiento | Casos de Uso |
|---|---|---|
| 0.0 | Sin corrección (puro Async-SGD) | Red rápida, pocos workers |
| 0.05 | Corrección leve | Red de alta velocidad |
| 0.1 | **Corrección balanceada** | Producción estándar |
| 0.5 | Corrección fuerte | Red lenta, muchos workers |
| 1.0 | Casi Sync-SGD | Máxima estabilidad |

**Impacto en α**:

```
Ejemplo: s = 10 (staleness)

λ=0.0:  α = 1.0/(1+0.0×10) = 1.00 ← Sin atenuación
λ=0.1:  α = 1.0/(1+0.1×10) = 0.50 ← Atenuación media
λ=1.0:  α = 1.0/(1+1.0×10) = 0.09 ← Fuerte atenuación
```

**Recomendación**:
- Default **0.1** es seguro
- Si loss oscila → aumentar a 0.2-0.5
- Si convergencia muy lenta → reducir a 0.05

### Batch Size

**Default**: 64  
**Rango**: (8, 512)  
**Efecto**: Muestras por gradient step

**Impacto**:

| Batch Size | Throughput | Memory | Gradient Noise | Convergence |
|---|---|---|---|---|
| 8 | Lento | Bajo | Alto (gradientes ruidosos) | Oscilatoria |
| 64 | Balanceado | Medio | Medio (recomendado) | Suave |
| 256 | Rápido | Alto | Bajo (gradientes suave) | Puede estancarse |
| 512 | MuyRápido | Muy alto | Muy bajo | Posible sobreajuste |

**Relación con LR**: Batch grande → puede usar LR más grande

**Recomendación**:
- Default **64** es estándar para ImageNet
- GPU disponible → aumentar a 128-256
- Memory limited → reducir a 32

### MLP Architecture

**Defaults**: hidden1=1024, hidden2=512

**Impacto**:

| Arquit | Params | Speed/Iter | Memory | Expresividad |
|---|---|---|---|---|
| (512, 256, 128) | 0.4M | 5ms | 20MB | Baja |
| (512, 512, 256) | 0.7M | 8ms | 30MB | Media |
| (512, 1024, 512) | 1.6M | 20ms | 50MB | **Alta** |
| (512, 2048, 1024) | 4.1M | 50ms | 150MB | MuyAlta |

**Trade-off**: Más parámetros = más datos, más lento, mejor generalization

**Recomendación**:
- Default **(1024, 512)** es bueno
- Time-critical → reducir a (512, 256)
- Mucho tiempo/data → aumentar a (2048, 1024)

### Steps per Report

**Default**: 500  
**Efecto**: Cada cuántos steps reportar métricas

```
if current_step % steps_per_report == 0:
    avg_loss, avg_acc = metrics.average()
    on_report(current_step, avg_loss, avg_acc)
```

**Impacto en gráficas**:

| Value | Puntos en Gráfica | Resolución |
|---|---|---|
| 100 | +100/1000 = 10 puntos/1K steps | Alta (detalle) |
| 500 | +20 puntos/1K steps | Media (estándar) |
| 1000 | +10 puntos/1K steps | Baja (suave) |

**Recomendación**:
- **500** es balance buen visual + overhead bajo
- Debugging → reducir a 100
- Producción → hasta 1000

### Metrics Window (Ventana Deslizante)

**Default**: 200  
**Rango**: (10, 5000)  
**Efecto**: Tamaño de la ventana deslizante para calcular promedios de loss/accuracy

```python
# Implementación (en parameter_server.py)
class RunningMetrics:
    def __init__(self, window=200):
        self._losses = deque(maxlen=window)  # Últimos 200 valores
        self._accs = deque(maxlen=window)
    
    def update(self, loss, acc):
        self._losses.append(loss)
        self._accs.append(acc)
    
    def snapshot(self):
        avg_loss = mean(self._losses)  # Promedio de los últimos 200
        avg_acc = mean(self._accs)
        return avg_loss, avg_acc
```

**Impacto en GUI**:

| Window | Suavizado | Responsividad | Latencia | Uso |
|---|---|---|---|---|
| 10 | Bajo (ruidoso) | Muy rpdo (0.1s) | Casi inmediato | DEBUG |
| 50 | Medio (bueno) | Responsivo (5s) | Minutos | ✅ **RECOMENDADO** |
| 200 | Alto (suave) | Lento (20s) | Varios min | PRODUCCIÓN |
| 500 | MuyAlto | Muy lento (50s) | 5-10 min | LARGA DURACIÓN |

**Relación con Steps Per Report**:

```
ventana=50, steps_per_report=500:
  - Accuracy GUI (encima gráfica) se actualiza cada step (~1s)
  - Accuracy en gráfica se actualiza cada 500 steps (~5-10 min)
  - Demora visible pero responsivo

ventana=200, steps_per_report=500:
  - Accuracy GUI demora más en estabilizarse
  - Accuracy en gráfica es muy suave
  - Menos volatilidad pero tardío
```

**¿Cuándo cambiar?**

- **Baja ventana (10-50)**: 
  - ✓ Ves cambios casi en tiempo real
  - ✓ Debugging y experimentación  
  - ✗ Ruidoso (accuracy fluctúa)

- **Alta ventana (200-500)**:
  - ✓ Muy suavizado
  - ✓ Tendencia clara
  - ✗ Demora ~200-500 steps antes de cambios visibles (~5-10 minutos)

**Recomendación operacional**:
- **50** para experimentación rápida (ves resultados cada minuto)
- **200** para entrenamiento largo estable (tendencia clara)

**Nota**: Esta ventana SOLO afecta el promedio mostrado. No afecta el entrenamiento real.

---

### Max Steps (Límite de Entrenamiento)

**Default**: 0 (sin límite)  
**Rango**: 0 o entero positivo  
**Efecto**: Detiene automáticamente el entrenamiento al alcanzar N steps

```
Si max_steps > 0 Y current_step >= max_steps:
    → Detener entrenamiento
    → Exportar métricas
    → Cerrar conexiones
```

**Comportamiento**:

| Valor | Efecto |
|-------|-------|
| 0 (default) | Sin límite, entrenamiento infinito |
| > 0 | Detiene al alcanzar el límite |

**Diferencia GUI vs Terminal**:

| Interfaz | Campo | Comportamiento al alcanzar límite |
|---------|-------|----------------------------------|
| GUI | "Límite steps (vacío=∞)" | Auto-detención sin askyesno + messagebox.showinfo |
| Terminal | `--max-steps N` | Detiene silenciosamente |

**Ejemplos de uso**:

```bash
# GUI: deixar campo vacío = sin límite
# GUI: introducir 10000 = detiene automático a step 10000

# Terminal: sin límite
python ps_imagenet.py --hf-token "hf_..."

# Terminal: detener a 10000 steps
python ps_imagenet.py --max-steps 10000 --hf-token "hf_..."

# Terminal: experimentar con 1000 steps
python ps_imagenet.py --max-steps 1000 --steps-per-report 100 --hf-token "hf_..."
```

---

## Combinaciones Recomendadas

### Scenario 1: Validación Rápida (2-3 horas)

```
--lr 0.01              # Convergencia rápida
--staleness-lambda 0.5 # Red potencialmente lenta
--batch-size 256       # Más imágenes/iter
--hidden1 512          # MLP pequeño
--hidden2 256
--steps-per-report 100 # Más detalles
```

**Esperado**:
- ~20k steps en 2 horas
- Loss: 7.0 → 3.0
- Accuracy: 0% → 10-15%

### Scenario 2: Producción (24 horas)

```
--lr 0.001              # Estándar, estable
--staleness-lambda 0.1  # Balance óptimo
--batch-size 64         # Balance memoria/throughput
--hidden1 1024          # MLP completo
--hidden2 512
--steps-per-report 500  # Menos overhead
```

**Esperado**:
- ~200k steps en 24 horas (múltiples workers)
- Loss: 7.0 → 0.5
- Accuracy: 0% → 40-60%

### Scenario 3: Investigación (Debugging)

```
--lr 0.001              # Valores default seguros
--staleness-lambda 0.1
--batch-size 32         # Menos memory, más iterations
--hidden1 256           # MLP tiny para rápida iteration
--hidden2 128
--dataset timm/imagenet-1k-wds  # Dataset público (sin token)
--steps-per-report 100  # Detalles frecuentes
--prefetch 2            # Menos memory overhead
```

**Esperado**:
- Rápido development loop
- Debugging más fácil
- Lower convergence pero OK para testing

---

## Configuración por Línea de Comando

### Parameter Server Terminal

```bash
python ps_imagenet.py \
  --host 0.0.0.0 \
  --port 9999 \
  --lr 0.001 \
  --staleness-lambda 0.1 \
  --hidden1 1024 \
  --hidden2 512 \
  --cnn-arch resnet18 \
  --steps-per-report 500 \
  --max-steps 50000 \
  --metrics-window 200 \
  --hf-token "hf_..."
```

### Parameter Server GUI

```bash
python ps_gui_imagenet.py --hf-token "hf_..."
```

Luego usar la interfaz gráfica para configurar

### Worker

```bash
python worker_imagenet.py \
  --server-host 127.0.0.1 \
  --server-port 9999 \
  --device cuda \
  --dataset ILSVRC/imagenet-1k \
  --shuffle-buffer 1000 \
  --prefetch 4 \
  --accum-steps 1
```

**Nota**: El Worker **no** recibe `--hf-token` por CLI. El token es gestionado completamente por el PS y se envía al Worker a través del mensaje CONFIG. Esto centraliza la gestión de credenciales.

---

## Variables de Entorno

```bash
export HF_TOKEN="hf_..."  # Variable de entorno para HuggingFace token

# Parameter Server: Lee token de variable de entorno (si no se usa --hf-token)
python ps_imagenet.py   # Automáticamente usa HF_TOKEN del entorno

# Worker: Lee parámetros (batch_size, seed, hf_token, etc.) del PS via CONFIG
python worker_imagenet.py --server-host 127.0.0.1
```

**Flujo de Token HuggingFace**:
1. PS recibe token vía: `--hf-token CLI_arg` o variable `HF_TOKEN` del entorno
2. PS almacena el token internamente
3. PS envía el token a cada Worker en el mensaje CONFIG
4. Worker utiliza el token del CONFIG para streaming de datos

---

## Regularización

### Gradient Clipping (GRAD_CLIP_MAX_NORM)

**Default**: 10.0  
**Rango**: (0.1, 100.0)  
**Efecto**: Recorta gradientes para evitar explosión

```python
# Implementación (en worker_node.py)
nn.utils.clip_grad_norm_(parameters, max_norm=GRAD_CLIP_MAX_NORM)
```

**Tabla de valores**:

| Valor | Efecto |
|-------|-------|
| 0.1 | Muy agresivo, puede bloquear aprendizaje |
| 1.0 | Recomendado para stability |
| 5.0 | Balance moderado |
| 10.0 | **Default actual** - permite más variación |
| Sin límite | Puede causar NaN por gradientes grandes |

**Nota**: Solo afecta entrenamiento E2E (Simple CNN). En MLP-only (ResNet-18) los gradientes son más pequeños por defecto.

---

### Label Smoothing (LABEL_SMOOTHING)

**Default**: 0.1  
**Rango**: (0.0, 1.0)  
**Efecto**: Suaviza las etiquetas para evitar overconfidence

```python
# Implementación (en worker_node.py)
loss = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Tabla de valores**:

| Valor | Efecto |
|-------|-------|
| 0.0 | Sin smoothing - hard labels |
| 0.1 | **Default推荐** - suave pero no terlalu |
| 0.2 | Más smoothing |
| 0.5 | Muy smoothing, puede hinder aprendizaje |

---

### Weight Decay (WEIGHT_DECAY)

**Default**: 1e-4 (0.0001)  
**Efecto**: Regularización L2 en SGD

```python
# Implementación (en worker_node.py)
optimizer = torch.optim.SGD(params, lr=lr, weight_decay=1e-4)
```

**Tabla de valores**:

| Valor | Efecto |
|-------|-------|
| 0.0 | Sin regularización |
| 1e-5 | Baja regularización |
| 1e-4 | **Default** - balance óptimo |
| 1e-3 | Alta regularización |
| 1e-2 | Puede impedir convergencia |

---

## Control de Workers

### Worker Accum Steps (WORKER_ACCUM_STEPS_DEFAULT)

**Default**: 1  
**Rango**: (1, 32)  
**Efecto**: Número de batches a acumular antes de enviar gradientes al PS

```
accum_steps > 1 Reduce overhead de comunicación
```

**Tabla**:

| Valor | Uso | Throughput |
|-------|-----|-----------|
| 1 | Default, cada batch se envía | Menor latencia |
| 2-4 | Reduces red, más memoria | Balance |
| 8+ | Para redes lentas | Máxima eficiencia |

---

### Validation Batch Size (VALIDATION_BATCH_SIZE_DEFAULT)

**Default**: 256  
**Efecto**: Batches usados para evaluación en validación

```
Mayor batch = evaluación más rápida pero más memoria
```

---

## Constantes del Proyecto (Utils/constants.py)

### Propósito

El módulo `Utils/constants.py` centraliza todos los valores de configuración del proyecto para evitar **magic numbers** dispersos en el código. Facilita el mantenimiento y asegura consistencia entre GUI y línea de comandos.

**Beneficios**:
- ✅ Un solo lugar para modificar valores por defecto
- ✅ Tipos definidos (type hints) para mejor soporte de IDE
- ✅ Nombres claros que documentan el propósito
- ✅ Consistency entre CLI y GUI

### Uso en Código

**Importación básica**:
```python
from Utils.constants import DEFAULT_BATCH_SIZE, IMAGE_SIZE, COLORS
```

### Catálogo Completo de Constantes

| Constante | Valor | Descripción |
|----------|-------|-------------|
| CLOCK_UPDATE_MS | 1000 | Intervalo de actualización del reloj de la GUI (ms) |
| COLORS | dict | Paleta principal de colores (estado, métricas y UI) |
| DEFAULT_BATCH_SIZE | 64 | Imágenes por batch en cada Worker |
| DEFAULT_HOST | "0.0.0.0" | Host de escucha del PS por defecto |
| DEFAULT_LR | 0.001 | Learning rate MLP |
| DEFAULT_LR_CNN | 0.001 | Learning rate CNN (E2E) |
| DEFAULT_PORT | 9999 | Puerto TCP del PS por defecto |
| DEFAULT_SEED | None | Semilla por defecto (None = aleatorio) |
| DEFAULT_STALENESS_LAMBDA | 0.1 | Factor de corrección staleness λ |
| EXPORT_DIR_DEFAULT | "./Exports" | Directorio de resultados exportados |
| FEATURE_DIM | 512 | Dimensión features de CNN |
| GRAD_CLIP_MAX_NORM | 10.0 | Umbral de gradient clipping |
| GUI_COLORS | dict | Paleta de colores específica de la GUI |
| GUI_INITIAL_XMAX | 4 | Límite inicial del eje X en gráficas GUI |
| HF_DATASET_DEFAULT | "ILSVRC/imagenet-1k" | Dataset por defecto |
| HIDDEN1_DEFAULT | 1024 | Neuronas capa oculta 1 del MLP |
| HIDDEN2_DEFAULT | 512 | Neuronas capa oculta 2 del MLP |
| IMAGE_SIZE | 224 | Resolución de imágenes |
| LABEL_SMOOTHING | 0.1 | Suavizado de etiquetas |
| MAX_LOG_LINES | 300 | Cantidad máxima de líneas en log (GUI) |
| MAX_STEPS_UNLIMITED | 0 | Valor para desactivar límites de steps |
| NUM_CLASSES | 1000 | Clases de dataset |
| POLL_TIMEOUT_MS | 100 | Intervalo de polling de la GUI (ms) |
| PREFETCH_DEFAULT | 4 | Batches en cola de prefetch |
| METRICS_WINDOW_DEFAULT | 50 | Ventana para promedios |
| STEPS_PER_REPORT_DEFAULT | 10 | Steps entre reportes |
| SHUFFLE_BUFFER_DEFAULT | 1000 | Valor de shuffle buffer |
| TOOLTIP_DELAY_MS | 500 | Delay de aparición de tooltips (ms) |
| VALIDATION_BATCH_SIZE_DEFAULT | 256 | Batch size por defecto para evaluación de validación |
| VAL_BATCHES_DEFAULT | 50 | Batches para validación GUI |
| WEIGHT_DECAY | 1e-4 | Valor de Weight decay L2 |
| WORKER_ACCUM_STEPS_DEFAULT | 1 | Accum steps por worker |
| WORKER_COLORS | list | Paleta rotativa para colorear workers en la GUI |
| WORKER_SERVER_HOST_DEFAULT | "127.0.0.1" | Host por defecto de conexión del Worker al PS |

---

## Impact Tuning Guide

Si **loss no baja**:
1. ✓ Aumentar `--lr` a 0.005-0.01
2. ✓ Reducir `--staleness-lambda` a 0.05
3. ✓ Verificar que PS está inicializando modelos
4. ✓ Ver logs: "MLP no recibido del PS"?

Si **loss diverge (NaN)**:
1. ✓ Reducir `--lr` a 0.0005
2. ✓ Aumentar `--staleness-lambda` a 0.5
3. ✓ Reducir `--batch-size` a 32
4. ✓ Revisar datasets (labels válidos?)

Si **muy lento**:
1. ✓ Aumentar `--batch-size` a 128-256
2. ✓ Aumentar `--prefetch` (si hay memory)
3. ✓ Usar GPU (`--device cuda`)
4. ✓ Reducir `--hidden1`, `--hidden2`

Si **memory error**:
1. ✓ Reducir `--batch-size` a 32
2. ✓ Reducir `--prefetch` a 2
3. ✓ Usar `--device cpu` (contraintuitivo, menos cache)
4. ✓ Reducir `--hidden1`, `--hidden2`

