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

### Steps per Report

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
  --seed 42 \
  --hf-token "hf_..." \
  --accum-steps 1
```

---

## Variables de Entorno

```bash
export HF_TOKEN="hf_..."  # Alternativa a --hf-token

python ps_imagenet.py   # Automáticamente usa HF_TOKEN
python worker_imagenet.py
```

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

