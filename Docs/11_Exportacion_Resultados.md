# Exportación de Resultados y Análisis

## Propósito del Sistema de Exportación

**Archivo**: `Utils/results_exporter.py`

El sistema de exportación **desacoplado** genera reportes automáticos al finalizar el entrenamiento:

1. **Automatización**: No requiere intervención manual
2. **Completitud**: Exporta configuración, métricas, logs y gráficas
3. **Reproducibilidad**: Timestamps únicos por experimento
4. **Análisis**: Datos en formato CSV + visualizaciones profesionales

---

## Arquitectura de Desacoplamiento

### Interfaz Pública

```python
from Utils.results_exporter import ResultsExporter

# Inicializar
exporter = ResultsExporter(
    config={
        "lr": 0.1,
        "lr_cnn": 0.001,
        "staleness_lambda": 0.1,
        "batch_size": 64,
        "image_size": 224,
        "seed": 42,
        "cnn_arch": "resnet18",
        "description": "Experimento 1"
    },
    export_dir="./Exports"  # Directorio base
)

# Durante entrenamiento: registrar métricas
exporter.record_metric(step=1, loss=7.05, accuracy=0.001, num_workers=1)
exporter.record_metric(step=2, loss=7.04, accuracy=0.002, num_workers=1)

# Durante entrenamiento: registrar logs
exporter.record_log("Worker 0 conectado")
exporter.record_log("Step 10 | loss=7.02 | acc=1.5%")

# Al finalizar: generar todos los archivos
result_dir = exporter.finalize()
# retorna: Path("./Exports/20260421_193015_550/")
```

### Thread Safety

- `record_metric()` y `record_log()` son **thread-safe**
- Usa locks internos para proteger buffers circulares
- No bloquea entrenamiento (buffering en memoria)

---

## Estructura de Salida

### Directorio por Sesión

```
./Exports/20260421_193015_550/
├── config.json           # ← Configuración del experimento
├── metrics.csv           # ← Series de tiempo
├── ps_logs.txt           # ← Logs del Parameter Server
├── metadata.json         # ← Estadísticas finales
├── plot_3panels.png      # ← Combinación: Loss | Accuracy | Workers
├── plot_loss.png         # ← Gráfica individual Loss
├── plot_accuracy.png     # ← Gráfica individual Accuracy
├── plot_workers.png      # ← Gráfica individual Workers
└── plot_comparison.png   # ← Comparación Loss vs Accuracy (ejes duales)
```

**Total**: 9 archivos por experimento

---

## Contenido de Archivos

### 1. config.json

**Propósito**: Registra configuración exacta del experimento para reproducibilidad.

```json
{
  "lr": 0.1,
  "lr_cnn": 0.001,
  "staleness_lambda": 0.1,
  "batch_size": 64,
  "image_size": 224,
  "seed": 42,
  "cnn_arch": "resnet18",
  "host": "0.0.0.0",
  "port": 9999,
  "description": "Distributed Async-SGD on ImageNet-1k"
}
```

### 2. metrics.csv

**Propósito**: Series de tiempo para análisis cuantitativo.

**Formato**:
```
step,loss,accuracy,num_workers
1,7.0706,0.0000,1
2,7.0633,0.0000,1
3,7.0375,0.0052,1
4,7.0262,0.0039,1
5,7.0419,0.0031,1
...
```

**Uso típico**:
```python
import pandas as pd

df = pd.read_csv("metrics.csv")
print(f"Loss final: {df['loss'].iloc[-1]:.4f}")
print(f"Accuracy máxima: {df['accuracy'].max():.2%}")
print(f"Workers promedio: {df['num_workers'].mean():.1f}")

# Gráficas personalizadas
import matplotlib.pyplot as plt
plt.plot(df['step'], df['loss'])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('custom_loss.png', dpi=150)
```

### 3. ps_logs.txt

**Propósito**: Historial completo de eventos del Parameter Server.

**Ejemplo**:
```
================================================================================
PARAMETER SERVER LOGS
================================================================================
Session: 20260421_193015_550
Start time: 2026-04-21 19:30:15.550245
================================================================================

[2026-04-21 19:30:15.551] [PARAM SRV] Escuchando en 0.0.0.0:9999
[2026-04-21 19:30:25.563] [PARAM SRV] Worker 0 conectado desde 127.0.0.1:50603
[2026-04-21 19:30:25.781] [PARAM SRV] Worker 0: CNN enviada — arch=resnet18 | cnn_params=120
[2026-04-21 19:30:26.535] [PARAM SRV] Worker 0: CNN cargada ✓ arch=resnet18 mode=freeze
[2026-04-21 19:30:41.435] [TRAIN MLP] Step 1 | loss=7.0706 | acc=0.00% | workers=1
[2026-04-21 19:30:45.490] [TRAIN MLP] Step 2 | loss=7.0633 | acc=0.00% | workers=1
...
================================================================================
END OF LOGS
================================================================================
```

### 4. metadata.json

**Propósito**: Estadísticas agregadas para resumen rápido.

```json
{
  "total_steps": 13,
  "loss_min": 7.0082,
  "loss_max": 7.0706,
  "loss_mean": 7.0351,
  "accuracy_min": 0.0031,
  "accuracy_max": 0.0052,
  "accuracy_mean": 0.0041,
  "workers_min": 1,
  "workers_max": 1,
  "workers_mean": 1.0,
  "training_time_seconds": 72.5,
  "timestamp": "20260421_193015_550"
}
```

### 5-9. plot_*.png

**Propósito**: Visualizaciones para reportes académicos/profesionales.

#### plot_3panels.png

Combinación de 3 gráficas en un solo archivo:

```
┌─────────────────────────────────────────────────┐
│ Loss Evolution                                  │
│ ┌───────────────────────────────────────────┐   │
│ │  7.08 ●                                   │   │
│ │       ●●●●●●●●●●●●                        │   │
│ │  7.02 ●    (con margen 10% superior)      │   │
│ └───────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│ Accuracy Evolution                              │
│ ┌───────────────────────────────────────────┐   │
│ │  0.006■                                   │   │
│ │        ■ ■  ■                             │   │
│ │  0.002 ■■■■■■■■■■■ (margen dinámico)      │   │
│ └───────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│ Active Workers Over Time                        │
│ ┌───────────────────────────────────────────┐   │
│ │  2 ▲                                      │   │
│ │    ▲▲▲▲▲▲▲▲▲▲▲▲▲                          │   │
│ │  1 ▲      (con margen 15% superior)       │   │
│ └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

**Escalas dinámicas**:
- Loss: `[loss_min - 5%, loss_max + 10%]`
- Accuracy: `[0, acc_max + max(0.05, 20%)]`
- Workers: `[0, workers_max + 15%]`

#### plot_loss.png, plot_accuracy.png, plot_workers.png

Gráficas individuales para zoom en cada métrica.

#### plot_comparison.png

Ejes Y duales (twinx) para comparación Loss vs Accuracy:

```
┌──────────────────────────────────────────┐
│ Loss vs Accuracy                         │
│ ┌──────────────────────────────────────┐ │
│ │ 7.08 ●         0.006 ■               │ │
│ │      ●●●●    ●  ■ ■  ■               │ │
│ │ 7.02          ●●■■■■■■■■■■           │ │
│ │ Loss (red)    Accuracy (green)       │ │
│ └──────────────────────────────────────┘ │
│ Training Step                            │
└──────────────────────────────────────────┘
```

---

## Integración con Parameter Server

### En ps_imagenet.py

```bash
python ps_imagenet.py \
  --export-dir ./resultados_exp1 \
  --max-steps 10000
```

**Resultado**: `./resultados_exp1/[timestamp]/` con 9 archivos.

### En Distributed/parameter_server.py

El `ParameterServer` crea `ResultsExporter` automáticamente:

```python
# En __init__
self._results_exporter = ResultsExporter(
    config=config,
    export_dir=self.export_dir
)

# En listen()
_log.add_log_handler(self._results_exporter.record_log)

# En _apply_update()
if self._results_exporter is not None:
    self._results_exporter.record_metric(step, avg_loss, avg_acc, n_workers)

# En stop()
export_path = self._results_exporter.finalize()
_log.info(f"Resultados exportados a: {export_path}")
```

---

## Uso Avanzado

### Exportar a directorio personalizado

```bash
# Valor por defecto
python ps_imagenet.py --export-dir ./Exports

# Personalizado
python ps_imagenet.py --export-dir /data/experiments/run_001

# Timestamps relativos (usando fecha)
python ps_imagenet.py --export-dir ./results_$(date +%Y%m%d)
```

### Procesar resultados con Python

```python
import json
import pandas as pd
from pathlib import Path

exp_dir = Path("./Exports/20260421_193015_550")

# 1. Cargar config
with open(exp_dir / "config.json") as f:
    config = json.load(f)
print(f"LR: {config['lr']}, Architecture: {config['cnn_arch']}")

# 2. Cargar métricas
df = pd.read_csv(exp_dir / "metrics.csv")
print(f"\nConvergencia:")
print(f"  Loss inicial: {df['loss'].iloc[0]:.4f}")
print(f"  Loss final:   {df['loss'].iloc[-1]:.4f}")
print(f"  Accuracy máx: {df['accuracy'].max():.2%}")

# 3. Cargar logs
with open(exp_dir / "ps_logs.txt") as f:
    logs = f.read()
# Analizar eventos, tiempos, errores

# 4. Cargar metadata
with open(exp_dir / "metadata.json") as f:
    meta = json.load(f)
print(f"\nTiempo total: {meta['training_time_seconds']:.1f}s")
```

### Comparar múltiples experimentos

```python
import pandas as pd
from pathlib import Path

# Cargar métricas de varios experimentos
experiments = {
    "exp1_lr001": "./Exports/exp1/metrics.csv",
    "exp2_lr010": "./Exports/exp2/metrics.csv",
    "exp3_lr100": "./Exports/exp3/metrics.csv",
}

results = {}
for name, csv_path in experiments.items():
    df = pd.read_csv(csv_path)
    results[name] = {
        "loss_final": df['loss'].iloc[-1],
        "acc_max": df['accuracy'].max(),
        "convergence_step": df[df['loss'] < 6.5].iloc[0] if any(df['loss'] < 6.5) else None
    }

# Análisis
for name, metrics in results.items():
    print(f"{name}: loss={metrics['loss_final']:.4f}, acc={metrics['acc_max']:.2%}")
```

---

## Limitaciones y Consideraciones

### Buffering en Memoria

- Las métricas se mantienen en buffers circulares de tamaño máximo 500 (configurable)
- **Ventaja**: No bloqueante durante entrenamiento
- **Desventaja**: Si experimento tiene >500 steps, solo se guardan últimos 500
- **Solución**: Usar tamaño de ventana más grande (ver `metrics_window` en __init__)

### Gráficas Deterministas

- **Ventaja**: Mismo formato cada vez, fácil comparación
- **Desventaja**: Escalas rígidas pueden cortar datos si hay picos inesperados
- **Solución**: Revisar datos en metrics.csv si sospecha anomalías

### Sin Compresión

- Archivos PNG sin compresión JPEG (PNG es lossless)
- 5 PNGs ≈ 850 KB totales por experimento
- **Solución**: Scripts de post-processing para comprimir

---

## Roadmap Futuro

1. **Exportación incremental**: Guardar métricas cada N steps (no solo al final)
2. **Dashboard web**: Visualizar resultados en navegador (Flask/Dash)
3. **Comparación automática**: Tablas de comparación entre experimentos
4. **Compresión**: Opción para PNG con compresión JPEG
5. **Filtros personalizados**: Exportar solo métricas seleccionadas

---

## Referencias

- **Format CSV**: RFC 4180 (standard CSV)
- **Format JSON**: RFC 8259 (JavaScript Object Notation)
- **Matplotlib**: https://matplotlib.org/stable/api/pyplot_api.html
- **Pandas**: https://pandas.pydata.org/docs/

