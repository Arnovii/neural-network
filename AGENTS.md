# AGENTS.md - Neural Network Training System

## Project Overview
Sistema de entrenamiento distribuido asíncrono con arquitectura Parameter Server para clasificación ImageNet-1k (1000 clases).

## What This System Does
- **Async-SGD**: Múltiples Workers entrenan en paralelo sin barreras de sincronización
- **Parameter Server**: Coordina gradientes y mantiene estado global (CNN + MLP)
- **Staleness Correction**: α(s) = 1/(1+λ·s) mitiga divergencia por asincronía
- **Streaming**: ImageNet desde HuggingFace bajo demanda (nunca descarga completo)

## Runtime Requirements
- Python >= 3.13.5
- HuggingFace token para ImageNet-1k (`HF_TOKEN` env var o `--hf-token` flag)
- Aceptar licencia en: https://huggingface.co/datasets/ILSVRC/imagenet-1k

## Entry Points

| File | Purpose |
|------|---------|
| `ps_imagenet.py` | Parameter Server modo terminal |
| `ps_gui_imagenet.py` | Parameter Server + GUI interactiva |
| `worker_imagenet.py` | Nodo Worker async |

## Usage

```bash
# PS con GUI (recomendado, no tiene export-dir integrado aún)
python ps_gui_imagenet.py --hf-token "hf_..."

# Worker (conecta al PS automáticamente)
python worker_imagenet.py --server-host 127.0.0.1

# PS terminal (CON exportación de resultados automática)
python ps_imagenet.py --lr 0.001 --staleness-lambda 0.1 --hf-token "hf_..."

# PS terminal con directorio personalizado para resultados
python ps_imagenet.py --export-dir ./mi_experimento --max-steps 10000 --hf-token "hf_..."
```

## Architecture

```
PS (0.0.0.0:9999)
    ├── CNN: ResNet-18 (512 features) o CNN Simple (ResNet-from-scratch)
    ├── MLP: 512→1024→512→1000
    └── FedAvg async + staleness correction

Worker 0 ──►  Worker 1 ──►  Worker N
(request→train→updates)  (async loop)
```

## Key Directories

| Directory | Contents |
|-----------|----------|
| `Distributed/` | ParameterServer, WorkerNode, Protocol (TCP) |
| `Model/` | CNNExtractor, MLPPyTorch |
| `Utils/` | Streaming, Logging, ResultsExporter |

## Message Protocol (10 types)
```
READY → WORKER_ID → CONFIG → CNN_WEIGHTS → CNN_ACK → START
[loop:] REQUEST_PARAMS ↔ PARAMS ↔ UPDATES
[stop:] STOP
```

## CNN Architectures

| Name | Mode | Pretrained | Trainable |
|------|------|------------|-----------|
| `resnet18` | Freeze | Yes (ImageNet) | MLP only |
| `simple` | E2E | No | CNN + MLP |

## Key Hyperparameters

| Param | Default | Description |
|-------|---------|------------|
| `--lr` | 0.001 | Learning rate MLP |
| `--lr-cnn` | 0.001 | LR CNN (E2E only) |
| `--staleness-lambda` | 0.1 | Staleness correction factor |
| `--hidden1` | 1024 | MLP hidden layer 1 |
| `--hidden2` | 512 | MLP hidden layer 2 |
| `--batch-size` | 64 | Batch size por Worker |
| `--image-size` | 224 | Image resolution |
| `--dataset` | ILSVRC/imagenet-1k | Dataset (specified in PS, not Worker) |
| `--max-steps` | 0 | Limit de steps (0=sin límite) |
| `--hf-token` | None | HuggingFace token |

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `ParameterServer` | `Distributed/parameter_server.py` | Coordinator, FedAvg async |
| `WorkerNode` | `Distributed/worker_node.py` | Training loop async |
| `CNNExtractor` | `Model/cnn_extractor.py` | ResNet-18 o Simple CNN |
| `MLPPyTorch` | `Model/mlp_pytorch.py` | 2-layer MLP classifier |
| `ImageNetStream` | `Utils/imagenet_streaming.py` | HuggingFace streaming |
| `PrefetchBuffer` | `Utils/imagenet_streaming.py` | Async prefetch |
| `ResultsExporter` | `Utils/results_exporter.py` | Exporta métricas/plots |
| `FormattedLogger` | `Utils/logging_util.py` | Colored logging |

## Plot Generation (ResultsExporter)
Sistema completamente desacoplado de exportación de resultados que genera:

### 9 Archivos por Experimento
```
./Exports/[timestamp]/
├── config.json           # Configuración del experimento
├── metrics.csv           # Series de tiempo (step, loss, acc, workers)
├── ps_logs.txt           # Todos los logs del Parameter Server
├── metadata.json         # Estadísticas finales
├── plot_3panels.png      # 3 gráficas (Loss | Accuracy | Workers) - 283 KB
├── plot_loss.png         # Gráfica individual de Loss
├── plot_accuracy.png     # Gráfica individual de Accuracy  
└── plot_workers.png      # Gráfica individual de Workers
```

### Estilos de Visualización
- Loss: `#E74C3C` (rojo), markers "o"
- Accuracy: `#27AE60` (verde), markers "s"
- Workers: `#3498DB` (azul), markers "^"
- Grid: alpha=0.15 (Loss/Workers), alpha=0.4 (Accuracy)
- Escala: Loss/Workers (±10%), Accuracy (dinámico ±20%)
- Resolución: 300 DPI, formato PNG, `bbox_inches="tight"`, `pad_inches=0.3`

## Dependencies (requirements.txt)

| Paquete | Versión | Propósito |
|--------|---------|---------|
| `python-dotenv` | 1.0.1 | Carga variables desde .env |
| `torch` | 2.10.0 | Redes neuronales |
| `torchvision` | 0.25.0 | Transformaciones de imágenes |
| `matplotlib` | 3.10.8 | Visualización de métricas |
| `datasets` | 4.8.4 | Streaming HuggingFace |
| `numpy` | 2.4.2 | Operaciones numéricas |
| `psutil` | 7.2.2 | Utilidades del sistema |

## No CI/Lint/Typecheck
Este repo NO tiene configurado:
- ruff/flake8 (lint)
- mypy (typecheck)
- test suite
- pre-commit hooks

## Important Design Decisions

1. **SGD-only optimizer**: No Adam — sus momentos (m,v) se desincronizan con FedAvg
2. **Separate LRs**: lr (MLP) + lr_cnn (E2E) — CNN necesita LR más bajo
3. **Label smoothing**: 0.1 — reduce overconfidence
4. **Weight decay**: 1e-4 (E2E) — regularización L2
5. **Gradient clipping**: max_norm=1.0 — estabilidad en E2E
6. **Sharding**: strided index — Workers cubren dataset sin overlap

## Exports Structure
```
./Exports/[timestamp]/
├── config.json
├── metrics.csv
├── ps_logs.txt
├── plot_3panels.png
├── plot_loss.png
├── plot_accuracy.png
├── plot_workers.png
└── metadata.json
```

## Common Issues

1. **HF Token missing**: `python worker_imagenet.py --hf-token "hf_..."`
2. **Worker can't connect**: Verificar PS host/port y firewall
3. **OOM**: Reducir batch_size o prefetch
4. **No convergence**: Ajustar staleness_lambda (0.1 default)

## File Key Lines and Features
| File | Lines | Content |
|------|-------|---------|
| `ps_gui_imagenet.py` | 1256-1326 | GUI plot rendering |
| `Distributed/parameter_server.py` | 1-100 | Design notes, async-SGD, staleness |
| `Distributed/worker_node.py` | 1-66 | Training modes, E2E vs MLP-only |
| `Model/cnn_extractor.py` | 1-51 | Architecture docs (ResNet-18 vs Simple) |
| `Utils/results_exporter.py` | 1-100 | Export system design, 9-file output |
| `Utils/results_exporter.py` | 279-370 | Plot generation (_generate_plots + 5 methods) |

## ResultsExporter Class (Utils/results_exporter.py)
- **Thread-safe**: record_metric() y record_log() con locks
- **Async finalize()**: Escritura no-bloqueante al finalizar
- **5 Plot methods**: _plot_3panels(), _plot_individual_loss/accuracy/workers(), _plot_comparison()
- **Dynamic scaling**: Loss ±10%, Accuracy ±20%, Workers ±15%
- **No dependencies**: Completamente desacoplado de ParameterServer
- **Integration**: Registrado via logging_util.add_log_handler() en PS.listen()

---

## Environment Configuration (.env)

El token de HuggingFace puede configurarse mediante archivo `.env`:

```bash
# Crear archivo .env en la raíz del proyecto
echo "HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > .env

# El sistema carga automáticamente el token desde .env
python ps_imagenet.py
```

### Prioridad de HF_TOKEN

```
CLI (--hf-token) > .env > HF_TOKEN (variable de entorno)
```

---

## CLI Reference

### Parameter Server (ps_imagenet.py)

| Parámetro | Default | Descripción |
|-----------|---------|------------|
| `--host` | 0.0.0.0 | Host del PS |
| `--port` | 9999 | Puerto TCP |
| `--lr` | 0.001 | Learning rate MLP |
| `--staleness-lambda` | 0.1 | Factor corrección staleness |
| `--hidden1` | 1024 | Capa oculta MLP 1 |
| `--hidden2` | 512 | Capa oculta MLP 2 |
| `--batch-size` | 64 | Batch size |
| `--image-size` | 224 | Resolución imágenes |
| `--dataset` | ILSVRC/imagenet-1k | Dataset HuggingFace |
| `--seed` | None | Semilla RNG |
| `--steps-per-report` | 500 | Steps entre reportes |
| `--max-steps` | 0 (ilimitado) | Límite de steps |
| `--hf-token` | None | Token HuggingFace |

### Parameter Server GUI (ps_gui_imagenet.py)

Interfaz gráfica con los mismos parámetros que ps_imagenet.py más configuración visual.

### Worker (worker_imagenet.py)

| Parámetro | Default | Descripción |
|-----------|---------|------------|
| `--server-host` | 127.0.0.1 | IP del PS |
| `--server-port` | 9999 | Puerto del PS |
| `--device` | auto | cpu, cuda, cuda:N, mps |
| `--shuffle-buffer` | 1000 | Buffer de shuffle |
| `--prefetch` | 4 | Batches en prefetch |
| `--seed` | None | Semilla RNG |
| `--accum-steps` | 1 | Batches a acumular |

**Nota**: El dataset se especifica en el PS, no en el Worker.
