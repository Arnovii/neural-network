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
# PS con GUI (recomendado)
python ps_gui_imagenet.py --hf-token "hf_..."

# Worker (conecta al PS automáticamente)
python worker_imagenet.py --server-host 127.0.0.1 --hf-token "hf_..."

# PS terminal
python ps_imagenet.py --lr 0.001 --staleness-lambda 0.1
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

## GUI Plot Style (export debe igualar esto)
- Loss: `#F44336`, Accuracy: `#2196F3`, Workers: `#4CAF50`
- Workers: usar `step()` (no `plot()`)
- Accuracy axis: 0-100 escala
- Grid alpha: 0.3

## Dependencies (requirements.txt)
- `torch==2.10.0`, `torchvision==0.25.0`
- `matplotlib==3.10.8`
- `datasets` (HuggingFace streaming)
- `numpy==2.4.2`, `psutil==7.0.0`

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
├── plot_comparison.png
└── metadata.json
```

## Common Issues

1. **HF Token missing**: `python worker_imagenet.py --hf-token "hf_..."`
2. **Worker can't connect**: Verificar PS host/port y firewall
3. **OOM**: Reducir batch_size o prefetch
4. **No convergence**: Ajustar staleness_lambda (0.1 default)

## File Key Lines
| File | Lines | Content |
|------|-------|---------|
| `ps_gui_imagenet.py` | 1256-1326 | GUI plot rendering |
| `Distributed/parameter_server.py` | 1-100 | Design notes |
| `Distributed/worker_node.py` | 1-66 | Training modes |
| `Model/cnn_extractor.py` | 1-51 | Architecture docs |
| `Utils/results_exporter.py` | 258-296 | Plot generation |