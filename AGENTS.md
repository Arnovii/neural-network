# AGENTS.md - Neural Network Training System

## Runtime Requirements
- Python >= 3.13.5
- HuggingFace token for ImageNet-1k dataset (set via `HF_TOKEN` env var or `--hf-token` flag)

## Entry Points

| File | Purpose |
|------|---------|
| `ps_imagenet.py` | Parameter Server (terminal mode) |
| `ps_gui_imagenet.py` | Parameter Server + GUI |
| `worker_imagenet.py` | Async Worker node |

## Key Directories

| Directory | Ownership |
|-----------|-----------|
| `Distributed/` | ParameterServer, WorkerNode, Protocol |
| `Model/` | CNNExtractor (ResNet-18/Simple), MLPPyTorch |
| `Utils/` | Streaming, Logging, ResultsExporter |

## Important Files

| File | Purpose |
|------|---------|
| `Utils/results_exporter.py` | Export metrics/plots to `./Exports/[timestamp]/` |
| `Utils/imagenet_streaming.py` | HuggingFace streaming, sharding, prefetch |
| `Utils/logging_util.py` | FormattedLogger with ANSI colors |
| `ps_gui_imagenet.py` | GUI with live matplotlib plots (lines 1256-1326) |

## Usage

```bash
# Parameter Server + GUI (recommended for monitoring)
python ps_gui_imagenet.py --hf-token "hf_..."

# Worker (connects to PS automatically)
python worker_imagenet.py --server-host 127.0.0.1 --hf-token "hf_..."

# Terminal mode PS
python ps_imagenet.py --lr 0.001 --staleness-lambda 0.1 --hf-token "hf_..."
```

## Dependencies
All in `requirements.txt`:
- torch, torchvision (neural networks)
- matplotlib (plotting)
- datasets (HuggingFace ImageNet streaming)
- numpy, psutil

## No Formal CI/Lint Config
This repo has no configured:
- lint (ruff, flake8)
- typecheck (mypy)
- test suite
- pre-commit hooks
- build system (use `python -m venv` + `pip install -r requirements.txt`)

## GUI Plot Style Reference
The export plots in `results_exporter.py` should match `ps_gui_imagenet.py` (lines 1280-1324):
- Loss: `#F44336`, accuracy: `#2196F3`, workers: `#4CAF50`
- Workers use `step()` (not `plot()`)
- Accuracy axis: 0-100 scale
- Grid alpha: 0.3