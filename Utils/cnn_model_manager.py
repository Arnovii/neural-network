"""
Utils/cnn_model_manager.py

Gestión de modelos CNN en modo "simple":
  - Guardar pesos y metadata
  - Cargar por hash
  - Listar modelos guardados
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any

from Model.cnn_extractor import CNNExtractor


MODELS_DIR = "Data/cnn_models"


def _ensure_models_dir() -> str:
    """Crear carpeta de modelos si no existe."""
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def _compute_weights_hash(weights_bytes: bytes) -> str:
    """Generar hash MD5 de los pesos (8 primeros caracteres hex)."""
    return hashlib.md5(weights_bytes).hexdigest()[:8]


def save_cnn_model(
    cnn: CNNExtractor,
    n_train: int,
    epochs: int,
    final_accuracy: float,
    final_loss: float,
    elapsed_time: float,
    description: str = "",
) -> str:
    """
    Guardar modelo CNN con metadata.

    :param cnn: Instancia de CNNExtractor entrenada
    :param n_train: Ejemplos usados en entrenamiento
    :param epochs: Épocas entrenadas
    :param final_accuracy: Precisión final
    :param final_loss: Pérdida final
    :param elapsed_time: Tiempo de entrenamiento (segundos)
    :param description: Descripción opcional
    :return: Hash del modelo guardado
    """
    if cnn.arch != "simple":
        raise ValueError(f"Solo modo 'simple' permite guardar. Arch={cnn.arch}")

    models_dir = _ensure_models_dir()

    # Obtener pesos
    weights_bytes = cnn._get_weights_bytes()
    model_hash = _compute_weights_hash(weights_bytes)

    # Rutas
    weights_file = os.path.join(models_dir, f"cnn_{model_hash}.pt")
    metadata_file = os.path.join(models_dir, f"cnn_{model_hash}.json")

    # Guardar pesos
    with open(weights_file, "wb") as f:
        f.write(weights_bytes)

    # Metadata
    metadata = {
        "hash": model_hash,
        "arch": "simple",
        "timestamp": datetime.now().isoformat(),
        "n_train": n_train,
        "epochs": epochs,
        "final_accuracy": float(final_accuracy),
        "final_loss": float(final_loss),
        "elapsed_seconds": float(elapsed_time),
        "elapsed_formatted": f"{int(elapsed_time // 60)}m {elapsed_time % 60:.2f}s",
        "description": description,
        "weights_file": weights_file,
        "metadata_file": metadata_file,
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return model_hash


def load_cnn_model(
    model_hash: str,
    device: str = "cpu",
    seed: int = 42,
) -> Optional[CNNExtractor]:
    """
    Cargar modelo CNN guardado.

    :param model_hash: Hash del modelo (8 chars)
    :param device: Dispositivo ("cpu", "cuda", "mps")
    :param seed: Semilla aleatoria
    :return: CNNExtractor cargada o None si no existe
    """
    models_dir = _ensure_models_dir()
    weights_file = os.path.join(models_dir, f"cnn_{model_hash}.pt")
    metadata_file = os.path.join(models_dir, f"cnn_{model_hash}.json")

    if not os.path.exists(weights_file):
        return None

    # Crear CNN
    cnn = CNNExtractor(
        arch="simple",
        device=device,
        seed=seed,
        cache_dir=os.path.join(MODELS_DIR, "cache"),
        input_size=224,
    )

    # Cargar pesos
    with open(weights_file, "rb") as f:
        weights_bytes = f.read()
    cnn.load_weights_from_bytes(weights_bytes)

    return cnn


def list_cnn_models() -> List[Dict[str, Any]]:
    """
    Listar todos los modelos guardados.

    :return: Lista de dicts con metadata de cada modelo
    """
    models_dir = _ensure_models_dir()
    models = []

    for metadata_file in sorted(Path(models_dir).glob("cnn_*.json")):
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            models.append(metadata)
        except Exception:
            pass

    return models


def delete_cnn_model(model_hash: str) -> bool:
    """
    Eliminar modelo CNN guardado.

    :param model_hash: Hash del modelo
    :return: True si se eliminó, False si no existe
    """
    models_dir = _ensure_models_dir()
    weights_file = os.path.join(models_dir, f"cnn_{model_hash}.pt")
    metadata_file = os.path.join(models_dir, f"cnn_{model_hash}.json")

    deleted = False
    if os.path.exists(weights_file):
        os.remove(weights_file)
        deleted = True
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
        deleted = True

    return deleted


def format_models_list(models: List[Dict[str, Any]]) -> str:
    """Formatear lista de modelos para impresión."""
    if not models:
        return "No hay modelos CNN guardados."

    lines = []
    lines.append("=" * 80)
    lines.append("MODELOS CNN GUARDADOS (modo 'simple')")
    lines.append("=" * 80)

    for i, m in enumerate(models, 1):
        lines.append(f"\n  [{i}] Hash: {m['hash']}")
        lines.append(f"      Fecha: {m['timestamp']}")
        lines.append(f"      Epochs: {m['epochs']}  |  N_train: {m['n_train']:,}")
        lines.append(
            f"      Accuracy: {m['final_accuracy']:.2f}%  |  Loss: {m['final_loss']:.4f}"
        )
        lines.append(f"      Tiempo: {m['elapsed_formatted']}")
        if m.get("description"):
            lines.append(f"      Nota: {m['description']}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def print_models() -> None:
    """Imprimir lista formateada de modelos."""
    models = list_cnn_models()
    print(format_models_list(models))
