"""
Model/cnn_extractor.py

Extractor CNN para ImageNet con PyTorch.

ARQUITECTURAS:
  "resnet18" → ResNet-18 con pesos ImageNet (recomendado).
               Con pretrained=True: pesos IMAGENET1K_V1.
               feature_dim = 512. Sin capa de clasificación.
             
  "simple"   → CNN propia de 3 bloques Conv→BN→ReLU→MaxPool.
               feature_dim = 512. Sin pesos preentrenados.
               Útil para experimentación sin descargar pesos externos.

DIFERENCIAS CON LA VERSIÓN ANTERIOR:
  - Eliminado prepare(), _load_features_if_cached(), _save_features(),
    _load_weights_if_cached(), list_saved_models(), load_metadata().
  - Solo se mantiene lo necesario para el sistema distribuido:
      __init__, _get_weights_bytes(), load_weights_from_bytes(),
      extract_batched(), set_trainable(), feature_dim.
"""

from __future__ import annotations

import hashlib
import io
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

FEATURE_DIM = 512


# ================================================================
# CNN SIMPLE
# ================================================================

class _SimpleCNN(nn.Module):
    """
    CNN de 3 bloques: Conv→BN→ReLU→MaxPool + proyección a 512.

    Diseñada para imágenes de cualquier resolución. Produce un
    vector de 512 features compatible con el MLP clasificador.
    """

    def __init__(self) -> None:
        super().__init__()

        def _block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            _block(3, 64),
            _block(64, 128),
            _block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, FEATURE_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


# ================================================================
# EXTRACTOR PÚBLICO
# ================================================================

class CNNExtractor:
    """
    Envuelve una CNN PyTorch y expone una interfaz simple para el sistema distribuido.

    El PS usa esta clase para:
      1. Serializar pesos y distribuirlos a Workers (_get_weights_bytes).
      2. Evaluar el modelo global en el split de validación (extract_batched).

    Los Workers usan esta clase para:
      1. Recibir y cargar pesos del PS (load_weights_from_bytes).
      2. Ejecutar el forward pass E2E durante el entrenamiento.

    :param arch:       'resnet18' o 'simple'.
    :param pretrained: Si True, carga pesos ImageNet para resnet18.
    :param device:     Dispositivo PyTorch ('cpu', 'cuda', 'mps').
    :param seed:       Semilla de inicialización (solo afecta a 'simple').
    """

    ARCHITECTURES = ("simple", "resnet18")

    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = True,
        device: str = "cpu",
        seed: Optional[int] = 42,
    ) -> None:
        if arch not in self.ARCHITECTURES:
            raise ValueError(f"arch debe ser {self.ARCHITECTURES}, recibido: {arch!r}")

        self.arch      = arch
        self.pretrained = pretrained
        self.seed      = seed
        self.device    = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)

        self._model = self._build(arch, pretrained).to(self.device)
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()

    @staticmethod
    def _build(arch: str, pretrained: bool) -> nn.Module:
        if arch == "simple":
            return _SimpleCNN()

        import torchvision.models as tvm
        weights = "IMAGENET1K_V1" if pretrained else None
        model = tvm.resnet18(weights=weights)
        model.fc = nn.Identity()  # type: ignore  # expone vector de 512 features
        return model

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    # ── Serialización de pesos (para distribuir por TCP) ─────────

    def _get_weights_bytes(self) -> bytes:
        """Serializa el state_dict a bytes para enviar por TCP."""
        buf = io.BytesIO()
        torch.save(self._model.state_dict(), buf)
        return buf.getvalue()

    def load_weights_from_bytes(self, weights_bytes: bytes) -> None:
        """Carga pesos desde bytes recibidos del PS por TCP."""
        buf = io.BytesIO(weights_bytes)
        state = torch.load(buf, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

    def _weights_hash(self) -> str:
        """Hash MD5 (8 hex) de los pesos actuales — para logging."""
        h = hashlib.md5()
        for t in self._model.state_dict().values():
            h.update(t.cpu().numpy().tobytes())
        return h.hexdigest()[:8]

    # ── Extracción de features ────────────────────────────────────

    def set_trainable(self, trainable: bool) -> None:
        """Activa o desactiva gradientes y modo train/eval."""
        for p in self._model.parameters():
            p.requires_grad_(trainable)
        self._model.train() if trainable else self._model.eval()

    def extract_batched(
        self,
        X: np.ndarray,
        batch_size: int = 512,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Extrae features en mini-batches.

        Usado por el PS durante la evaluación de validación para
        procesar el split completo sin agotar la VRAM.

        :param X:          Imágenes (N, 3, H, W) float32.
        :param batch_size: Imágenes por batch.
        :param verbose:    Imprimir progreso.
        :return:           Features (N, feature_dim) float32.
        """
        N      = len(X)
        parts  = []
        starts = range(0, N, batch_size)

        for i, start in enumerate(starts, 1):
            chunk = X[start : start + batch_size]
            with torch.inference_mode():
                t = torch.from_numpy(chunk).to(self.device)
                parts.append(self._model(t).cpu().numpy())
            if verbose:
                done = min(start + batch_size, N)
                print(f"\r  [CNN] {done}/{N} imgs ({i}/{len(starts)} batches)",
                      end="", flush=True)

        if verbose:
            print()
        return np.concatenate(parts, axis=0)