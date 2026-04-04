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
        """
        Inicializa el codificador CNN de 3 bloques (Conv→BN→ReLU→MaxPool).

        Construye capas de extracción de características convolucionales seguidas de
        pooling adaptativo y cabeza de proyección produciendo 512 características dimensionales.
        Imágenes de entrada de cualquier tamaño soportado debido a AdaptiveAvgPool2d.

        Arquitectura:
          - Bloque 1: Conv2d(3,64) → BN → ReLU → MaxPool2d(2)
          - Bloque 2: Conv2d(64,128) → BN → ReLU → MaxPool2d(2)
          - Bloque 3: Conv2d(128,256) → BN → ReLU → MaxPool2d(2)
          - AdaptiveAvgPool2d((1,1))
          - Cabeza FC: Flatten → Linear(256,512) → ReLU

        :returns: None
        :rtype: None

        :raises None
        """
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
        """
        Extrae características de 512 dimensiones del batch de imágenes.

        Pasa batch de entrada a través de capas de extracción de características
        convolucionales y cabeza totalmente conectada.

        :param x: Tensor batch de imágenes (batch_size, 3, height, width) en rango [0, 1] o [0, 255]
        :type x: torch.Tensor

        :returns: Tensores de características (batch_size, 512) float32
        :rtype: torch.Tensor

        :raises RuntimeError: Si ancho/alto de entrada < 8 (incompatible con strides de MaxPool)
        """
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
        """
        Inicializa el extractor CNN para feature extraction desde ImageNet.
        
        :param arch: Arquitectura CNN ('resnet18' preentrenado o 'simple' aleatorio).
                     - 'resnet18': ResNet-18 con pesos IMAGENET1K_V1 si pretrained=True
                     - 'simple': CNN de 3 bloques Conv2d sin pretrain (experimentación)
        :type arch: str
        :param pretrained: Si True, carga pesos ImageNet para resnet18 (recomendado).
                          Si False, inicializa con pesos aleatorios (convergencia más lenta).
        :type pretrained: bool
        :param device: Dispositivo PyTorch ('cpu', 'cuda', 'cuda:0', 'mps').
                      Modelos congelados: se copian a este device y se cargaán una sola vez.
        :type device: str
        :param seed: Semilla RNG para PyTorch (solo afecta 'simple').
                    Si None, no se fija ningún seed.
        :type seed: Optional[int]
        
        :raises ValueError: Si arch no está en ARCHITECTURES.
        """
        if arch not in self.ARCHITECTURES:
            raise ValueError(f"arch debe ser {self.ARCHITECTURES}, recibido: {arch!r}")

        self.arch = arch
        self.pretrained = pretrained
        self.seed = seed
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)

        self._model = self._build(arch, pretrained).to(self.device)
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()

    @staticmethod
    def _build(arch: str, pretrained: bool) -> nn.Module:
        """
        Construye y retorna la arquitectura CNN solicitada.

        Instancia ResNet-18 con opción de preentrenamiento ImageNet o
        la CNN personalizada SimpleCNN. Ambas arquitecturas producen 512 características.

        :param arch: Nombre de arquitectura ('resnet18' o 'simple')
        :type arch: str
        :param pretrained: Si True y arch='resnet18', carga pesos IMAGENET1K_V1. \
                          Si arch='simple', parámetro ignorado (sin pesos preentrenados disponibles).
        :type pretrained: bool

        :returns: Módulo PyTorch inicializado
        :rtype: nn.Module

        :raises ValueError: Si arch no está en ['resnet18', 'simple']
        """
        if arch == "simple":
            return _SimpleCNN()

        import torchvision.models as tvm

        weights = "IMAGENET1K_V1" if pretrained else None
        model = tvm.resnet18(weights=weights)
        model.fc = nn.Identity()  # type: ignore  # expone vector de 512 features
        return model

    @property
    def feature_dim(self) -> int:
        """
        Obtiene dimensión de característica de salida de este CNN.

        Siempre retorna 512 (fijo por ambas arquitecturas ResNet-18 y SimpleCNN).

        :returns: Dimensionalidad del vector de características
        :rtype: int

        :raises None
        """
        return FEATURE_DIM

    # ── Serialización de pesos (para distribuir por TCP) ─────────

    def _get_weights_bytes(self) -> bytes:
        """
        Serializa pesos del modelo CNN a bytes para distribución por TCP.

        Exporta el state_dict completo del modelo usando torch.save() a buffer BytesIO,
        produciendo string de bytes para transmisión por red a Workers. Usado por
        ParameterServer para empaquetar y difundir pesos CNN actualizados.

        Formato: torch.save(state_dict, BytesIO) → bytes

        :returns: State_dict serializado (formato PyTorch)
        :rtype: bytes

        :raises RuntimeError: Si serialización de state_dict falla (modelo corrupto)
        """
        buf = io.BytesIO()
        torch.save(self._model.state_dict(), buf)
        return buf.getvalue()

    def load_weights_from_bytes(self, weights_bytes: bytes) -> None:
        """
        Carga pesos del modelo CNN desde bytes serializados (del ParameterServer).

        Deserializa state_dict recibido por TCP (formato torch.save) y aplica
        al modelo local. Usado durante loop de entrenamiento del Worker cuando
        se sincroniza pesos CNN globales del ParameterServer para cómputo de gradientes.

        Establece modelo a modo eval después de cargar para deshabilitar actualizaciones
        de dropout/batch norm.

        :param weights_bytes: State_dict serializado de ParameterServer._get_weights_bytes()
        :type weights_bytes: bytes

        :returns: None
        :rtype: None

        :raises RuntimeError: Si pesos incompatibles con arquitectura actual
        :raises pickle.UnpicklingError: Si stream de bytes corrupto o formato inválido
        """
        buf = io.BytesIO(weights_bytes)
        state = torch.load(buf, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

    def _weights_hash(self) -> str:
        """
        Computa hash MD5 de pesos actuales del modelo (primeros 8 caracteres hex).

        Úsin para logging y debugging para verificar sincronización de pesos entre
        ParameterServer y Workers. Mismo pesos producen mismo hash.

        :returns: Primeros 8 caracteres del digest hex MD5 de tensores de pesos concatenados
        :rtype: str

        :raises None
        """
        h = hashlib.md5()
        for t in self._model.state_dict().values():
            h.update(t.cpu().numpy().tobytes())
        return h.hexdigest()[:8]

    # ── Extracción de features ────────────────────────────────────

    def set_trainable(self, trainable: bool) -> None:
        """
        Habilita o deshabilita cómputo de gradientes y modo train/eval para el CNN.

        Establece requires_grad en todos los parámetros e intercambia modelo entre train()
        (habilita dropout, actualizaciones de batch norm) y eval() (inferencia determinista).

        :param trainable: Si True, habilita gradientes y modo train. Si False, deshabilita \
                         gradientes e intercambia a modo eval (solo-inferencia).
        :type trainable: bool

        :returns: None
        :rtype: None

        :raises None
        """
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
        Extrae caracter\u00edsticas de CNN en mini-batches para gestionar memoria eficientemente.

        Usado por ParameterServer durante evaluaci\u00f3n de validaci\u00f3n para procesar splits\n        de validaci\u00f3n grandes sin agotar VRAM. Procesa array de entrada en chunks\n        configurables, acumulando resultados.\n\n        :param X: Batch de imagen de entrada (N, 3, height, width) float32 en [0,1] o [0,255]\n        :type X: np.ndarray\n        :param batch_size: Im\u00e1genes por forward pass (default: 512, ajustar para VRAM)\n        :type batch_size: int\n        :param verbose: Si True, imprime progreso a stdout\n        :type verbose: bool\n\n        :returns: Caracter\u00edsticas extra\u00eddas (N, 512) float32\n        :rtype: np.ndarray\n\n        :raises RuntimeError: Si modelo en modo training (llamar set_trainable(False) primero)\n        :raises OutOfMemoryError: Si batch_size demasiado grande para VRAM disponible\n        """
        N = len(X)
        parts = []
        starts = range(0, N, batch_size)

        for i, start in enumerate(starts, 1):
            chunk = X[start : start + batch_size]
            with torch.inference_mode():
                t = torch.from_numpy(chunk).to(self.device)
                parts.append(self._model(t).cpu().numpy())
            if verbose:
                done = min(start + batch_size, N)
                print(
                    f"\r  [CNN] {done}/{N} imgs ({i}/{len(starts)} batches)",
                    end="",
                    flush=True,
                )

        if verbose:
            print()
        return np.concatenate(parts, axis=0)
