"""
Model/cnn_extractor.py

Extractor CNN para ImageNet con PyTorch.

ARQUITECTURAS:
  "resnet18" → ResNet-18 con pesos ImageNet (siempre preentrenada).
               Carga IMAGENET1K_V1 weights automáticamente.
               feature_dim = 512. Sin capa de clasificación.
               Se construye CONGELADA (requires_grad=False, eval).
               Semántica: extractor de características fijo.

  "simple"   → CNN propia de 3 bloques Conv→BN→ReLU→MaxPool.
               feature_dim = 512. Sin pesos preentrenados.
               Se construye ENTRENABLE (requires_grad=True, eval).
               Semántica: red entrenable desde cero en modo E2E.

DIFERENCIA CLAVE VS VERSIÓN ANTERIOR:
  El requires_grad y preentrenamiento se determinan automáticamente por
  arquitectura en __init__, sin parámetro `pretrained` ni dependencias frágiles.
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
                nn.BatchNorm2d(
                    out_ch
                ),  # Normaliza cada canal, haciendo el entrenamiento más estable
                nn.ReLU(inplace=True),  # Introduce no-linealidad
                nn.MaxPool2d(2),  # Reduce tamaño a la mitad
            )

        self.features = nn.Sequential(
            _block(3, 64),
            _block(64, 128),
            _block(128, 256),
            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),  # Convierte cualquier tamaño en (batch_size, 256, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),  # Pasa de (256,1,1) a (256)
            nn.Linear(
                256, FEATURE_DIM
            ),  # Crea el vector final de características (256 -> 512)
            nn.ReLU(inplace=True),  # Añade no-linealidad final
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

    El estado inicial de requires_grad refleja la semántica de cada arquitectura:

      "resnet18": congelada desde la construcción (requires_grad=False).
                  Nunca necesita gradientes — es un extractor fijo.
                  _train_batch no necesita gestionar su requires_grad.

      "simple":   entrenable desde la construcción (requires_grad=True).
                  Participa en backprop en cada batch E2E.
                  _train_batch no necesita activar requires_grad manualmente.

    Ambas se construyen en eval() para inferencia determinista de BN.
    _train_batch pone 'simple' en train() antes del forward para actualizar
    las running stats de BatchNorm durante el entrenamiento.

    :param arch:   'resnet18' o 'simple' (determina automáticamente pretrained + requires_grad).
    :param device: Dispositivo PyTorch ('cpu', 'cuda', 'mps').
    :param seed:   Semilla de inicialización (solo afecta a 'simple').
    """

    ARCHITECTURES = ("simple", "resnet18")

    def __init__(
        self,
        arch: str = "resnet18",
        device: str = "cpu",
        seed: Optional[int] = 42,
    ) -> None:
        """
        Inicializa el extractor CNN para feature extraction desde ImageNet.

        La comportamiento de preentrenamiento se determina automáticamente por arquitectura:
        - 'resnet18': Carga pesos IMAGENET1K_V1, congelada permanentemente.
        - 'simple': CNN de 3 bloques Conv2d sin pretrain, entrenable E2E.

        :param arch: Arquitectura CNN ('resnet18' o 'simple').
                     - 'resnet18': ResNet-18 con pesos ImageNet (congelada).
                     - 'simple': CNN personalizada de 3 bloques (entrenable).
        :type arch: str
        :param device: Dispositivo PyTorch ('cpu', 'cuda', 'cuda:0', 'mps').
                      Modelos congelados: se copian a este device y se cargan una sola vez.
        :type device: str
        :param seed: Semilla RNG para PyTorch (solo afecta 'simple').
                    Si None, no se fija ningún seed.
        :type seed: Optional[int]

        :raises ValueError: Si arch no está en ARCHITECTURES.
        """
        if arch not in self.ARCHITECTURES:
            raise ValueError(f"arch debe ser {self.ARCHITECTURES}, recibido: {arch!r}")

        self.arch = arch
        self.seed = seed
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)

        # Determinar pretrained automáticamente por arquitectura:
        # resnet18 → siempre con ImageNet weights (congelada)
        # simple → sin preentrenamiento (entrenable)
        pretrained = arch == "resnet18"
        self._model = self._build(arch, pretrained).to(self.device)

        # El estado de requires_grad refleja la semántica permanente de la arquitectura:
        #   resnet18 → congelada: nunca necesita gradientes
        #   simple   → entrenable: participa en E2E backprop
        #
        # Esto evita que _train_batch tenga que gestionar requires_grad en cada iteración,
        # eliminando la dependencia frágil de orden de ejecución que existía antes.
        trainable = arch == "simple"
        for p in self._model.parameters():
            p.requires_grad_(trainable)  # Controla si PyTorch calcula gradientes

        # Ambas en eval() inicialmente. _train_batch pondrá 'simple' en train()
        # antes del forward para que BatchNorm actualice sus running stats.
        self._model.eval()

    @staticmethod
    def _build(arch: str, pretrained: bool) -> nn.Module:
        """
        Construye y retorna la arquitectura CNN solicitada.

        Instancia ResNet-18 con preentrenamiento ImageNet o la CNN
        personalizada SimpleCNN. Ambas arquitecturas producen 512 características.

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

    # -------------------------- Serialización de pesos (para distribuir por TCP)--------------------------

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
        buf = io.BytesIO()  # Crea un buffer en memoria
        torch.save(self._model.state_dict(), buf)
        return buf.getvalue()  # Retorna el paquete de bytes

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
        buf = io.BytesIO(weights_bytes)  # Lee desde buffer
        state = torch.load(buf, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state)  # Reemplaza todos los pesos actuales
        self._model.eval()

    def _weights_hash(self) -> str:
        """
        Computa hash MD5 de pesos actuales del modelo (primeros 8 caracteres hex).

        Se puede usar para logging y debugging para verificar sincronización de pesos entre
        ParameterServer y Workers. Mismo pesos producen mismo hash.

        :returns: Primeros 8 caracteres del digest hex MD5 de tensores de pesos concatenados
        :rtype: str

        :raises None
        """
        hash = hashlib.md5()
        for t in self._model.state_dict().values():
            hash.update(t.cpu().numpy().tobytes())
        return hash.hexdigest()[:8]

    # -------------------------- Extracción de features --------------------------

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
        Extrae features en mini-batches.

        Usado por el PS durante la evaluación de validación.

        :param X: Imágenes (N, 3, H, W) float32.
        :type X: np.ndarray

        :param batch_size: Imágenes por batch.
        :type batch_size: int

        :param verbose: Imprimir progreso.
        :type verbose: bool

        :return: Features (N, feature_dim) float32.
        """
        num_images = len(X)

        # Permite recorrer por bloques
        parts = []
        starts = range(0, num_images, batch_size)

        for index, start in enumerate(starts, 1):
            chunk = X[start : start + batch_size]
            with torch.inference_mode():  # No calcula gradientes
                t = torch.from_numpy(chunk).to(self.device)  # Convierte NumPy a PyTorch
                parts.append(
                    self._model(t).cpu().numpy()
                )  # Pasa por la red y vuelve a NumPy
            if verbose:
                done = min(start + batch_size, num_images)
                print(
                    f"\r  [CNN] {done}/{num_images} imgs ({index}/{len(starts)} batches)",
                    end="",
                    flush=True,
                )

        if verbose:
            print()
        return np.concatenate(parts, axis=0)  # Junta todos los batches
