"""
Model/cnn_extractor.py

Extractor CNN para ImageNet con PyTorch.

ARQUITECTURAS:
  "resnet18" → ResNet-18 con pesos ImageNet (siempre preentrenada).
               Carga IMAGENET1K_V1 weights automáticamente.
               feature_dim = 512. Sin capa de clasificación.
               Se construye CONGELADA (requires_grad=False, eval).
               Semántica: extractor de características fijo.

  "simple"   → CNN propia con bloques residuales ligeros (ResBlockLite).
               4 bloques residuales + proyección a 512.
               feature_dim = 512. Sin pesos preentrenados.
               Se construye ENTRENABLE (requires_grad=True, eval).
               Semántica: red entrenable desde cero en modo E2E.

MEJORAS VS VERSIÓN ANTERIOR (_SimpleCNN):
  - 3 bloques sin skip → 4 ResBlockLite con skip connections
    Ventaja: gradientes fluyen directamente a capas iniciales (sin vanishing)
  - MaxPool entre bloques → stride=2 en Conv (preserva información espacial)
  - Sin fc expansiva (256→512) → conv final produce 256ch + proj fc(256→512)
  - Zero-init de BN final de cada bloque residual (estabiliza inicio E2E)
  - Dropout(0.1) antes de la proyección (regularización ligera)
  - Kaiming Normal init (mejor que Uniform para redes con skip)

CLAVE:
  El requires_grad y preentrenamiento se determinan automáticamente por
  arquitectura en __init__, sin parámetro `pretrained` ni dependencias frágiles.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 512


# ================================================================
# BLOQUE RESIDUAL LIGERO
# ================================================================


class _ResBlockLite(nn.Module):
    """
    Bloque residual ligero: Conv→BN→ReLU→Conv→BN + shortcut.

    Beneficios sobre un bloque conv simple:
      - Skip connection: gradiente fluye directamente desde la salida
        a la entrada, evitando vanishing gradient en E2E training.
      - Zero-init del BN final: al inicio, el bloque actúa como identidad
        (out ≈ shortcut), lo que estabiliza las primeras iteraciones de SGD.
      - Stride en la primera Conv (en lugar de MaxPool): preserva más
        información espacial que el pooling de máximo.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        """
        Inicializa un bloque residual ligero con stride opcional.

        :param in_ch: Número de canales de entrada.
        :type in_ch: int

        :param out_ch: Número de canales de salida.
        :type out_ch: int

        :param stride: Stride de la primera convolución (1 = sin submuestreo, 2 = submuestreo).
        :type stride: int

        :returns: None
        :rtype: None
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Shortcut: proyección 1×1 si hay cambio de canales o stride>1
        self.shortcut: nn.Module = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

        # Zero-init del BN final del camino residual:
        # Hace que el bloque inicie como identidad (out = shortcut + 0·residual).
        # Con todos los bloques iniciados así, la red completa aproxima la identidad
        # al inicio, lo que estabiliza el loss en las primeras iteraciones E2E.
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


# ================================================================
# CNN SIMPLE MEJORADA
# ================================================================


class _SimpleCNN(nn.Module):
    """
    CNN con 4 bloques residuales ligeros + proyección a 512.

    Arquitectura:
      Stem:   Conv(3→32, stride=2) + BN + ReLU
      Layer1: ResBlockLite(32→32,  stride=1)
      Layer2: ResBlockLite(32→64,  stride=2)
      Layer3: ResBlockLite(64→128, stride=2)
      Layer4: ResBlockLite(128→256, stride=2)
      GAP:    AdaptiveAvgPool2d(1×1)
      Drop:   Dropout(p=0.1)
      Proj:   Linear(256→512)

    Con imagen 224×224:
      Stem   → 112×112
      Layer2 →  56×56
      Layer3 →  28×28
      Layer4 →  14×14
      GAP    →   1×1
      Proj   →    512

    Parámetros: ~1.36M (vs 502K de la versión anterior).
    El overhead de serialización TCP es ~5 MB por round-trip,
    manejable en redes locales (LAN) y razonable incluso en WAN.

    Ventajas sobre la versión anterior:
      - Skip connections: gradiente llega sin atenuación a las capas iniciales
      - Stride=2 en Conv: preserva información espacial (vs MaxPool que descarta)
      - Zero-init BN final: estabiliza las primeras iteraciones E2E
      - Dropout(0.1): regularización ligera, reduce coadaptación
      - Kaiming Normal: mejor calibración de varianza para redes profundas
    """

    def __init__(self) -> None:
        super().__init__()

        # Stem: reducción inicial de resolución sin pérdida de información
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 4 bloques residuales en progresión de canales
        self.layer1 = _ResBlockLite(32, 32, stride=1)
        self.layer2 = _ResBlockLite(32, 64, stride=2)
        self.layer3 = _ResBlockLite(64, 128, stride=2)
        self.layer4 = _ResBlockLite(128, 256, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.1)
        self.proj = nn.Linear(256, FEATURE_DIM)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Kaiming Normal para Conv y Linear; ones/zeros para BN.

        No sobreescribe el zero-init ya aplicado en _ResBlockLite.bn2.weight.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                # bn2.weight en ResBlockLite ya fue zero-init — no sobreescribir
                if m.weight is not None and m.weight.data.abs().sum() > 0:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa un batch de imágenes a través de la CNN simple mejorada.

        :param x: Batch de imágenes (B, 3, H, W).
        :type x: torch.Tensor

        :returns: Características extraídas (B, feature_dim).
        :rtype: torch.Tensor
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = self.dropout(x)
        return self.proj(x.flatten(1))


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

    :param arch:   'resnet18' o 'simple' (determina automáticamente trainable + pesos).
    :param device: Dispositivo PyTorch ('cpu', 'cuda', 'mps').
    :param seed:   Semilla RNG (default: None = aleatorio, solo afecta a 'simple').
    """

    ARCHITECTURES = ("simple", "resnet18")

    def __init__(
        self,
        arch: str = "resnet18",
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        """
        Inicializa el extractor CNN con la arquitectura especificada.

        :param arch: Arquitectura seleccionada ('resnet18', 'resnet50', o 'simple').
        :type arch: str

        :param device: Dispositivo PyTorch ('cpu', 'cuda', etc.).
        :type device: str

        :param seed: Semilla para reproducibilidad (None = sin fijar).
        :type seed: Optional[int]

        :returns: None
        :rtype: None

        :raises ValueError: Si arch no está en ARCHITECTURES permitidas.
        """
        if arch not in self.ARCHITECTURES:
            raise ValueError(f"arch debe ser {self.ARCHITECTURES}, recibido: {arch!r}")

        self.arch = arch
        self.seed = seed
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)

        trainable = arch == "simple"
        self._model = self._build(arch, trainable).to(self.device)

        for p in self._model.parameters():
            p.requires_grad_(trainable)

        self._model.eval()

    @staticmethod
    def _build(arch: str, trainable: bool) -> nn.Module:
        """
        Construye la arquitectura CNN especificada.

        :param arch: Tipo de arquitectura a construir ('simple', 'resnet18', 'resnet50').
        :type arch: str
        :param trainable: Si True, descarga pesos pre-entrenados; si False, usa de ImageNet.
        :type trainable: bool
        :returns: Módulo PyTorch construido.
        :rtype: nn.Module
        """
        if arch == "simple":
            return _SimpleCNN()

        import torchvision.models as tvm

        weights = "IMAGENET1K_V1" if not trainable else None
        model = tvm.resnet18(weights=weights)
        model.fc = nn.Identity()  # type: ignore[assignment]
        return model

    @property
    def feature_dim(self) -> int:
        """
        Retorna la dimensión de las características extraídas.

        :returns: Dimensión del vector de características.
        :rtype: int
        """
        return FEATURE_DIM

    # ── Serialización ─────────────────────────────────────────────

    def _get_weights_bytes(self) -> bytes:
        """
        Serializa el state_dict del modelo a bytes para envío por TCP.

        :returns: Representación binaria del estado del modelo.
        :rtype: bytes
        """
        buf = io.BytesIO()
        torch.save(self._model.state_dict(), buf)
        return buf.getvalue()

    def load_weights_from_bytes(self, weights_bytes: bytes) -> None:
        """
        Carga pesos desde bytes recibidos del Servidor de Parámetros por TCP.

        load_state_dict no modifica requires_grad — el estado correcto
        establecido en __init__ se preserva después de cada sincronización.

        :param weights_bytes: Representación binaria del estado a cargar.
        :type weights_bytes: bytes

        :returns: None
        :rtype: None
        """
        buf = io.BytesIO(weights_bytes)
        state = torch.load(buf, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
