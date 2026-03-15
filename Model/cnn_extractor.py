"""
Model/cnn_extractor.py

Extractor de características convolucional para CIFAR-10 basado en PyTorch.

──────────────────────────────────────────────────────────────────
ROL EN LA ARQUITECTURA DISTRIBUIDA
──────────────────────────────────────────────────────────────────
La red completa se divide en dos etapas con responsabilidades distintas:

    ┌─────────────────────────────────────────────────────────────┐
    │  CIFAR-10 image  (3 × 32 × 32)                              │
    │         │                                                   │
    │         ▼                                                   │
    │  ┌─────────────────┐                                        │
    │  │  CNN Extractor  │  ← Este módulo (PyTorch)               │
    │  │  (convolucional)│    Pesos FIJOS durante el distribuido  │
    │  └────────┬────────┘                                        │
    │           │  feature vector  (feature_dim,)                 │
    │           ▼                                                 │
    │  ┌─────────────────┐                                        │
    │  │   MLP NumPy     │  ← Model/mlp.py                        │
    │  │  (clasificador) │    Pesos entrenados por el PS/Workers  │
    │  └────────┬────────┘                                        │
    │           │                                                 │
    │           ▼                                                 │
    │   10 clases CIFAR-10                                        │
    └─────────────────────────────────────────────────────────────┘

──────────────────────────────────────────────────────────────────
¿POR QUÉ CONGELAR LA CNN Y ENTRENAR SOLO EL MLP?
──────────────────────────────────────────────────────────────────
El Algoritmo de Diego distribuye el cálculo de gradientes entre
Workers y promedia en el PS. Esto funciona directamente sobre el
MLP (NumPy puro, gradientes serializables con Pickle).

Para la CNN, propagar gradientes por red en cada época multiplicaría
el tráfico ×100 (pesos convolucionales >> pesos MLP) y añadiría
complejidad al protocolo sin beneficio pedagógico.

En cambio, usar la CNN como extractor de features fijos es una
práctica real de ingeniería (transfer learning) y permite mantener
el principio pedagógico del Algoritmo de Diego: los Workers calculan
gradientes del MLP sobre sus chunks, el PS los promedia y actualiza.

──────────────────────────────────────────────────────────────────
ARQUITECTURAS DISPONIBLES
──────────────────────────────────────────────────────────────────
  "simple"   → CNN diseñada desde cero, 3 bloques conv.
               Sin pesos pretrained. Pedagógicamente transparente.
               feature_dim = 512

  "resnet18" → ResNet-18 de torchvision, pesos ImageNet opcionales.
               feature_dim = 512

La arquitectura se especifica al construir el extractor y queda
fija para toda la sesión. Los Workers no la negocian con el PS;
simplemente deben usar la misma al iniciar.

──────────────────────────────────────────────────────────────────
PREPROCESO DE IMÁGENES
──────────────────────────────────────────────────────────────────
La CNN espera imágenes de forma (N, 3, 32, 32), normalizadas
canal a canal con la media y std estándar de CIFAR-10:

    μ = [0.4914, 0.4822, 0.4465]
    σ = [0.2470, 0.2435, 0.2616]

El loader (Utils/cifar_loader.py) devuelve los datos en ese formato.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Constante exportada ───────────────────────────────────────────
# Todos los módulos que necesiten saber el tamaño del vector de
# features lo leen aquí, evitando números mágicos dispersos.
FEATURE_DIM = 512


# ================================================================
# CNN "SIMPLE" — diseñada desde cero, sin pretraining
# ================================================================


class _SimpleCNN(nn.Module):
    """
    CNN de 3 bloques convolucionales diseñada para CIFAR-10 (32×32).

    Bloque = Conv2d → BatchNorm → ReLU → MaxPool

    Motivación pedagógica de cada decisión:

    • BatchNorm: normaliza las activaciones de cada capa durante el
      entrenamiento, lo que estabiliza el gradiente y permite tasas
      de aprendizaje más altas.

    • MaxPool 2×2: reduce las dimensiones espaciales a la mitad en
      cada bloque. Después de 3 bloques: 32 → 16 → 8 → 4 px.
      Proporciona invarianza a pequeñas traslaciones.

    • Stride=1 en todas las convoluciones: MaxPool hace el downsampling.
      Separar los dos roles (extracción vs. reducción) hace la
      arquitectura más legible y fácil de ajustar.

    Flujo de dimensiones (batch B ignorado):
        (3, 32, 32) → conv1 → (64, 32, 32) → pool → (64, 16, 16)
                    → conv2 → (128, 16, 16) → pool → (128, 8, 8)
                    → conv3 → (256, 8, 8)   → pool → (256, 4, 4)
                    → AdaptiveAvgPool(1, 1) → (256, 1, 1)
                    → Flatten → (256,)
                    → fc → (512,)

    La capa fc proyecta a FEATURE_DIM=512 para tener la misma
    interfaz que ResNet-18 y facilitar la comparación entre
    arquitecturas sin cambiar el MLP.
    """

    def __init__(self) -> None:
        super().__init__()

        def _block(in_ch: int, out_ch: int) -> nn.Sequential:
            # Sequential significa que se ejecuta en orden
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            _block(3, 64),  # (3,32,32) → (64,16,16)
            _block(64, 128),  # (64,16,16) → (128,8,8)
            _block(128, 256),  # (128,8,8) → (256,4,4)
            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),  # → (256,1,1) — robusto ante cambios de input size
        )

        # Proyección a FEATURE_DIM para unificar la interfaz
        self.fc = nn.Sequential(
            nn.Flatten(),  # (256,1,1) → (256,)
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
    Envuelve una CNN PyTorch y expone una interfaz NumPy pura.

    El Worker la instancia una vez al arrancar. A partir de ese
    momento llama a ``extract(X_raw)`` en cada época para obtener
    el vector de features que alimenta al MLP NumPy.

    :param arch: Arquitectura CNN. ``"simple"`` (default) o ``"resnet18"``.
    :type arch: str

    :param pretrained: Solo relevante con ``arch="resnet18"``. Si True,
                       descarga pesos ImageNet. Default False para no
                       requerir conexión en cada arranque de Worker.
    :type pretrained: bool

    :param device: Dispositivo PyTorch: ``"cpu"``, ``"cuda"``, ``"mps"``.
                   Default ``"cpu"``; la CNN es pequeña y el cuello de
                   botella real está en la comunicación TCP, no en la extracción.
    :type device: str

    :param seed: Semilla para inicialización aleatoria de la CNN simple.
                 Garantiza que todos los Workers usen exactamente los
                 mismos pesos del extractor (crítico para reproducibilidad).
    :type seed: int | None
    """

    ARCHITECTURES = ("simple", "resnet18")

    def __init__(
        self,
        arch: str = "simple",
        pretrained: bool = False,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        if arch not in self.ARCHITECTURES:
            raise ValueError(
                f"Arquitectura desconocida: {arch!r}. Opciones: {self.ARCHITECTURES}"
            )

        self.arch = arch

        # Convierte el string en un objeto PyTorch que controla dónde correr la CNN
        self.device = torch.device(device)

        # Semilla antes de construir la red para reproducibilidad
        if seed is not None:
            torch.manual_seed(seed)

        self._model = self._build(arch, pretrained).to(self.device)

        # Congela todos los parámetros: la CNN es un extractor fijo.
        # El gradiente solo fluye por el MLP NumPy, que es lo que el
        # PS promedia. Esto mantiene el Algoritmo de Diego intacto.
        for param in self._model.parameters():
            param.requires_grad_(False)

        self._model.eval()  # BatchNorm en modo inferencia desde el inicio

    # ── Construcción de modelos ───────────────────────────────────

    @staticmethod
    def _build(arch: str, pretrained: bool) -> nn.Module:
        if arch == "simple":
            return _SimpleCNN()

        # ResNet-18: elimina la capa de clasificación original (fc)
        # para exponer el vector de 512 features antes de la clasificación.
        import torchvision.models as tvm

        # ResNet-18 es una CNN mucho más profunda que tiene 18 capas
        # convolucionales. Al final produce un vector de 512 features
        # antes de la capa de clasificación
        weights = "IMAGENET1K_V1" if pretrained else None
        model = tvm.resnet18(weights=weights)

        # Elimina cabeza clasificadora → output (512,)
        # La capa fc de ResNet-18 original convierte los 512 features en 1000 clases de ImageNet,
        # nn.Identity() reemplaza esa capa con una función que no hace nada, así la CNN devuelve
        # el vector de 512 features, sin convertirlo en clases
        model.fc = nn.Identity()  # type: ignore
        return model

    # ── Interfaz pública ──────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        """Dimensión del vector de features producido por la CNN."""
        return FEATURE_DIM

    def extract(self, X: np.ndarray) -> np.ndarray:
        """
        Extrae features de un batch de imágenes CIFAR-10.

        :param X: Imágenes normalizadas, forma ``(N, 3, 32, 32)``, float32.
        :type X: np.ndarray

        :return: Matriz de features ``(N, feature_dim)``, float32.
        :rtype: np.ndarray
        """
        # torch.no_grad() evita reservar memoria para el grafo de cómputo,
        # reduciendo el uso de RAM a la mitad durante la inferencia.
        with torch.no_grad():
            t = torch.from_numpy(X).to(self.device)
            features = self._model(t)  # Ejecuta la CNN sin gradientes
            return features.cpu().numpy()

    def extract_batched(
        self,
        X: np.ndarray,
        batch_size: int = 512,
    ) -> np.ndarray:
        """
        Extrae features en mini-batches para no saturar la RAM/VRAM.

        Para CIFAR-10 completo (50 000 imágenes × 3×32×32) el tensor
        pesa ~590 MB en float32. Procesarlo en chunks de 512 imágenes
        mantiene el pico de memoria por debajo de 12 MB por batch.

        :param X: Imágenes ``(N, 3, 32, 32)``, float32.
        :type X: np.ndarray

        :param batch_size: Número de imágenes por paso.
        :type batch_size: int = 512

        :return: Features ``(N, feature_dim)``, float32.
        """
        parts = []
        for i in range(0, len(X), batch_size):
            parts.append(self.extract(X[i : i + batch_size]))

        # Devuelve el mismo resultado que extract, pero por partes y concatenado
        return np.concatenate(parts, axis=0)

    def get_state(self) -> bytes:
        """
        Serializa los pesos de la CNN como bytes.

        Útil para que el PS verifique que todos los Workers usan
        exactamente el mismo extractor (mismo hash de estado).
        No forma parte del protocolo mínimo pero facilita debugging.

        :return: Bytes del state_dict serializado con torch.save.
        :rtype: bytes
        """
        import io

        buf = io.BytesIO()
        torch.save(self._model.state_dict(), buf)
        return buf.getvalue()
