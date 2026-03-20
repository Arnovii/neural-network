"""
Model/cnn_extractor.py

Extractor de características convolucional para CIFAR-10 basado en PyTorch.

──────────────────────────────────────────────────────────────────
ROL EN LA ARQUITECTURA DISTRIBUIDA
──────────────────────────────────────────────────────────────────
    imagen (3×32×32)
        │
        ▼  CNN Extractor — PyTorch, pesos FIJOS tras preentrenamiento
        │
        ▼  feature vector (feature_dim,)
        │
        ▼  MLP NumPy — pesos DISTRIBUIDOS por el Algoritmo de Diego

──────────────────────────────────────────────────────────────────
CACHÉ EN DOS NIVELES
──────────────────────────────────────────────────────────────────
Data/feature_cache/

  Nivel 1 — Pesos CNN:
    {arch}_{seed}_weights.pt

  Nivel 2 — Features extraídos:
    {arch}_{weights_hash8}_{split}_X.npy
    {arch}_{weights_hash8}_{split}_Y.npy

La clave de features incluye un hash MD5 (8 hex) de los pesos CNN
actuales. Esto garantiza que si la CNN tiene pesos distintos (aleatorios
vs preentrenados), los archivos de caché son distintos y nunca se mezclan.

Flujo en prepare():
    ¿Features en caché (con hash actual)? → carga instantánea  (< 0.3 s)
    ¿No? → ¿Pesos en caché?               → carga pesos        (< 0.5 s)
         → ¿No, arch=simple?              → pretrain           (~1-2 min)
    → extrae features + guarda caché                           (~30 s)

──────────────────────────────────────────────────────────────────
ARQUITECTURAS
──────────────────────────────────────────────────────────────────
  "simple"   → CNN propia, 3 bloques Conv→BN→ReLU→MaxPool.
               feature_dim = 512. Preentrenamiento local la 1ª vez.

  "resnet18" → ResNet-18 torchvision.
               Con pretrained=True: pesos ImageNet.
               feature_dim = 512.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Constante exportada ───────────────────────────────────────────
# Todos los módulos que necesiten saber el tamaño del vector de
# features lo leen aquí, evitando números mágicos dispersos.
FEATURE_DIM = 512


# ================================================================
# CNN SIMPLE — diseñada desde cero
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
        """
        Forward pass a través de la CNN.

        :param x: Imágenes de entrada.
        :type x: torch.Tensor de shape (batch_size, 3, 32, 32) float32.

        :return: Vector de características extraído.
        :rtype: torch.Tensor de shape (batch_size, FEATURE_DIM) float32.
        """
        return self.fc(self.features(x))


# ================================================================
# EXTRACTOR PÚBLICO
# ================================================================


class CNNExtractor:
    """
    Envuelve una CNN PyTorch y expone una interfaz NumPy pura.

    Uso típico en el Worker:
        cnn = CNNExtractor(arch="simple", seed=42)
        X_feat, Y = cnn.prepare(X_train, Y_train, split="train")

    Uso típico en el PS (evaluación, sin reentrenar):
        cnn = CNNExtractor(arch="simple", seed=42)
        X_feat, Y = cnn.prepare(X_test, Y_test, split="test", pretrain_epochs=0)

    :param arch: ``"simple"`` o ``"resnet18"``.
    :param pretrained: Pesos ImageNet para resnet18 (ignorado para simple).
    :param device: ``"cpu"``, ``"cuda"`` o ``"mps"``.
    :param seed: Semilla de inicialización. Misma en PS y Workers.
    :param cache_dir: Directorio de caché. None = Data/feature_cache/.
    """

    ARCHITECTURES = ("simple", "resnet18")

    def __init__(
        self,
        arch: str = "simple",
        pretrained: bool = False,
        device: str = "cpu",
        seed: int | None = 42,
        cache_dir: str | None = None,
    ) -> None:
        if arch not in self.ARCHITECTURES:
            raise ValueError(
                f"Arch debe ser uno de {self.ARCHITECTURES}, recibido: {arch!r}"
            )

        self.arch = arch
        self.pretrained = pretrained
        self.seed = seed

        # Convierte el string en un objeto PyTorch que controla dónde correr la CNN
        self.device = torch.device(device)

        self._cache_dir = cache_dir or self._default_cache_dir()
        os.makedirs(self._cache_dir, exist_ok=True)

        # Semilla antes de construir la red para reproducibilidad
        if seed is not None:
            torch.manual_seed(seed)

        base = self._build(arch, pretrained)

        # ResNet-18 fue diseñado para 224×224 (ImageNet). CIFAR-10 tiene
        # imágenes de 32×32. Envolver la red con un upscale automático
        # permite aprovechar los pesos ImageNet correctamente.
        if arch == "resnet18":
            self._model = self._make_resnet_wrapper(base).to(self.device)
        else:
            self._model = base.to(self.device)

        for param in self._model.parameters():
            param.requires_grad_(False)
        self._model.eval()  # BatchNorm en modo inferencia desde el inicio

    # ── Construcción ─────────────────────────────────────────────

    @staticmethod
    def _build(arch: str, pretrained: bool) -> nn.Module:
        """
        Construye la arquitectura CNN especificada.

        :param arch: Arquitectura a construir.
        :type arch: str, debe ser "simple" o "resnet18".

        :param pretrained: Si True, carga pesos ImageNet para ResNet-18.
        :type pretrained: bool.

        :return: Modelo CNN sin entrenar o con pesos preentrenados.
        :rtype: nn.Module (_SimpleCNN o ResNet-18 torchvision).

        :raises ValueError: Si arch no es válido (aunque el chequeo está en __init__).
        """
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
        model.fc = nn.Identity()  # type: ignore  — expone el vector de 512 features
        return model

    def _make_resnet_wrapper(self, base: nn.Module) -> nn.Module:
        """
        Envuelve ResNet-18 con un upscale 32→224 para CIFAR-10.

        ResNet-18 fue diseñado para imágenes ImageNet de 224×224.
        Su primera capa es Conv2d(kernel=7, stride=2) seguida de
        MaxPool(3,2), lo que reduce 224→56→28 px antes del primer bloque.
        Con imágenes de 32×32, esa reducción produce mapas de 8×8,
        demasiado pequeños para que los pesos ImageNet sean efectivos.

        Redimensionar a 224×224 antes del forward permite que la red
        procese las imágenes en la escala para la que fue entrenada,
        obteniendo features de mayor calidad y pasando de ~60% a ~75-80%.

        :param base: Modelo ResNet-18 base.
        :type base: nn.Module.

        :return: ResNet-18 envuelto con interpolación automática.
        :rtype: nn.Module (_ResNetWrapper).
        """

        class _ResNetWrapper(nn.Module):
            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Upscale de 32×32 a 224×224 con interpolación bilineal.
                # antialias=True evita artefactos de aliasing al ampliar.
                x = torch.nn.functional.interpolate(
                    x,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
                return self.model(x)

        return _ResNetWrapper(base)

    @staticmethod
    def _default_cache_dir() -> str:
        """
        Obtiene la ruta por defecto del directorio de caché de features.

        :return: Ruta al directorio Data/feature_cache/ en la raíz del proyecto.
        :rtype: str.
        """
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, "Data", "feature_cache")

    # ── hash de pesos actuales ────────────────────────────────────

    def _weights_hash(self) -> str:
        """
        Calcula el MD5 (8 hex primeros) de los pesos CNN actuales.

        Se usa como parte de la clave de caché de features para garantizar
        que features extraídos con distintos pesos (aleatorios vs preentrenados)
        nunca se mezclen. Calcular el hash de ~2 MB tarda < 5 ms.

        :return: Hash MD5 truncado a 8 caracteres hexadecimales de los pesos.
        :rtype: str.
        """
        h = hashlib.md5()
        for tensor in self._model.state_dict().values():
            h.update(tensor.cpu().numpy().tobytes())
        return h.hexdigest()[:8]

    # ── rutas de caché ────────────────────────────────────────────

    def _weights_cache_path(self) -> str:
        """
        Obtiene la ruta del archivo de caché de pesos CNN.

        :return: Ruta al archivo {arch}_{seed}_weights.pt.
        :rtype: str.
        """
        seed_str = str(self.seed) if self.seed is not None else "none"
        return os.path.join(self._cache_dir, f"{self.arch}_{seed_str}_weights.pt")

    def _feature_cache_paths(self, split: str) -> Tuple[str, str]:
        """
        Calcula las rutas de caché para features con hash de pesos actual.

        La clave es {arch}_{hash}_{split} — no incluye n porque:
        - train: siempre 50 000 imágenes (el slider n_train solo controla
          cuántas usa el round-robin, no cuántas se extraen).
        - test:  siempre 10 000 imágenes en CIFAR-10, sin excepción.
        Incluir n sería ruido que podría causar fallos de caché si se
        llamara con un valor ligeramente distinto.

        :param split: Identificador del conjunto ("train" o "test").
        :type split: str.

        :return: Tupla (ruta_features_X, ruta_features_Y).
        :rtype: Tuple[str, str].
        """
        wh = self._weights_hash()
        key = f"{self.arch}_{wh}_{split}"
        return (
            os.path.join(self._cache_dir, f"{key}_X.npy"),
            os.path.join(self._cache_dir, f"{key}_Y.npy"),
        )

    # ── guardado / carga de pesos ─────────────────────────────────

    def _save_weights(self) -> None:
        """
        Guarda los pesos CNN actuales en el archivo de caché.

        :return: None
        :rtype: NoneType.
        """
        torch.save(self._model.state_dict(), self._weights_cache_path())

    def _get_weights_bytes(self) -> bytes:
        """
        Serializa el state_dict a bytes para enviarlo por TCP.

        El PS llama este método para distribuir sus pesos CNN a los
        Workers via el mensaje CNN_WEIGHTS del protocolo. Los Workers
        llaman a load_weights_from_bytes() con los bytes recibidos.
        Tamaño aproximado: ~2 MB para simple, ~44 MB para resnet18.

        :return: Bytes del state_dict serializados con torch.save.
        :rtype: bytes.
        """
        import io as _io

        buf = _io.BytesIO()
        torch.save(self._model.state_dict(), buf)
        return buf.getvalue()

    def load_weights_from_bytes(self, weights_bytes: bytes) -> None:
        """
        Carga pesos CNN desde bytes recibidos por TCP.

        El Worker llama este método al recibir CNN_WEIGHTS del PS.
        Garantiza que PS y Worker usan exactamente la misma CNN sin
        necesitar filesystem compartido entre máquinas.

        :param weights_bytes: Bytes generados por _get_weights_bytes().
        :type weights_bytes: bytes.

        :return: None
        :rtype: NoneType.
        """
        import io as _io

        buf = _io.BytesIO(weights_bytes)
        state = torch.load(buf, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

    def _metadata_path(self) -> str:
        """
        Ruta del archivo JSON de metadata asociado a los pesos CNN.

        El nombre sigue la misma clave que _weights_cache_path():
        {arch}_{seed}_metadata.json
        Esto garantiza que pesos y metadata siempre van juntos.
        """
        seed_str = str(self.seed) if self.seed is not None else "none"
        return os.path.join(self._cache_dir, f"{self.arch}_{seed_str}_metadata.json")

    def _save_metadata(
        self,
        epochs: int,
        final_loss: float,
        final_acc: float,
        elapsed: float,
    ) -> None:
        """
        Guarda la metadata del preentrenamiento en JSON.

        Se llama al terminar pretrain(). El archivo queda junto
        al .pt con el mismo prefijo de nombre.

        :param epochs: Épocas de preentrenamiento realizadas.
        :param final_loss: Pérdida de la última época.
        :param final_acc: Precisión de la última época (0-100).
        :param elapsed: Tiempo total de entrenamiento en segundos.
        """
        import datetime

        meta = {
            "arch": self.arch,
            "seed": self.seed,
            "weights_hash": self._weights_hash(),
            "epochs": epochs,
            "final_loss": round(final_loss, 6),
            "final_acc": round(final_acc, 4),
            "elapsed_s": round(elapsed, 2),
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with open(self._metadata_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_metadata(self) -> "dict | None":
        """
        Carga la metadata del modelo actual desde disco.

        :return: Dict con arch, seed, weights_hash, epochs, final_loss,
                 final_acc, elapsed_s, created_at; o None si no existe.
        """
        path = self._metadata_path()
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def list_saved_models(cls, cache_dir: "str | None" = None) -> "list[dict]":
        """
        Lista todos los modelos CNN preentrenados localmente.

        Busca pares (weights.pt + metadata.json) en cache_dir y
        devuelve la lista ordenada por final_acc descendente.

        :param cache_dir: Directorio de caché. None = default.
        :return: Lista de dicts de metadata, ordenada por precisión.
        """
        if cache_dir is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(root, "Data", "feature_cache")

        models = []
        if not os.path.isdir(cache_dir):
            return models

        for fname in os.listdir(cache_dir):
            if not fname.endswith("_metadata.json"):
                continue
            meta_path = os.path.join(cache_dir, fname)
            # Verificar que el .pt correspondiente existe
            weights_name = fname.replace("_metadata.json", "_weights.pt")
            weights_path = os.path.join(cache_dir, weights_name)
            if not os.path.exists(weights_path):
                continue
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                meta["weights_path"] = weights_path
                meta["metadata_path"] = meta_path
                models.append(meta)
            except Exception:
                continue

        models.sort(key=lambda m: m.get("final_acc", 0.0), reverse=True)
        return models

    def load_from_path(self, weights_path: str) -> None:
        """
        Carga pesos CNN desde una ruta explícita de archivo .pt.

        Permite cargar cualquier modelo guardado, no solo el que
        coincide con arch/seed actuales.

        :param weights_path: Ruta absoluta al archivo .pt.
        """
        import torch

        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

    def _load_weights_if_cached(self) -> bool:
        """
        Carga pesos CNN desde caché si el archivo existe.

        :return: True si los pesos fueron cargados desde caché, False si no existe.
        :rtype: bool.
        """
        path = self._weights_cache_path()
        if not os.path.exists(path):
            return False
        self._model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self._model.eval()
        return True

    # ── guardado / carga de features ─────────────────────────────

    def _load_features_if_cached(
        self, split: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Carga features extraídos desde caché si existen.

        :param split: Conjunto a cargar ("train" o "test").
        :type split: str.

        :return: Tupla (X_features, Y) si ambos archivos existen, None si no.
        :rtype: Optional[Tuple[np.ndarray, np.ndarray]].
        """
        px, py = self._feature_cache_paths(split)
        if os.path.exists(px) and os.path.exists(py):
            return np.load(px), np.load(py)
        return None

    def _save_features(self, split: str, X_feat: np.ndarray, Y: np.ndarray) -> None:
        """
        Guarda features extraídos en el caché.

        :param split: Identificador del conjunto ("train" o "test").
        :type split: str.

        :param X_feat: Features extraídos.
        :type X_feat: np.ndarray de shape (N, feature_dim) float32.

        :param Y: Etiquetas asociadas.
        :type Y: np.ndarray de shape (N,) int32.

        :return: None
        :rtype: NoneType.
        """
        px, py = self._feature_cache_paths(split)
        np.save(px, X_feat)
        np.save(py, Y)

    # ── preentrenamiento ──────────────────────────────────────────

    def pretrain(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 256,
        verbose: bool = True,
        on_epoch: "Callable[[int, int, float, float], None] | None" = None,
    ) -> None:
        """
        Preentrenamiento supervisado de la CNN simple en CIFAR-10.

        Solo aplica para arch="simple". Guarda los pesos en caché al
        terminar para que los arranques posteriores sean instantáneos.

        :param X_train: Imágenes de entrenamiento.
        :type X_train: np.ndarray de shape (N, 3, 32, 32) float32 normalizado.

        :param Y_train: Etiquetas de entrenamiento.
        :type Y_train: np.ndarray de shape (N,) int32.

        :param epochs: Número de épocas de preentrenamiento.
        :type epochs: int, default=10.

        :param lr: Tasa de aprendizaje del optimizador Adam.
        :type lr: float, default=1e-3.

        :param batch_size: Número de ejemplos por batch.
        :type batch_size: int, default=256.

        :param verbose: Si True, imprime el progreso del entrenamiento.
        :type verbose: bool, default=True.

        :return: None
        :rtype: NoneType.
        """
        if self.arch != "simple":
            return

        if verbose:
            print(f"[CNN] Preentrenando CNN simple ({epochs} épocas, lr={lr})...")

        # Activa el entrenamiento de la CNN
        for param in self._model.parameters():
            param.requires_grad_(True)
        self._model.train()

        # Crea un clasificador temporal
        classifier = nn.Linear(FEATURE_DIM, 10).to(self.device)

        # El algoritmo Adam es un optimizador variante de SGD con
        # momentum y learning rate adaptativo.
        optimizer = torch.optim.Adam(
            list(self._model.parameters()) + list(classifier.parameters()), lr=lr
        )
        criterion = nn.CrossEntropyLoss()
        N = len(X_train)

        # Crea un generador aleatorio
        rng = np.random.RandomState(self.seed if self.seed is not None else 0)
        _t_pretrain_start = time.perf_counter()
        _final_loss, _final_acc = 0.0, 0.0

        if on_epoch is not None:
            on_epoch(0, epochs, 0.0, 0.0)

        for epoch in range(1, epochs + 1):
            idx = rng.permutation(N)
            total_loss, correct = 0.0, 0

            # Divide el entrenamiento en mini-batches
            for start in range(0, N, batch_size):
                b = idx[start : start + batch_size]

                # Los datos NumPy se convierten a tensores PyTorch
                xb = torch.from_numpy(X_train[b]).to(self.device)
                yb = torch.from_numpy(Y_train[b].astype(np.int64)).to(self.device)

                # Esto limpia gradientes acumulados del batch anterior
                optimizer.zero_grad()

                # Se obtienen las features
                feats = self._model(xb)
                logits = classifier(feats)
                loss = criterion(logits, yb)
                loss.backward()

                # El optimizador actualiza los pesos usando los gradientes
                optimizer.step()

                # Se acumula la pérdida total y las predicciones correctas
                total_loss += loss.item() * len(b)
                correct += (logits.argmax(1) == yb).sum().item()

            epoch_loss = total_loss / N
            epoch_acc  = 100.0 * correct / N
            _final_loss, _final_acc = epoch_loss, epoch_acc
            if verbose:
                print(
                    f"  Época {epoch:2d}/{epochs}  "
                    f"loss={epoch_loss:.4f}  acc={epoch_acc:.1f}%"
                )
            if on_epoch is not None:
                on_epoch(epoch, epochs, epoch_loss, epoch_acc)

        # Congela la CNN para desactivar el aprendizaje
        for param in self._model.parameters():
            param.requires_grad_(False)

        # Cambia a modo evaluación
        self._model.eval()
        _elapsed = time.perf_counter() - _t_pretrain_start
        self._save_weights()
        self._save_metadata(epochs, _final_loss, _final_acc, _elapsed)

        if verbose:
            print(
                f"[CNN] Pesos guardados en caché ({self._weights_cache_path()}).\n"
                f"      Hash de pesos: {self._weights_hash()}\n"
                f"      Metadata guardada: acc={_final_acc:.1f}%  "
                f"loss={_final_loss:.4f}  tiempo={_elapsed:.1f}s\n"
            )

    # ── método principal: prepare() ───────────────────────────────

    def prepare(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        split: str = "train",
        pretrain_epochs: int = 10,
        pretrain_lr: float = 1e-3,
        batch_size: int = 2048,
        verbose: bool = True,
        on_epoch: "Callable[[int, int, float, float], None] | None" = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara los features con máxima reutilización de caché.

        Orden de decisión:

            1. ¿Caché de features (con hash de pesos actuales)?
               → carga instantánea.
            2. ¿Caché de pesos? → carga pesos (el hash cambia).
               → volver a buscar caché de features con nuevo hash.
            3. ¿No hay nada? → pretrain (arch=simple) o usar pesos tal cual.
            4. Extraer features y guardar caché.

        El PS llama con pretrain_epochs=0 para nunca reentrenar.

        :param X: Imágenes de entrada.
        :type X: np.ndarray de shape (N, 3, 32, 32) float32.

        :param Y: Etiquetas de entrada.
        :type Y: np.ndarray de shape (N,) int32.

        :param split: Identificador del conjunto.
        :type split: str, "train" o "test", default="train".

        :param pretrain_epochs: Épocas de pretrain (0 = nunca reentrenar).
        :type pretrain_epochs: int, default=10.

        :param pretrain_lr: Tasa de aprendizaje para el pretrain.
        :type pretrain_lr: float, default=1e-3.

        :param batch_size: Tamaño del batch para extracción de features.
        :type batch_size: int, default=2048.

        :param verbose: Si True, imprime información de progreso.
        :type verbose: bool, default=True.

        :return: Tupla (X_features, Y) con features extraídos.
        :rtype: Tuple[np.ndarray, np.ndarray] donde X_features shape (N, feature_dim).
        """
        # ── Paso 1: ¿features ya en caché con los pesos actuales? ─
        cached = self._load_features_if_cached(split)
        if cached is not None:
            X_feat, Y_cached = cached
            if verbose:
                print(
                    f"[CNN] Features '{split}' en caché "
                    f"(hash={self._weights_hash()}): {X_feat.shape}"
                )
            return X_feat, Y_cached

        # ── Paso 2: ¿pesos en caché? → cargarlos y volver a buscar ─
        if self._load_weights_if_cached():
            if verbose:
                print(
                    f"[CNN] Pesos cargados desde caché "
                    f"(hash={self._weights_hash()}, no se reentrenará)."
                )
            # Con los pesos cargados el hash cambia → comprobar features
            cached = self._load_features_if_cached(split)
            if cached is not None:
                X_feat, Y_cached = cached
                if verbose:
                    print(f"[CNN] Features '{split}' en caché tras cargar pesos.")
                return X_feat, Y_cached

        # ── Paso 3: ni features ni pesos → pretrain si corresponde ─
        elif self.arch == "simple" and pretrain_epochs > 0:
            self.pretrain(
                X,
                Y,
                epochs=pretrain_epochs,
                lr=pretrain_lr,
                batch_size=batch_size,
                verbose=verbose,
                on_epoch=on_epoch,
            )
        # resnet18 sin pretrained o pretrain_epochs=0: usar pesos actuales

        # ── Paso 4: extraer features con los pesos definitivos ─────
        if verbose:
            print(
                f"[CNN] Extrayendo features '{split}' "
                f"({len(X)} imgs, hash={self._weights_hash()})..."
            )
        t0 = time.perf_counter()
        X_feat = self.extract_batched(X, batch_size=batch_size, verbose=verbose)
        elapsed = time.perf_counter() - t0
        self._save_features(split, X_feat, Y)

        if verbose:
            print(
                f"[CNN] Features guardados en caché: {X_feat.shape}  ({elapsed:.1f}s)\n"
            )
        return X_feat, Y

    # ── extracción directa ────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    def extract(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza forward pass sin gradientes sobre un batch de imágenes.

        :param X: Imágenes de entrada.
        :type X: np.ndarray de shape (batch_size, 3, 32, 32) float32.

        :return: Features extraídos del batch.
        :rtype: np.ndarray de shape (batch_size, feature_dim) float32.
        """
        with torch.inference_mode():
            t = torch.from_numpy(X).to(self.device)
            return self._model(t).cpu().numpy()

    def extract_batched(
        self,
        X: np.ndarray,
        batch_size: int = 2048,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Extrae features en mini-batches para controlar el uso de RAM.

        Procesa el array X dividiéndolo en batches y extrayendo features
        de cada uno. Útil para datasets grandes que no caben en GPU.

        :param X: Imágenes de entrada.
        :type X: np.ndarray de shape (N, 3, H, W) float32.

        :param batch_size: Número de imágenes a procesar por batch.
        :type batch_size: int, default=2048.

        :param verbose: Si True, imprime una barra de progreso por consola.
        :type verbose: bool, default=False.

        :return: Features de todas las imágenes concatenadas.
        :rtype: np.ndarray de shape (N, feature_dim) float32.
        """
        N = len(X)
        starts = list(range(0, N, batch_size))
        n_batch = len(starts)
        parts = []

        for idx, i in enumerate(starts, 1):
            parts.append(self.extract(X[i : i + batch_size]))

            if verbose:
                done = int(20 * idx / n_batch)
                bar = "█" * done + "░" * (20 - done)
                n_done = min(i + batch_size, N)
                print(
                    f"\r  [CNN] Extrayendo features [{bar}] "
                    f"{n_done}/{N} imgs  ({idx}/{n_batch} batches)",
                    end="",
                    flush=True,
                )

        if verbose:
            print()  # salto de línea al terminar

        return np.concatenate(parts, axis=0)