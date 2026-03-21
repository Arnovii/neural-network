"""
Utils/feature_scaler.py

StandardScaler de features CNN para mejorar la convergencia del MLP.

──────────────────────────────────────────────────────────────────
¿POR QUÉ NORMALIZAR FEATURES?
──────────────────────────────────────────────────────────────────
ResNet-18 produce vectores de 512 dimensiones. Con 1000 clases,
distintas dimensiones pueden tener escalas muy diferentes:
  - Dimensión 42: valores en [0.0, 0.8]
  - Dimensión 317: valores en [0.0, 15.3]

Sin normalizar, el MLP es muy sensible al learning rate y necesita
más épocas para converger. El StandardScaler lleva cada dimensión
a media≈0 y std≈1, haciendo el gradiente más uniforme.

──────────────────────────────────────────────────────────────────
ESTRATEGIA: CALCULADO OFFLINE CON SUBSET
──────────────────────────────────────────────────────────────────
No calculamos las estadísticas sobre 1.28M imágenes completas.
Usamos un subset de 50k features de train (un shard):
  - La distribución de features es estable entre muestras
  - Un subset de 50k da estadísticas con error < 0.5%
  - Mucho más rápido que procesar 1.28M

Las estadísticas (mean, std) se guardan en caché junto con los
shards de features y se aplican consistentemente en train y test.

──────────────────────────────────────────────────────────────────
USO
──────────────────────────────────────────────────────────────────
  # Worker: calcular y guardar
  scaler = FeatureScaler()
  scaler.fit(X_features_shard0)          # solo sobre train
  scaler.save(cache_dir, weights_hash)

  # Worker: cargar y aplicar
  scaler = FeatureScaler.load(cache_dir, weights_hash)
  X_norm = scaler.transform(X_features)  # train y test
"""

import os
from typing import Optional

import numpy as np


class FeatureScaler:
    """
    StandardScaler para features CNN.

    Calcula media y std sobre un subset de features de train
    y los aplica consistentemente a train y test.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self._fitted = False

    # ── Ajuste ────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """
        Calcula media y std de las features.

        :param X: Features de entrenamiento, shape (N, feature_dim).
        :return: self (encadenamiento).
        """
        self.mean_ = X.mean(axis=0, keepdims=True).astype(np.float32)
        self.std_ = X.std(axis=0, keepdims=True).astype(np.float32)
        # Evitar división por cero en dimensiones constantes
        assert self.std_ is not None
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_).astype(np.float32)
        self._fitted = True
        return self

    # ── Transformación ────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Normaliza features: (X - mean) / std.

        :param X: Features a normalizar, shape (N, feature_dim).
        :return: Features normalizados, mismo shape, float32.
        :raises RuntimeError: Si no se ha llamado fit() antes.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureScaler no ajustado. Llama fit() o load() primero."
            )
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Ajusta y transforma en un paso."""
        return self.fit(X).transform(X)

    # ── Persistencia ──────────────────────────────────────────────

    def save(self, cache_dir: str, weights_hash: str) -> None:
        """
        Guarda mean y std en disco como archivos .npy.

        La clave incluye el hash de los pesos CNN para que distintos
        modelos tengan escaladores distintos.

        :param cache_dir: Directorio de caché de features.
        :param weights_hash: Hash MD5 (8 hex) de los pesos CNN.
        """
        if not self._fitted:
            raise RuntimeError("No hay estadísticas que guardar.")
        os.makedirs(cache_dir, exist_ok=True)
        assert self.mean_ is not None and self.std_ is not None
        np.save(self._mean_path(cache_dir, weights_hash), self.mean_)
        np.save(self._std_path(cache_dir, weights_hash), self.std_)

    @classmethod
    def load(cls, cache_dir: str, weights_hash: str) -> Optional["FeatureScaler"]:
        """
        Carga un FeatureScaler desde disco.

        :return: FeatureScaler listo para transformar, o None si no existe.
        """
        mp = cls._mean_path(cache_dir, weights_hash)
        sp = cls._std_path(cache_dir, weights_hash)
        if not (os.path.exists(mp) and os.path.exists(sp)):
            return None
        scaler = cls()
        scaler.mean_ = np.load(mp)
        scaler.std_ = np.load(sp)
        scaler._fitted = True
        return scaler

    @classmethod
    def exists(cls, cache_dir: str, weights_hash: str) -> bool:
        """Comprueba si el escalador está en caché."""
        return os.path.exists(
            cls._mean_path(cache_dir, weights_hash)
        ) and os.path.exists(cls._std_path(cache_dir, weights_hash))

    # ── Rutas ─────────────────────────────────────────────────────

    @staticmethod
    def _mean_path(cache_dir: str, weights_hash: str) -> str:
        return os.path.join(cache_dir, f"scaler_{weights_hash}_mean.npy")

    @staticmethod
    def _std_path(cache_dir: str, weights_hash: str) -> str:
        return os.path.join(cache_dir, f"scaler_{weights_hash}_std.npy")
