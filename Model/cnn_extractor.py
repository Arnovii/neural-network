"""
Model/cnn_extractor.py

Extractor CNN para ImageNet con PyTorch.

ARQUITECTURAS:
  "resnet18" → ResNet-18 con pesos ImageNet (siempre preentrenada).
               Carga IMAGENET1K_V1 weights automáticamente.
               feature_dim = 512. Sin capa de clasificación.
               Se construye CONGELADA (requires_grad=False, eval).
               Semántica: extractor de características fijo.

  "simple"   → ResNet-18 canónico desde cero (sin pesos preentrenados).
               Arquitectura estándar: Stem + 4 layers × 2 bloques + GAP.
               feature_dim = 512. Totalmente entrenable desde el inicio (requires_grad=True).
               Se construye ENTRENABLE (requires_grad=True, eval).
               Semántica: red entrenable end-to-end en modo E2E.

ARQUITECTURA "simple" (SIMPLE CNN):
  - 11.2M parámetros (vs 11.7M anterior con proj+Dropout redundantes)
  - Stride efectivo 32× (igual que resnet18 preentrenada)
  - Skip connections fuertes → gradientes fluyen correctamente en Async-SGD con staleness
  - BasicBlock(3x3) estándar: Conv→BN→ReLU→Conv→BN + shortcut
  - Inicialización Kaiming Normal para todos los Conv/Linear
  - GAP directo → 512-dim: sin Dropout ni proyección extra post-GAP

POR QUÉ SIN DROPOUT POST-GAP:
  El Dropout(0.5) post-GAP interferiere con BatchNorm en Async-SGD:
    - BN asume distribución estable de activaciones: Dropout la altera estocásticamente
    - Con staleness variable, los gradientes ya llegan atenuados por α(s)=1/(1+λ·s)
    - Añadir ruido de Dropout encima introduce inestabilidad numérica innecesaria
  La regularización en E2E viene del weight_decay=1e-4 en SGD (WorkerNode)
  y del label_smoothing=0.1 en CrossEntropyLoss, ambos más estables que Dropout.

POR QUÉ SIN PROYECCIÓN 512→512:
  La proyección Linear(512→512) es semánticamente una identidad aprendida (~262K params).
  El GAP ya produce exactamente FEATURE_DIM tras Layer4.
  Eliminarla equipara la arquitectura al ResNet-18 canónico de torchvision
  y reduce el riesgo de colapso de representaciones en las primeras épocas E2E.

MEJORAS PARA IMAGENET-1K DISTRIBUIDO:
  - ResNet-18 canónico es arquitectura validada, optimizada en PyTorch
  - Suficiente capacidad para ImageNet (11.2M params)
  - Skip connections robustos para manejar staleness en Async-SGD
  - Sin dependencia de pesos preentrenados → completamente "from scratch"

CLAVE:
  El requires_grad se determina por arquitectura en __init__:
  - "resnet18" → requires_grad=False (congelada, pesos ImageNet)
  - "simple" → requires_grad=True (entrenable desde cero)
"""

from __future__ import annotations

import io
import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.constants import FEATURE_DIM


# ================================================================
# BLOQUE RESIDUAL BÁSICO (ResNet-18)
# ================================================================


class _BasicBlock(nn.Module):
    """
    Bloque residual básico estándar de ResNet: Conv→BN→ReLU→Conv→BN + shortcut.

    Estructura:
      - Conv(in_ch, out_ch, 3x3, stride) con BN
      - ReLU
      - Conv(out_ch, out_ch, 3x3) con BN (sin ReLU, se aplica após shortcut)
      - Shortcut: identidad o Conv 1x1 si stride > 1 o cambio de canales
      - ReLU(residual + shortcut)

    Zero-init del BN final estabiliza el inicio (bloque ≈ identidad).
    """

    expansion = 1  # out_ch = in_ch * expansion (diferente en Bottleneck)

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        """
        Inicializa un bloque residual básico de ResNet-18.

        Estructura: Conv(3×3) → BN → ReLU → Conv(3×3) → BN + shortcut.
        El shortcut es identidad si stride=1 y in_ch=out_ch, sino Conv(1×1).
        Zero-init en BN final hace que el bloque inicie como identidad.

        :param in_ch:
            Número de canales de entrada.
        :type in_ch:
            int

        :param out_ch:
            Número de canales de salida (número de filtros).
        :type out_ch:
            int

        :param stride:
            Stride de la primera convolución. Default: 1 (sin submuestreo espacial).
            Usar stride=2 para reducir resolución espacial a la mitad.
        :type stride:
            int, optional

        :returns:
            Nada.
        :rtype:
            None

        :examples:
            Crear un bloque basic que mantiene resolución (stride=1):

            .. code-block:: python

                block = _BasicBlock(64, 64, stride=1)
                x = torch.randn(8, 64, 56, 56)
                y = block(x)  # output: (8, 64, 56, 56)

            Crear un bloque que reduce resolución a la mitad (stride=2):

            .. code-block:: python

                block = _BasicBlock(64, 128, stride=2)
                x = torch.randn(8, 64, 56, 56)
                y = block(x)  # output: (8, 128, 28, 28)

        :note:
            El costo computacional de este bloque es aproximadamente 2× Conv(3×3),
            más shortcut (identidad o Conv 1×1 si cambian canales/stride).
            Para una red ResNet-18 con 8 bloques, el tiempo total forward es ~O(8).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Shortcut: proyección 1×1 si stride > 1 o cambio de canales
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
        """
        Pase forward a través del bloque residual.

        Aplica: Conv→BN→ReLU→Conv→BN + shortcut→ReLU.

        :param x:
            Tensor de entrada de forma (B, in_ch, H, W) donde:
            - B es tamaño de batch
            - in_ch es número de canales de entrada (debe coincidir con __init__)
            - H, W son altura y ancho espaciales
        :type x:
            torch.Tensor

        :returns:
            Tensor de salida de forma (B, out_ch, H', W') donde:
            - H' = H si stride=1, H' = H//2 si stride=2
            - W' = W si stride=1, W' = W//2 si stride=2
            - out_ch es número de canales de salida (del __init__)
        :rtype:
            torch.Tensor

        :example:
            .. code-block:: python

                block = _BasicBlock(64, 64, stride=1)
                x = torch.randn(4, 64, 56, 56)
                y = block(x)
                assert y.shape == (4, 64, 56, 56)
        """
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


# ================================================================
# CNN SIMPLE: RESNET-18 CANÓNICO FROM SCRATCH
# ================================================================


class _ResNet18FromScratch(nn.Module):
    """
    ResNet-18 canónico sin pesos preentrenados, entrenable desde el inicio.

    Arquitectura idéntica al ResNet-18 de torchvision, sin la capa fc final
    (reemplazada por GAP directo → 512-dim). Sin Dropout ni proyección extra.

    Arquitectura:
      Stem:   Conv(3→64, 7×7, stride=2) + BN + ReLU + MaxPool(3×3, stride=2) → stride 4×
      Layer1: BasicBlock(64→64,   stride=1) × 2  → 56×56
      Layer2: BasicBlock(64→128,  stride=2) × 2  → 28×28
      Layer3: BasicBlock(128→256, stride=2) × 2  → 14×14
      Layer4: BasicBlock(256→512, stride=2) × 2  → 7×7
      GAP:    AdaptiveAvgPool2d(1×1) → (B, 512)  [FEATURE_DIM, directo]

    Con imagen 224×224:
      Stem   →   56×56  (Conv 7×7 s=2 + MaxPool 3×3 s=2)
      Layer1 →   56×56  (stride=1)
      Layer2 →   28×28  (stride=2)
      Layer3 →   14×14  (stride=2)
      Layer4 →    7×7   (stride=2)
      GAP    →    1×1   → flatten → (B, 512)

    Stride efectivo total: 32× (igual que ResNet-18 preentrenada).

    Parámetros: ~11.2M (vs 11.7M anterior; diferencia = proj 512→512 eliminada)
      - Stem:   ~9.4K
      - Layer1: ~148K
      - Layer2: ~526K
      - Layer3: ~2.1M
      - Layer4: ~8.4M
      Total: ~11.2M

    Por qué sin Dropout post-GAP:
      BatchNorm en cada bloque ya regulariza las activaciones intermedias.
      En Async-SGD con staleness, los gradientes llegan atenuados por α(s).
      Añadir Dropout post-GAP introduce ruido adicional que interfiere con BN
      y desestabiliza el averaging del PS. La regularización viene de
      weight_decay=1e-4 en SGD y label_smoothing=0.1 en CrossEntropyLoss.

    Por qué sin proyección 512→512:
      El GAP ya produce 512-dim (FEATURE_DIM). Una Linear(512→512) aprende
      una transformación que tiende a la identidad, con ~262K parámetros extra
      que ralentizan la convergencia en las primeras épocas sin beneficio claro.
      El ResNet-18 canónico de torchvision (con fc eliminada) produce 512-dim
      directamente desde GAP: ésta es la arquitectura de referencia.
    """

    def __init__(self) -> None:
        """
        Construye ResNet-18 canónico completamente inicializado desde cero.

        Sin parámetros: la arquitectura es fija y determinista.

        Inicialización:
            - Convolutions: Kaiming Normal con fan_out y nonlinearity='relu'
            - BatchNorm: ones para weights, zeros para bias (excepto bloques con zero-init)
            - Cada _BasicBlock: BN final con weight zero-init para estabilidad en E2E

        Parámetros totales: ~11,176,512

        :returns:
            Nada. La red se inicializa en el constructor.
        :rtype:
            None

        :raises:
            No hay validación. Los tensores se inicializan OK.

        :examples:
            Crear modelo SIMPLE CNN:

            .. code-block:: python

                model = _ResNet18FromScratch()
                x = torch.randn(4, 3, 224, 224)
                y = model(x)
                assert y.shape == (4, 512)

        :note:
            Completamente entrenable: requires_grad=True para todos los parámetros.
            Ideal para Async-SGD distribuido en ImageNet-1K.
            Stride efectivo total: 32× (igual que ResNet-18 preentrenada).
        """
        super().__init__()

        # Stem: reducción inicial 224→56 (idéntico al ResNet-18 de torchvision)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 4 layers, cada una con 2 bloques (arquitectura ResNet-18 canónica)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # GAP directo → FEATURE_DIM (sin Dropout ni proyección extra)
        # Equivalente a torchvision.models.resnet18 con fc reemplazado por Identity()
        self.gap = nn.AdaptiveAvgPool2d(1)

        self._init_weights()

    def _make_layer(
        self, in_ch: int, out_ch: int, num_blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """
        Construye un "layer" de ResNet-18 con bloques residuales.

        Estructura: primer bloque con stride especificado, resto con stride=1.

        :param in_ch:
            Número de canales de entrada para el primer bloque.
        :type in_ch:
            int

        :param out_ch:
            Número de canales de salida (uniforme para toda la layer).
        :type out_ch:
            int

        :param num_blocks:
            Cantidad de bloques _BasicBlock a instanciar en esta layer.
            Típicamente 2 para ResNet-18 (3 para ResNet-50, etc).
        :type num_blocks:
            int

        :param stride:
            Stride para el PRIMER bloque de la layer. Default: 1.
            Use stride=2 para submuestreo (reduce H,W a la mitad).
        :type stride:
            int, optional

        :returns:
            nn.Sequential con num_blocks BasicBlocks conectados secuencialmente.
        :rtype:
            nn.Sequential

        :example:
            Crear layer con 2 bloques, submuestreo:

            .. code-block:: python

                layer = self._make_layer(64, 128, num_blocks=2, stride=2)
                x = torch.randn(4, 64, 56, 56)
                y = layer(x)
                assert y.shape == (4, 128, 28, 28)

        :note:
            En ResNet-18:
            - Layer1: _make_layer(64, 64, 2, stride=1) → remain 56×56
            - Layer2: _make_layer(64, 128, 2, stride=2) → reduce a 28×28
            - Layer3: _make_layer(128, 256, 2, stride=2) → reduce a 14×14
            - Layer4: _make_layer(256, 512, 2, stride=2) → reduce a 7×7
        """
        layers = [_BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(_BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """
        Inicializa todos los pesos con Kaiming Normal (He initialization).

        Strategia:
            - Conv2d: Kaiming Normal, fan_out, nonlinearity='relu'
            - BatchNorm2d: weight=ones, bias=zeros
              (excepto bloques con zero-init en BN final)
            - Linear: Kaiming Normal, fan_out, nonlinearity='relu'
              (no hay Linear en esta arquitectura; incluido por extensibilidad)

        Kaiming Normal es óptimo para redes profundas con ReLU.
        Mantiene varianza de activaciones constante a través de capas.

        :returns:
            Nada. Modifica weights en-place.
        :rtype:
            None

        :note:
            Llamado automáticamente en __init__.
            No es necesario llamar manualmente.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pase forward a través de ResNet-18 canónico para extracción de características.

        Arquitectura del pipeline:
            1. Stem: Conv(7×7, s=2) + BN + ReLU + MaxPool(3×3, s=2) → stride 4×
            2. Layer1-4: 8 bloques BasicBlock con stride submuestreo en layer2-4
            3. GAP: AdaptiveAvgPool2d reduce spatial dims a 1×1 → flatten a (B, 512)

        Transformación espacial:
            - 224×224 → Stem → 56×56 (stride 4×)
            - 56×56 → Layer1 → 56×56 (stride 1)
            - 56×56 → Layer2 → 28×28 (stride 2)
            - 28×28 → Layer3 → 14×14 (stride 2)
            - 14×14 → Layer4 → 7×7 (stride 2)
            - 7×7 → GAP → 1×1 → flatten → (B, 512)

        Total stride: 32× (224→7 = 32×).

        :param x:
            Batch de imágenes de entrada. Forma (B, 3, H, W) donde:
            - B: tamaño de batch
            - 3: canales RGB
            - H, W: altura y ancho. Optimizado para H=W=224 (ImageNet)
              pero acepta cualquier tamaño (reducido por stride 32×)
        :type x:
            torch.Tensor

        :returns:
            Características extraídas. Forma (B, 512) donde:
            - B: tamaño de batch (mismo que entrada)
            - 512: dimensión de características (FEATURE_DIM)
        :rtype:
            torch.Tensor

        :example:
            Forward pass típico:

            .. code-block:: python

                model = _ResNet18FromScratch()
                batch = torch.randn(8, 3, 224, 224)  # ImageNet batch
                features = model(batch)
                assert features.shape == (8, 512)

        :note:
            - BatchNorm: usa running stats en eval(), actualiza en train()
            - Requiere model.train() o model.eval() antes de forward()
            - Sin Dropout: determinista en eval() y train() (solo BN varía)

        :raises:
            RuntimeError si X tiene número de canales ≠ 3 o tipo incorrecto.
        """
        x = self.stem(x)  # (B,3,224,224) → (B,64,56,56)
        x = self.layer1(x)  # (B,64,56,56)  → (B,64,56,56)
        x = self.layer2(x)  # (B,64,56,56)  → (B,128,28,28)
        x = self.layer3(x)  # (B,128,28,28) → (B,256,14,14)
        x = self.layer4(x)  # (B,256,14,14) → (B,512,7,7)
        x = self.gap(x)  # (B,512,7,7)   → (B,512,1,1)
        x = x.flatten(1)  # (B,512,1,1)   → (B,512)
        return x  # FEATURE_DIM, directo sin proyección


# ================================================================
# EXTRACTOR PÚBLICO
# ================================================================


class CNNExtractor:
    """
    Envuelve una CNN PyTorch y expone interfaz para el sistema distribuido Async-SGD.

    Esta clase encapsula TWO ARQUITECTURAS distintas con COMPORTAMIENTO FIJO:

    **ResNet-18 con pesos preentrenados (arch='resnet18')**:
        - Carga IMAGENET1K_V1 weights automáticamente
        - requires_grad=False: CNN CONGELADA, NO recibe gradientes
        - Funciona como extractor de características fijo
        - Ideal para: MLP-only training donde CNN es pretrained feature extractor

    **SIMPLE CNN (arch='simple')**:
        - Sin pesos preentrenados: inicialización desde cero
        - requires_grad=True: CNN ENTRENABLE, participa en backprop
        - Modo E2E: CNN + MLP se entrenan juntas
        - Arquitectura canónica ResNet-18 (sin Dropout/proyección extra)
        - Ideal para: Async-SGD distribuido sin dependencia de pesos preentrenados

    **Comportamiento compartido**:
        - Ambas producen 512-dim feature vectors (FEATURE_DIM)
        - Se construyen en eval() para inferencia determinista de BatchNorm
        - Soportan serialización/deserialización para comunicación TCP distribuida
        - requires_grad NO cambia después de seleccionar arquitectura

    :param arch:
        Arquitectura a seleccionar. Opciones: 'resnet18' (preentrenada),
        'simple' (SIMPLE CNN o ResNet-18 from scratch). Default: 'resnet18'.
    :type arch:
        str

    :param device:
        Dispositivo PyTorch donde colocar la CNN. Opciones: 'cpu', 'cuda',
        'cuda:0', 'mps'. Default: 'cpu'. Se auto-detecta si se usa CUDA/MPS.
    :type device:
        str

    :param seed:
        Semilla RNG para reproducibilidad. Solo afecta a arch='simple'.
        Default: None (sin semilla, aleatorio).
    :type seed:
        int | None

    :raises ValueError:
        Si arch no está en ('simple', 'resnet18').
    :raises RuntimeError:
        Si device no es válido o no está disponible.

    :example:
        Crear extractor ResNet-18 preentrenado (MLP-only):

        .. code-block:: python

            extractor = CNNExtractor(arch='resnet18', device='cuda')
            x = torch.randn(8, 3, 224, 224)
            features = extractor._model(x)  # (8, 512)

        Crear extractor SIMPLE CNN (E2E):

        .. code-block:: python

            extractor = CNNExtractor(arch='simple', device='cuda', seed=42)
            x = torch.randn(8, 3, 224, 224)
            features = extractor._model(x)  # (8, 512), entrenable

    :note:
        - Para distribuido: serializar con _get_weights_bytes(), transportar TCP,
          deserializar con load_weights_from_bytes().
        - requires_grad se establece EN __init__ y NO cambia después.
        - En WorkerNode._train_batch: modelo se pone train() antes forward.
    """

    ARCHITECTURES = ("simple", "resnet18")

    def __init__(
        self,
        arch: str = "resnet18",
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        """Inicializa extractor CNN con arquitectura y dispositivo especificados.

        Flujo:
            1. Valida que arch esté en ('simple', 'resnet18')
            2. Detecta y crea dispositivo PyTorch
            3. Fija semilla RNG si seed != None
            4. Construye arquitectura con _build(arch, trainable)
            5. Establece requires_grad según arquitectura
            6. Inicializa en eval() para inferencia determinista

        Estado post-__init__:
            - Modelo PyTorch completo en self._model
            - requires_grad establecido permanentemente (NO cambia después)
            - Listo para forward pass o serialización

        Args:
            arch: Arquitectura. Opciones válidas: 'resnet18' (preentrenada, congelada),
                  'simple' (SIMPLE CNN, entrenable). Default: 'resnet18'.
            device: Dispositivo PyTorch. Opciones: 'cpu', 'cuda' (auto GPU0),
                   'cuda:N' (GPU N), 'mps' (Apple Metal). Default: 'cpu'.
            seed: Semilla RNG para reproducibilidad. Si None, no fija nada (aleatorio).
                  Solo afecta a arch='simple' (SIMPLE CNN).
                  No afecta a arch='resnet18' (pesos prefijos de ImageNet).

        Returns:
            None. Inicializa self._model, self.arch, self.device, self.seed.

        Raises:
            ValueError: Si arch no está en ARCHITECTURES = ('simple', 'resnet18').
                        Mensaje: "arch debe ser ('simple', 'resnet18'), recibido: {arch!r}"
            RuntimeError: Si device no es válido (ej: 'cuda' pero GPU no disponible).
                        PyTorch levanta RuntimeError automáticamente.

        Example:
            # ResNet-18 preentrenado (congelado)
            cnn = CNNExtractor(arch='resnet18', device='cuda')
            # Reentrenable desde cero
            cnn = CNNExtractor(arch='simple', device='cpu', seed=42)

        Note:
            - requires_grad se establece UNA SOLA VEZ en __init__ y NO cambia
            - model.train() / model.eval() controla BatchNorm pero NO affecta requires_grad
            - Para cambiar requires_grad después: usa p.requires_grad_(False) manualmente
            - Las correcciones distribuidas NO modifican requires_grad
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
        Factory method que construye la arquitectura CNN especificada.

        Lógica:
            - arch='simple': construye _ResNet18FromScratch() (ResNet-18 canónico sin pesos)
            - arch='resnet18': descarga ResNet-18 con pesos IMAGENET1K_V1 usando torchvision

        Esta es una función helper interna llamada solo desde __init__.
        NO debe ser llamada directamente (use constructor CNNExtractor).

        :param arch:
            Cadena de arquitectura. Valores válidos: 'simple', 'resnet18'.
        :type arch:
            str

        :param trainable:
            Flag que controla si cargar pesos ImageNet (False) o inicializar
            desde cero (True). Solo usado para arch='resnet18' (ignora para 'simple').
        :type trainable:
            bool

        :returns:
            Módulo PyTorch construido (nn.Module). Sin wrapping en CNNExtractor.
            Listo para to(device) y requires_grad_(bool).
        :rtype:
            nn.Module

        :raises ImportError:
            Si arch='resnet18' y torchvision no está instalado.
        :raises Exception:
            Si descarga de pesos ImageNet falla.

        :example:
            Construcción interna (usado por __init__):

            .. code-block:: python

                # No llamar directamente - use CNNExtractor(arch='simple')
                model = CNNExtractor._build('simple', trainable=True)
                assert isinstance(model, _ResNet18FromScratch)

        :note:
            - Private method: prefijo `_` indica no usar directamente
            - trainable=True para 'simple' pero retorna lo mismo
            - arch='resnet18' con trainable=True no carga pesos (None)
        """
        if arch == "simple":
            return _ResNet18FromScratch()

        import torchvision.models as tvm

        weights = "IMAGENET1K_V1" if not trainable else None
        model = tvm.resnet18(weights=weights)
        model.fc = nn.Identity()  # type: ignore[assignment]
        return model

    @property
    def feature_dim(self) -> int:
        """
        Propiedad read-only: dimensión de características extraídas.

        La CNN produce vectores de features de 512 dimensiones para ambas
        arquitecturas (resnet18 preentrenada y SIMPLE CNN).

        El MLP clasificador usa este valor para dimensionar su capa de entrada:
        MLP(feature_dim, hidden1, hidden2, n_classes=1000).

        :returns:
            Dimensión del espacio de características (siempre 512).
        :rtype:
            int

        :example:
            .. code-block:: python

                cnn = CNNExtractor(arch='simple', device='cpu')
                assert cnn.feature_dim == 512

                # Usar para construir MLP
                from Model.mlp_pytorch import MLPPyTorch
                mlp = MLPPyTorch(feature_dim=cnn.feature_dim, hidden1=2048, hidden2=1024, n_classes=1000)

        :note:
            Constante global FEATURE_DIM (no configurable per-instancia).
        """
        return FEATURE_DIM

    # ── Serialización ─────────────────────────────────────────────

    def _get_weights_bytes(self) -> bytes:
        """
        Serializa estado de la CNN a bytes para transmisión TCP distribuida.

        Flujo:
            1. Extrae state_dict() del módulo PyTorch (Dict[str, Tensor])
            2. Serializa con torch.save() a BytesIO usando HIGHEST_PROTOCOL
            3. Retorna bytes completos (incluyendo metadata de PyTorch)

        Tamaño aproximado:
            - ResNet-18 preentrenada (11.2M params): ~44.9 MB
            - Similar para ambas arquitecturas (mismo tamaño state_dict)

        Use case:
            - Parameter Server envía pesos a Workers vía TCP
            - Workers reciben bytes y deserializan con load_weights_from_bytes()

        :returns:
            Representación binaria completa del state_dict. Contiene
            todos los parámetros y buffers (ej: BatchNorm running_mean).
        :rtype:
            bytes

        :example:
            Serialización completa para transmisión:

            .. code-block:: python

                cnn = CNNExtractor(arch='simple', device='cpu')
                weights_bytes = cnn._get_weights_bytes()
                print(f"Serialized size: {len(weights_bytes) / (1024**2):.1f} MB")
                # Output: Serialized size: ~44.9 MB

                # Enviar por TCP...
                # En Worker: cnn_worker.load_weights_from_bytes(weights_bytes)

        :note:
            - Usa torch.save() internamente (protocol=HIGHEST_PROTOCOL)
            - Contiene metadata de tipos, shapes, versiones PyTorch
            - requires_grad NO se serializa (se preserva en deserialización)
        """
        buf = io.BytesIO()
        torch.save(self._model.state_dict(), buf)
        return buf.getvalue()

    def load_weights_from_bytes(self, weights_bytes: bytes) -> None:
        """
        Deserializa y carga pesos desde bytes recibidos del Parameter Server.

        Flujo:
            1. Crea BytesIO desde bytes recibidos
            2. Deserializa con torch.load() usando map_location=device
            3. Carga en state_dict() con load_state_dict()
            4. Fuerza model.eval() para BatchNorm determinista

        Garantías:
            - requires_grad NO se modifica (se preserva desde __init__)
            - BatchNorm buffers (running_mean, etc) se actualizan
            - En train mode: las corridas stats se actualizarán en forward

        Sincronización distribuida:
            - Llamado en WorkerNode._sync_cnn(cnn_state)
            - Cnn_state viene del Parameter Server via msg['payload']['cnn_weights']
            - Mantiene consistencia entre todos los Workers

        :param weights_bytes:
            Bytes serializados con torch.save() de state_dict().
            Formato: binario (no texto), producido por _get_weights_bytes().
        :type weights_bytes:
            bytes

        :returns:
            Nada. Modifica self._model in-place.
        :rtype:
            None

        :raises RuntimeError:
            Si bytes no son válidos (corrupted, wrong format).
        :raises ValueError:
            Si state_dict keys no coinciden (arquitectura mismatch).

        :example:
            Recibir pesos del Parameter Server y cargar:

            .. code-block:: python

                # En Worker, dentro de _handle_worker_handshake()
                cnn = CNNExtractor(arch='simple', device='cuda')

                # Recibir bytes del PS vía TCP
                weights_bytes = receive_message(socket)['payload']['cnn_weights']

                # Cargar y sincronizar
                cnn.load_weights_from_bytes(weights_bytes)

                # Verificar estado
                assert cnn._model.training == False
                p = list(cnn._model.parameters())[0]
                assert p.requires_grad == True  # from arch='simple'

        :note:
            - CRÍTICO: requires_grad se preserva (no se resetea)
            - BatchNorm stats se ACTUALIZAN desde el PS (importante para E2E)
            - Llamado frecuentemente (~1 vez/batch en Async-SGD)
            - Overhead: ~40ms para ResNet-18 en CPU
        """
        buf = io.BytesIO(weights_bytes)
        state = torch.load(buf, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
