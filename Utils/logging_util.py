"""
Utils/logging_util.py

Módulo de logging unificado para consistencia en mensajes de progreso
y fases de ejecución en toda la aplicación.

Proporciona funciones para registrar eventos con formato consistente:
    [FASE] Descripción ... (progreso/contexto) métrica

Fases:
    - LOAD: carga de datos
    - PREP: preprocesamiento o extracción de características
    - TRAIN: entrenamiento de modelos
    - EVAL: evaluación de modelos
    - INFO: información general
"""

import sys
from datetime import datetime
from typing import Optional


class FormattedLogger:
    """
    Logger con formato unificado para fases de ejecución.

    Ejemplo de salida:
        [LOAD DATA]   Cargando CIFAR-10... (10 000 imágenes)
        [PREP FEAT]   Extrayendo características... (32.5%)
        [TRAIN MLP]   Época 5/10 | precisión=82.3% | pérdida=0.451
    """

    PHASES = {
        "load": "LOAD DATA",
        "prep": "PREP FEAT",
        "train": "TRAIN MLP",
        "eval": "EVAL",
        "cnn": "CNN TRAIN",
        "ps": "PARAM SRV",
        "worker": "WORKER",
        "info": "INFO",
        "warn": "WARN",
        "error": "ERROR",
    }

    COLORS = {
        "load": "\033[94m",  # azul
        "prep": "\033[92m",  # verde
        "train": "\033[93m",  # amarillo
        "eval": "\033[96m",  # cian
        "cnn": "\033[95m",  # magenta
        "ps": "\033[94m",  # azul
        "worker": "\033[92m",  # verde
        "info": "\033[97m",  # blanco
        "warn": "\033[33m",  # naranja
        "error": "\033[91m",  # rojo
    }

    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True, use_timestamp: bool = False):
        """
        Inicializa el logger con colores y timestamp opcionales.

        :param use_colors: Si True, colorea los mensajes ANSI.
        :type use_colors: bool, default=True.

        :param use_timestamp: Si True, precede cada mensaje con timestamp.
        :type use_timestamp: bool, default=False.

        :return: None (inicialización in-place).
        :rtype: NoneType.
        """
        self.use_colors = use_colors
        self.use_timestamp = use_timestamp

    def _format_phase(self, phase: str) -> str:
        """
        Obtiene la etiqueta de fase formateada con color opcional.

        :param phase: Clave de fase (load, prep, train, eval, etc.).
        :type phase: str.

        :return: Etiqueta formateada con códigos ANSI opcionales.
        :rtype: str.
        """
        label = self.PHASES.get(phase, phase.upper())
        if self.use_colors:
            color = self.COLORS.get(phase, "")
            return f"{color}[{label}]{self.RESET}"
        return f"[{label}]"

    def _format_timestamp(self) -> str:
        """
        Retorna timestamp en formato HH:MM:SS si está habilitado, sino lista vacía.

        Utilizado internamente por log() para incluir información temporal en registros.

        :return: String con timestamp encerrado en espacios (ej: " 14:32:47 ") o space vacío.
        :rtype: str
        """
        if self.use_timestamp:
            return f" {datetime.now().strftime('%H:%M:%S')} "
        return " "

    def log(
        self,
        phase: str,
        message: str,
        progress: Optional[str] = None,
        metric: Optional[str] = None,
        file=None,
    ) -> None:
        """
        Registra un mensaje con formato unificado.

        :param phase: Tipo de fase (load, prep, train, eval, etc.)
        :param message: Mensaje descriptivo principal
        :param progress: Información de progreso opcional (ej: "45%", "32/100")
        :param metric: Métrica opcional (ej: "precisión=85.3%", "pérdida=0.512")
        :param file: Archivo para escribir (default: stdout)
        """
        if file is None:
            file = sys.stdout

        phase_fmt = self._format_phase(phase)
        timestamp = self._format_timestamp()

        parts = [phase_fmt + timestamp + message]

        if progress:
            parts.append(f"({progress})")

        if metric:
            parts.append(f"| {metric}")

        output = " ".join(parts)
        print(output, file=file)
        file.flush()

    def load(
        self,
        message: str,
        progress: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> None:
        """Registra evento de carga de datos."""
        self.log("load", message, progress, metric)

    def prep(
        self,
        message: str,
        progress: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> None:
        """Registra evento de preprocesamiento/extracción."""
        self.log("prep", message, progress, metric)

    def train(
        self,
        message: str,
        progress: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> None:
        """Registra evento de entrenamiento."""
        self.log("train", message, progress, metric)

    def eval(
        self,
        message: str,
        progress: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> None:
        """Registra evento de evaluación."""
        self.log("eval", message, progress, metric)

    def cnn(
        self,
        message: str,
        progress: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> None:
        """Registra evento de entrenamiento CNN."""
        self.log("cnn", message, progress, metric)

    def ps(
        self,
        message: str,
        progress: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> None:
        """Registra evento del Parameter Server."""
        self.log("ps", message, progress, metric)

    def worker(
        self,
        message: str,
        progress: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> None:
        """Registra evento de Worker."""
        self.log("worker", message, progress, metric)

    def info(self, message: str) -> None:
        """
        Registra mensaje informativo con tipo de fase 'info'.

        Wrapper sobre log() para facilitar logging de información general sin fase específica.

        :param message: Mensaje a registrar.
        :type message: str

        :return: None (escribe a stdout o archivo configurado).
        :rtype: NoneType.
        """
        self.log("info", message)

    def warn(self, message: str) -> None:
        """
        Registra advertencia con tipo de fase 'warn'.

        Wrapper sobre log() para alertas y condiciones no críticas.

        :param message: Mensaje de advertencia a registrar.
        :type message: str

        :return: None (escribe a stdout o archivo configurado).
        :rtype: NoneType.
        """
        self.log("warn", message)

    def error(self, message: str) -> None:
        """
        Registra error con tipo de fase 'error'.

        Wrapper sobre log() para errores y excepciones.

        :param message: Mensaje de error a registrar.
        :type message: str

        :return: None (escribe a stdout o archivo configurado).
        :rtype: NoneType.
        """
        self.log("error", message)

    def section(self, title: str) -> None:
        """
        Registra un título de sección rodeado de líneas separadoras.

        Imprime título centrado entre 70 caracteres de "=" para mejorar legibilidad visual
        de logs estructurados. Útil para separar fases de entrenamiento o reproducibilidad.

        :param title: Título de la sección a mostrar.
        :type title: str

        :return: None (escribe a stdout).
        :rtype: NoneType.
        """
        line = "=" * 70
        print(f"\n{line}")
        print(f"  {title}")
        print(f"{line}\n")


# Instancia global para usar en toda la aplicación
# (No usa colores por defecto en GUI para evitar caracteres ANSI en widgets de texto)
_global_logger = FormattedLogger(use_colors=False, use_timestamp=False)


def get_logger(use_colors: bool = False) -> FormattedLogger:
    """
    Obtiene instancia del logger.

    :param use_colors: Si True, retorna logger con colores ANSI (para terminal)
    :return: Instancia de FormattedLogger
    """
    return FormattedLogger(use_colors=use_colors, use_timestamp=False)
