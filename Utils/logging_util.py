"""Utils/logging_util.py — Logger unificado para el sistema distribuido."""

from typing import Optional


class FormattedLogger:
    PHASES = {
        "ps": "PARAM SRV",
        "worker": "WORKER",
        "train": "TRAIN MLP",
        "warn": "WARN",
        "error": "ERROR",
    }
    COLORS = {
        "ps": "\033[94m",
        "worker": "\033[92m",
        "train": "\033[93m",
        "warn": "\033[33m",
        "error": "\033[91m",
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True) -> None:
        """
        Inicializa un FormattedLogger con output coloreado opcional.

        :param use_colors: Si se deben usar códigos de color ANSI en output.
        :type use_colors: bool

        :returns: None
        :rtype: None
        """
        self.use_colors = use_colors

    def _fmt(self, phase: str) -> str:
        """
        Formatea una etiqueta de fase con códigos de color ANSI opcionales.

        Busca la fase en diccionario PHASES para obtener etiqueta con padding, y opcionalmente
        la envuelve con códigos de color del diccionario COLORS.

        :param phase: Identificador de fase (ej: 'ps', 'worker', 'train', 'warn', 'error').
        :type phase: str

        :returns: String de fase formateado, con o sin códigos de color.
        :rtype: str
        """
        label = self.PHASES.get(phase, phase.upper())
        if self.use_colors:
            c = self.COLORS.get(phase, "")
            return f"{c}[{label}]{self.RESET}"
        return f"[{label}]"

    def log(
        self,
        phase: str,
        message: str,
        progress: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> None:
        """
        Registra mensaje formateado con información de progreso y métrica opcionales.

        Combina etiqueta de fase formateada, mensaje, string de progreso, y string de métrica
        en una sola línea de salida e imprime a stdout.

        :param phase: Identificador de fase para formateo (ej: 'ps', 'worker').
        :type phase: str

        :param message: Texto del mensaje principal.
        :type message: str

        :param progress: Información de progreso opcional (ej: "100 / 1000").
        :type progress: Optional[str]

        :param metric: String de métrica opcional (ej: "loss=0.23").
        :type metric: Optional[str]

        :returns: None
        :rtype: None
        """
        parts = [self._fmt(phase) + " " + message]
        if progress:
            parts.append(f"({progress})")
        if metric:
            parts.append(f"| {metric}")
        print(" ".join(parts), flush=True)

    def ps(self, msg: str, progress=None, metric=None):
        """
        Registra mensaje con formateo de fase 'ps' (Parameter Server).

        Método de conveniencia que llama log() con phase='ps'.

        :param msg: Texto del mensaje.
        :type msg: str

        :param progress: Información de progreso opcional.
        :type progress: Optional[str]

        :param metric: Información de métrica opcional.
        :type metric: Optional[str]

        :returns: None
        :rtype: None
        """
        self.log("ps", msg, progress, metric)

    def worker_msg(self, worker_id, msg: str, progress=None, metric=None):
        """
        Registra mensaje con formateo de fase 'worker' incluido el Worker ID.

        Agrega prefijo [Wid] al mensaje para identificación del Worker.

        :param worker_id: ID del Worker (int o None para mostrar '?')
        :type worker_id: Optional[int]

        :param msg: Texto del mensaje.
        :type msg: str

        :param progress: Información de progreso opcional.
        :type progress: Optional[str]

        :param metric: Información de métrica opcional.
        :type metric: Optional[str]

        :returns: None
        :rtype: None
        """
        wid = worker_id if worker_id is not None else "?"
        full_msg = f"[W{wid}] {msg}"
        self.log("worker", full_msg, progress, metric)

    def train(self, msg: str, progress=None, metric=None):
        """
        Registra mensaje con formateo de fase 'train' (Entrenamiento MLP).

        Método de conveniencia que llama log() con phase='train'.

        :param msg: Texto del mensaje.
        :type msg: str

        :param progress: Información de progreso opcional.
        :type progress: Optional[str]

        :param metric: Información de métrica opcional.
        :type metric: Optional[str]

        :returns: None
        :rtype: None
        """
        self.log("train", msg, progress, metric)

    def warn(self, msg: str):
        """
        Registra mensaje de advertencia con formateo de fase 'warn'.

        Método de conveniencia que llama log() con phase='warn'.

        :param msg: Texto del mensaje de advertencia.
        :type msg: str

        :returns: None
        :rtype: None
        """
        self.log("warn", msg)

    def error(self, msg: str):
        """
        Registra mensaje de error con formateo de fase 'error'.

        Método de conveniencia que llama log() con phase='error'.

        :param msg: Texto del mensaje de error.
        :type msg: str

        :returns: None
        :rtype: None
        """
        self.log("error", msg)

    def section(self, title: str) -> None:
        """
        Imprime encabezado de sección formateado con título.

        Muestra línea de signos de igualdad antes y después del título, con el
        título centrado entre ellos.

        :param title: Texto del encabezado de sección.
        :type title: str

        :returns: None
        :rtype: None
        """
        line = "=" * 60
        print(f"\n{line}\n  {title}\n{line}\n")


def get_logger(use_colors: bool = False) -> FormattedLogger:
    """
    Función de fábrica para instanciar un FormattedLogger.

    :param use_colors: Si se deben habilitar códigos de color ANSI (default: False).
    :type use_colors: bool

    :returns: Nueva instancia FormattedLogger.
    :rtype: FormattedLogger
    """
    return FormattedLogger(use_colors=use_colors)
