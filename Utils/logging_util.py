"""Utils/logging_util.py — Logger unificado para el sistema distribuido."""
import sys
from typing import Optional


class FormattedLogger:
    PHASES = {
        "ps":     "PARAM SRV",
        "worker": "WORKER   ",
        "train":  "TRAIN MLP",
        "warn":   "WARN     ",
        "error":  "ERROR    ",
    }
    COLORS = {
        "ps":     "\033[94m",
        "worker": "\033[92m",
        "train":  "\033[93m",
        "warn":   "\033[33m",
        "error":  "\033[91m",
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True) -> None:
        self.use_colors = use_colors

    def _fmt(self, phase: str) -> str:
        label = self.PHASES.get(phase, phase.upper())
        if self.use_colors:
            c = self.COLORS.get(phase, "")
            return f"{c}[{label}]{self.RESET}"
        return f"[{label}]"

    def log(self, phase: str, message: str,
            progress: Optional[str] = None,
            metric:   Optional[str] = None) -> None:
        parts = [self._fmt(phase) + " " + message]
        if progress: parts.append(f"({progress})")
        if metric:   parts.append(f"| {metric}")
        print(" ".join(parts), flush=True)

    def ps(self, msg: str, progress=None, metric=None):
        self.log("ps", msg, progress, metric)
    def worker(self, msg: str, progress=None, metric=None):
        self.log("worker", msg, progress, metric)
    def train(self, msg: str, progress=None, metric=None):
        self.log("train", msg, progress, metric)
    def warn(self, msg: str):
        self.log("warn", msg)
    def error(self, msg: str):
        self.log("error", msg)
    def section(self, title: str) -> None:
        line = "=" * 60
        print(f"\n{line}\n  {title}\n{line}\n")


def get_logger(use_colors: bool = False) -> FormattedLogger:
    return FormattedLogger(use_colors=use_colors)