"""
Utils/results_exporter.py

Exportación de resultados de entrenamiento a JSON.

──────────────────────────────────────────────────────────────────
RESPONSABILIDAD
──────────────────────────────────────────────────────────────────
Centraliza la construcción del diccionario de resultados y su
escritura en disco. Tanto ``ps_terminal.py`` como ``ps_gui.py``
delegan aquí, evitando duplicación de lógica.

──────────────────────────────────────────────────────────────────
DESTINO DE LOS ARCHIVOS
──────────────────────────────────────────────────────────────────
Los resultados se guardan en ``Exports/`` (en la raíz del proyecto).
El directorio se crea automáticamente si no existe.

Cada experimento produce un archivo con nombre único basado en el
timestamp de finalización, por ejemplo:

    Exports/resultado_20250112_153042.json

Esto evita sobreescribir experimentos anteriores y permite
comparar varias ejecuciones sin gestión manual de archivos.

──────────────────────────────────────────────────────────────────
ESTRUCTURA DEL JSON GENERADO
──────────────────────────────────────────────────────────────────
{
  "configuracion": {
    "epochs": int,
    "cnn_arch": str,
    "hidden1": int,
    "hidden2": int,
    "learning_rate": float,
    "momentum": float,
    "n_train": int,
    "workers": int,
    "seed": int | null
  },
  "tiempo_ejecucion_segundos": float,
  "resumen": {
    "precision_final_entrenamiento": float,
    "mejor_precision_entrenamiento": float,
    "perdida_final_entrenamiento":   float,
    // solo si hay datos de prueba:
    "precision_final_prueba":        float,
    "mejor_precision_prueba":        float,
    "perdida_final_prueba":          float
  },
  "historial": {
    "accuracies":      [float, ...],
    "losses":          [float, ...],
    // solo si hay datos de prueba:
    "test_accuracies": [float, ...],
    "test_losses":     [float, ...]
  }
}
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

# Directorio de resultados relativo a la raíz del proyecto
_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Exports"
)


def export_results(
    history: Dict[str, list],
    config: Dict[str, Any],
    elapsed: float,
) -> str:
    """
    Construye el diccionario de resultados y lo escribe en ``Exports/``.

    El nombre del archivo incluye un timestamp para que cada ejecución
    produzca un archivo distinto y no se sobreescriban experimentos previos.

    :param history: Historial devuelto por ``ParameterServer.train()``.
                    Claves esperadas: ``"accuracies"``, ``"losses"`` y,
                    opcionalmente, ``"test_accuracies"`` y ``"test_losses"``.
    :type history: Dict[str, list]

    :param config: Hiperparámetros del experimento. Claves esperadas:
                   ``"epochs"``, ``"cnn_arch"``, ``"hidden1"``, ``"hidden2"``,
                   ``"learning_rate"``, ``"momentum"``, ``"n_train"``,
                   ``"workers"``, ``"seed"``.
    :type config: Dict[str, Any]

    :param elapsed: Tiempo total de entrenamiento en segundos.
    :type elapsed: float

    :return: Ruta absoluta del archivo JSON generado.
    :rtype: str
    """
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    # Si el historial está vacío (Worker desconectado antes de completar épocas),
    # no hay nada que guardar — evita IndexError en history["accuracies"][-1].
    if not history.get("accuracies"):
        return ""

    has_test = bool(history.get("test_accuracies"))

    results: Dict[str, Any] = {
        "configuracion": config,
        "tiempo_ejecucion_segundos": round(elapsed, 2),
        "resumen": {
            "precision_final_entrenamiento": round(history["accuracies"][-1], 4),
            "mejor_precision_entrenamiento": round(max(history["accuracies"]), 4),
            "perdida_final_entrenamiento": round(history["losses"][-1], 6),
        },
        "historial": {
            "accuracies": [round(v, 4) for v in history["accuracies"]],
            "losses": [round(v, 6) for v in history["losses"]],
        },
    }

    if has_test:
        results["resumen"]["precision_final_prueba"] = round(
            history["test_accuracies"][-1], 4
        )
        results["resumen"]["mejor_precision_prueba"] = round(
            max(history["test_accuracies"]), 4
        )
        results["resumen"]["perdida_final_prueba"] = round(
            history["test_losses"][-1], 6
        )
        results["historial"]["test_accuracies"] = [
            round(v, 4) for v in history["test_accuracies"]
        ]
        results["historial"]["test_losses"] = [
            round(v, 6) for v in history["test_losses"]
        ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resultado_{timestamp}.json"
    json_path = os.path.join(_RESULTS_DIR, filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return json_path
