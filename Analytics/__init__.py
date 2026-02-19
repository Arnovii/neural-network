"""
Analytics — Ejecución de experimentos y análisis de resultados.

Expone el orquestador de experimentos, las estadísticas por época/partición
y las funciones de preparación de datos para gráficos.
"""

from Analytics.experiment_runner import (
    run_single_experiment,
    run_multiple_experiments,
    compare_configurations,
)

from Analytics.statistics_engine import (
    compute_epoch_statistics,
    compute_partition_statistics,
    compute_convergence_epoch,
    compute_std,
)

from Analytics.chart_generator import (
    prepare_accuracy_chart_data,
    prepare_partition_comparison_data,
    prepare_convergence_data,
    prepare_distribution_data,
    prepare_comparison_chart_data,
    export_to_csv,
)

__all__ = [
    # experiment_runner
    "run_single_experiment",
    "run_multiple_experiments",
    "compare_configurations",
    # statistics_engine
    "compute_epoch_statistics",
    "compute_partition_statistics",
    "compute_convergence_epoch",
    "compute_std",
    # chart_generator
    "prepare_accuracy_chart_data",
    "prepare_partition_comparison_data",
    "prepare_convergence_data",
    "prepare_distribution_data",
    "prepare_comparison_chart_data",
    "export_to_csv",
]
