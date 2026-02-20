"""
Analytics — Ejecución de experimentos y análisis de resultados.
"""
from Analytics.experiment_runner import (
    run_single_experiment, run_multiple_experiments, compare_configurations,
)
from Analytics.statistics_engine import (
    compute_epoch_statistics, compute_partition_statistics,
    compute_convergence_epoch, compute_std,
)
from Analytics.chart_generator import (
    prepare_accuracy_chart_data, prepare_partition_comparison_data,
    prepare_convergence_data, prepare_distribution_data,
    prepare_comparison_chart_data, export_to_csv,
)

__all__ = [
    "run_single_experiment", "run_multiple_experiments", "compare_configurations",
    "compute_epoch_statistics", "compute_partition_statistics",
    "compute_convergence_epoch", "compute_std",
    "prepare_accuracy_chart_data", "prepare_partition_comparison_data",
    "prepare_convergence_data", "prepare_distribution_data",
    "prepare_comparison_chart_data", "export_to_csv",
]
