"""
Módulo de análisis estadístico y visualización para Algoritmo de Diego.
"""

from .statistics_engine import (
    compute_epoch_statistics,
    compute_partition_statistics,
    aggregate_experiments,
    compute_confidence_interval,
)

from .chart_generator import (
    prepare_accuracy_chart_data,
    prepare_partition_comparison_data,
    prepare_convergence_data,
    prepare_distribution_data,
)

from .experiment_runner import (
    run_single_experiment,
    run_multiple_experiments,
    compare_configurations,
    save_experiment_results,
    load_experiment_results,
)

__all__ = [
    # Statistics Engine
    "compute_epoch_statistics",
    "compute_partition_statistics",
    "aggregate_experiments",
    "compute_confidence_interval",
    # Chart Generator
    "prepare_accuracy_chart_data",
    "prepare_partition_comparison_data",
    "prepare_convergence_data",
    "prepare_distribution_data",
    # Experiment Runner
    "run_single_experiment",
    "run_multiple_experiments",
    "compare_configurations",
    "save_experiment_results",
    "load_experiment_results",
]
