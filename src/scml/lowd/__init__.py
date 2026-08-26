"""Low-D track: SCOPE metric, AdaBox clustering, and SLCD parameter transfer.

This track ships first and complete. Public API:

    from scml.lowd import AdaBox, SLCD, scope_score, scope_report

Tuning is the heart of AdaBox. Up to ~5,000 points use AdaBox().tune(X, y)
(exhaustive Grid Search on the full data, absolute density) -- the protocol
behind the published benchmarks. Beyond that use SLCD(...).fit_predict(X, y)
(Random Search on a sample, relative/scale-invariant density, then deploy).
Plain AdaBox(...).fit(X) runs the raw engine (ablation only).

To compare AdaBox against DBSCAN/OPTICS/HDBSCAN on your own dataset in one
call, use benchmark_dataset(...).
"""

from .adabox import AdaBox
from .slcd import SLCD, stratified_sample, stratified_sample_excluding, default_sample_size
from .metrics import scope_score, scope_report
from .prepare import prepare_dataset
from .baselines import optimize_dbscan, optimize_optics, optimize_hdbscan
from .benchmark import (benchmark_dataset, compare_algorithms,
                        plot_metric_bars, plot_clustering_results)

__all__ = ["AdaBox", "SLCD", "scope_score", "scope_report",
           "stratified_sample", "stratified_sample_excluding",
           "default_sample_size", "prepare_dataset", "benchmark_dataset",
           "compare_algorithms", "plot_metric_bars", "plot_clustering_results",
           "optimize_dbscan", "optimize_optics", "optimize_hdbscan"]
