"""High-D track: AdaGraph, Graph-SCOPE, and high-D SLCD.

Clusters directly in the original high-dimensional space using a kNN
neighbourhood graph, instead of projecting to 2-D first.

Public API:

    from scml.highd import AdaGraph, SLCD

AdaGraph is the clustering algorithm; SLCD (Sample -> Learn -> Classify ->
Deploy) scales it by tuning on a density-aware sample and classifying the
remaining points by a two-pass kNN vote. This differs from the low-D SLCD
(Sample -> Label -> Calibrate -> Deploy), which transfers tuned parameters;
the two implementations are not interchangeable.
"""

from .adagraph import AdaGraph
from .slcd import SLCD, default_sample_size
from .graph_scope import (graph_scope_score, graph_scope_report,
                          build_knn_graph, compute_graph_scope)
from .benchmark import benchmark_highd, compare_k_selection, tune_hdbscan
from ._engine import precompute_knn, density_aware_sample, prototype_deploy

__all__ = ["AdaGraph", "SLCD", "default_sample_size",
           "graph_scope_score", "graph_scope_report", "build_knn_graph",
           "compute_graph_scope", "benchmark_highd", "compare_k_selection",
           "tune_hdbscan", "precompute_knn", "density_aware_sample",
           "prototype_deploy"]
