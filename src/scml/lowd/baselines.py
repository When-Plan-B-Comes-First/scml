"""
Baseline clustering algorithms, tuned under the same protocol as AdaBox.

Every baseline is given a full grid search over its own parameter space and is
scored with the *same* objective used to tune AdaBox (SCOPE by default). This
is what makes the comparison fair: no algorithm is run at default settings
while another is tuned.

These grids reproduce the ones used in the published RQ1 benchmark.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, OPTICS

from .metrics import scope_score

# HDBSCAN: prefer scikit-learn's built-in (>=1.3), fall back to the standalone
# `hdbscan` package, and degrade gracefully if neither is present.
try:
    from sklearn.cluster import HDBSCAN as _SKHDBSCAN
    _HDBSCAN_SOURCE = "sklearn"
except ImportError:  # pragma: no cover
    _SKHDBSCAN = None
    try:
        import hdbscan as _hdbscan_pkg
        _HDBSCAN_SOURCE = "hdbscan"
    except ImportError:
        _hdbscan_pkg = None
        _HDBSCAN_SOURCE = None

HAS_HDBSCAN = _HDBSCAN_SOURCE is not None


def _score(X, y_true, labels, metric="SCOPE"):
    """Score a candidate labelling with the shared tuning objective."""
    try:
        if metric == "ARI":
            from sklearn.metrics import adjusted_rand_score
            return adjusted_rand_score(y_true, labels)
        return scope_score(X, y_true, labels)
    except Exception:
        return 0.0


def optimize_dbscan(X, y_true, metric="SCOPE"):
    """Grid-search DBSCAN, returning ``(labels, params)``."""
    best_score, best_params, best_labels = -1, None, None

    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5]
    min_samples_values = [3, 5, 7, 10, 15, 20, 30]

    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
                if len(np.unique(labels[labels >= 0])) > 0:
                    s = _score(X, y_true, labels, metric)
                    if s > best_score:
                        best_score = s
                        best_params = {"eps": eps, "min_samples": min_samples}
                        best_labels = labels
            except Exception:
                continue

    if best_labels is None:
        best_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
        best_params = {"eps": 0.5, "min_samples": 5}

    return best_labels, best_params


def optimize_optics(X, y_true, metric="SCOPE"):
    """Grid-search OPTICS, returning ``(labels, params)``."""
    best_score, best_params, best_labels = -1, None, None

    min_samples_values = [3, 5, 7, 10, 15]
    xi_values = [0.01, 0.05, 0.1, 0.2]
    min_cluster_size_values = [5, 10, 15, 20]

    for min_samples in min_samples_values:
        for xi in xi_values:
            for min_cluster_size in min_cluster_size_values:
                try:
                    labels = OPTICS(min_samples=min_samples, xi=xi,
                                    min_cluster_size=min_cluster_size).fit_predict(X)
                    if len(np.unique(labels[labels >= 0])) > 0:
                        s = _score(X, y_true, labels, metric)
                        if s > best_score:
                            best_score = s
                            best_params = {"min_samples": min_samples, "xi": xi,
                                           "min_cluster_size": min_cluster_size}
                            best_labels = labels
                except Exception:
                    continue

    if best_labels is None:
        best_labels = OPTICS(min_samples=5, xi=0.05,
                             min_cluster_size=10).fit_predict(X)
        best_params = {"min_samples": 5, "xi": 0.05, "min_cluster_size": 10}

    return best_labels, best_params


def _make_hdbscan(min_cluster_size, min_samples, epsilon, method):
    """Build an HDBSCAN estimator from whichever backend is installed."""
    if _HDBSCAN_SOURCE == "sklearn":
        return _SKHDBSCAN(min_cluster_size=min_cluster_size,
                          min_samples=min_samples,
                          cluster_selection_epsilon=epsilon,
                          cluster_selection_method=method)
    return _hdbscan_pkg.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                cluster_selection_epsilon=epsilon,
                                cluster_selection_method=method,
                                gen_min_span_tree=False)


def optimize_hdbscan(X, y_true, metric="SCOPE"):
    """Grid-search HDBSCAN, returning ``(labels, params)``.

    If no HDBSCAN backend is installed, returns all-noise labels and a params
    dict containing an ``error`` key, so callers can skip it cleanly.
    """
    if not HAS_HDBSCAN:
        return np.full(len(X), -1), {"error": "HDBSCAN not installed"}

    best_score, best_params, best_labels = -1, None, None

    min_cluster_size_values = [5, 10, 15, 20, 30, 50]
    min_samples_values = [1, 3, 5, 10, 15]
    cluster_selection_epsilon_values = [0.0, 0.1, 0.3, 0.5]
    cluster_selection_method_values = ["eom", "leaf"]

    for min_cluster_size in min_cluster_size_values:
        for min_samples in min_samples_values:
            for epsilon in cluster_selection_epsilon_values:
                for method in cluster_selection_method_values:
                    try:
                        labels = _make_hdbscan(min_cluster_size, min_samples,
                                               epsilon, method).fit_predict(X)
                        if len(np.unique(labels[labels >= 0])) > 0:
                            s = _score(X, y_true, labels, metric)
                            if s > best_score:
                                best_score = s
                                best_params = {
                                    "min_cluster_size": min_cluster_size,
                                    "min_samples": min_samples,
                                    "cluster_selection_epsilon": epsilon,
                                    "cluster_selection_method": method,
                                }
                                best_labels = labels
                    except Exception:
                        continue

    if best_labels is None:
        best_labels = _make_hdbscan(15, 5, 0.0, "eom").fit_predict(X)
        best_params = {"min_cluster_size": 15, "min_samples": 5,
                       "cluster_selection_epsilon": 0.0,
                       "cluster_selection_method": "eom"}

    return best_labels, best_params
