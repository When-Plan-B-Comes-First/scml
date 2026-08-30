"""
Baseline clustering algorithms, tuned under the same protocol as AdaBox.

Every baseline is given a full grid search over its own parameter space and is
scored with the *same* objective used to tune AdaBox (SCOPE by default). This
is what makes the comparison fair: no algorithm is run at default settings
while another is tuned.

These grids reproduce the ones used in the published RQ1 benchmark.
"""

from __future__ import annotations

import time

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


def _estimate_avg_neighbor_fraction(X, eps, sample_size=300, seed=0):
    """Fast estimate of what fraction of the dataset a typical point's
    eps-neighborhood covers, using a random sample and a radius query.

    Used to detect (eps, min_samples) combinations that are guaranteed to
    collapse into one giant cluster before paying for the full fit -- at such
    density, DBSCAN's neighbor graph can have close to n^2 edges, which is
    both extremely expensive and produces a degenerate result regardless.
    Skipping these loses nothing: they were never going to be the best answer.
    """
    from sklearn.neighbors import NearestNeighbors
    n = len(X)
    if n <= sample_size:
        sample = X
    else:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=sample_size, replace=False)
        sample = X[idx]
    nn = NearestNeighbors(radius=eps).fit(X)
    counts = nn.radius_neighbors(sample, return_distance=False)
    avg_neighbors = np.mean([len(c) for c in counts])
    return avg_neighbors / n


def optimize_dbscan(X, y_true, metric="SCOPE", max_seconds=None,
                    density_guard=0.3):
    """Grid-search DBSCAN, returning ``(labels, params)``.

    Some (eps, min_samples) combinations are pathologically expensive on
    large datasets: a large eps on standardized data can give nearly every
    point thousands of neighbors, pushing DBSCAN's neighbor graph toward
    O(n^2) edges -- this can exhaust memory or take a very long time. Such
    configurations also always collapse into one giant cluster, which scores
    poorly regardless, so skipping them costs nothing in search quality.

    ``density_guard`` (default 0.3) skips a candidate eps if a fast sample
    check shows the typical point's neighborhood already covers more than
    this fraction of the whole dataset -- a reliable predictor of a
    degenerate, single-cluster result. Set to ``None`` to disable and try
    every combination regardless of cost (not recommended above a few
    thousand points).

    By default the full (non-skipped) grid is always searched
    (``max_seconds=None``), matching the exhaustive protocol used to produce
    the published results -- fairness to the baselines means giving them the
    same unrestricted search AdaBox gets. ``max_seconds`` is an optional extra
    safety valve for capped, faster runs; it is off by default.
    """
    best_score, best_params, best_labels = -1, None, None
    t0 = time.time()
    n_tried, n_skipped_dense, n_skipped_degenerate = 0, 0, 0

    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5]
    min_samples_values = [3, 5, 7, 10, 15, 20, 30]
    n_total = len(eps_values) * len(min_samples_values)

    # Pre-screen eps values once (density doesn't depend on min_samples) so
    # the expensive check runs 11 times, not 77.
    skip_eps = set()
    if density_guard is not None and len(X) > 2000:
        for eps in eps_values:
            frac = _estimate_avg_neighbor_fraction(X, eps)
            if frac > density_guard:
                skip_eps.add(eps)

    stopped_early = False
    for min_samples in min_samples_values:
        # eps_values must be ascending: once a given eps collapses the data
        # into a single cluster, every larger eps provably does too (larger
        # eps only ever merges, never splits), so the rest of the row is both
        # useless and progressively more expensive -- large-eps fits on big
        # data build near-complete neighbor graphs and can exhaust memory.
        for eps in sorted(eps_values):
            if max_seconds is not None and time.time() - t0 > max_seconds:
                stopped_early = True
                break
            if eps in skip_eps:
                n_skipped_dense += 1
                continue
            try:
                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
                n_tried += 1
                n_found = len(np.unique(labels[labels >= 0]))
                if n_found > 0:
                    s = _score(X, y_true, labels, metric)
                    if s > best_score:
                        best_score = s
                        best_params = {"eps": eps, "min_samples": min_samples}
                        best_labels = labels
                # Degenerate result: everything in one cluster. Larger eps
                # cannot improve on this, so stop scanning this row.
                if n_found <= 1:
                    n_skipped_degenerate += sum(1 for e in eps_values if e > eps)
                    break
            except Exception:
                continue
        if stopped_early:
            break

    total_skipped = n_skipped_dense + n_skipped_degenerate
    if total_skipped:
        reasons = []
        if n_skipped_dense:
            reasons.append(f"{n_skipped_dense} with eps in {sorted(skip_eps)} "
                           f"(neighborhood >{density_guard:.0%} of data)")
        if n_skipped_degenerate:
            reasons.append(f"{n_skipped_degenerate} after eps collapsed to a "
                           f"single cluster (larger eps cannot improve)")
        print(f"    (DBSCAN: ran {n_tried}/{n_total} combinations; skipped "
              f"{total_skipped} that were provably degenerate: "
              + "; ".join(reasons) + ")")
    if stopped_early:
        print(f"    (DBSCAN: stopped after {max_seconds}s, "
              f"{n_tried} combinations tried)")

    if best_labels is None:
        best_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
        best_params = {"eps": 0.5, "min_samples": 5}

    return best_labels, best_params


def optimize_optics(X, y_true, metric="SCOPE", max_seconds=None):
    """Grid-search OPTICS, returning ``(labels, params)``.

    By default the full grid is always searched (``max_seconds=None``). See
    :func:`optimize_dbscan` for why this is the default and when to override it.
    """
    best_score, best_params, best_labels = -1, None, None
    t0 = time.time()
    n_tried = 0

    min_samples_values = [3, 5, 7, 10, 15]
    xi_values = [0.01, 0.05, 0.1, 0.2]
    min_cluster_size_values = [5, 10, 15, 20]
    n_total = len(min_samples_values) * len(xi_values) * len(min_cluster_size_values)

    stopped_early = False
    for min_samples in min_samples_values:
        for xi in xi_values:
            for min_cluster_size in min_cluster_size_values:
                if max_seconds is not None and time.time() - t0 > max_seconds:
                    stopped_early = True
                    break
                try:
                    labels = OPTICS(min_samples=min_samples, xi=xi,
                                    min_cluster_size=min_cluster_size).fit_predict(X)
                    n_tried += 1
                    if len(np.unique(labels[labels >= 0])) > 0:
                        s = _score(X, y_true, labels, metric)
                        if s > best_score:
                            best_score = s
                            best_params = {"min_samples": min_samples, "xi": xi,
                                           "min_cluster_size": min_cluster_size}
                            best_labels = labels
                except Exception:
                    continue
            if stopped_early:
                break
        if stopped_early:
            break

    if stopped_early:
        print(f"    (OPTICS: stopped after {max_seconds}s, "
              f"{n_tried}/{n_total} combinations tried)")

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


def optimize_hdbscan(X, y_true, metric="SCOPE", max_seconds=None):
    """Grid-search HDBSCAN, returning ``(labels, params)``.

    By default the full grid is always searched (``max_seconds=None``); see
    :func:`optimize_dbscan` for why. If no HDBSCAN backend is installed,
    returns all-noise labels and a params dict containing an ``error`` key,
    so callers can skip it cleanly.
    """
    if not HAS_HDBSCAN:
        return np.full(len(X), -1), {"error": "HDBSCAN not installed"}

    best_score, best_params, best_labels = -1, None, None
    t0 = time.time()
    n_tried = 0

    min_cluster_size_values = [5, 10, 15, 20, 30, 50]
    min_samples_values = [1, 3, 5, 10, 15]
    cluster_selection_epsilon_values = [0.0, 0.1, 0.3, 0.5]
    cluster_selection_method_values = ["eom", "leaf"]
    n_total = (len(min_cluster_size_values) * len(min_samples_values)
              * len(cluster_selection_epsilon_values)
              * len(cluster_selection_method_values))

    stopped_early = False
    for min_cluster_size in min_cluster_size_values:
        for min_samples in min_samples_values:
            for epsilon in cluster_selection_epsilon_values:
                for method in cluster_selection_method_values:
                    if max_seconds is not None and time.time() - t0 > max_seconds:
                        stopped_early = True
                        break
                    try:
                        labels = _make_hdbscan(min_cluster_size, min_samples,
                                               epsilon, method).fit_predict(X)
                        n_tried += 1
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
                if stopped_early:
                    break
            if stopped_early:
                break
        if stopped_early:
            break

    if stopped_early:
        print(f"    (HDBSCAN: stopped after {max_seconds}s, "
              f"{n_tried}/{n_total} combinations tried)")

    if best_labels is None:
        best_labels = _make_hdbscan(15, 5, 0.0, "eom").fit_predict(X)
        best_params = {"min_cluster_size": 15, "min_samples": 5,
                       "cluster_selection_epsilon": 0.0,
                       "cluster_selection_method": "eom"}

    return best_labels, best_params
