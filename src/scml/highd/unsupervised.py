"""
Unsupervised tuning for AdaGraph -- no ground-truth labels required.

The supervised path (``AdaGraph().tune(X, y)``) scores candidates with SCOPE,
which needs ``y``. That is fine for benchmarking against labelled datasets, but
it is useless in the setting clustering is usually deployed in: you are
clustering precisely *because* you do not have labels.

This module closes that gap. It scores candidates with **Graph-SCOPE**, which
judges a clustering from kNN-graph topology alone. Same search space, same
engine, same 12 parameters -- only the objective changes:

    supervised:    SCOPE(X, y_true, labels)        needs y
    unsupervised:  Graph-SCOPE(knn_indices, labels) needs nothing but the data

Typical uses: reinforcement-learning state abstraction, exploratory analysis,
production pipelines where labels never exist.

A caution worth stating plainly. Graph-SCOPE is a *selection signal*, and
selecting with it does not guarantee agreement with any particular ground
truth you might later obtain -- topological structure and semantic labels are
not the same thing. When you do have labels, judge with SCOPE or ARI; never
report Graph-SCOPE as evidence that a Graph-SCOPE-selected clustering is good,
because that is circular.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._engine import AdaBoxGraph, precompute_knn
from .graph_scope import compute_graph_scope


def _sample_param_space(rng, reduced=False, aggressive=False):
    """Draw one configuration from the AdaGraph search space.

    Mirrors the spaces used by the supervised tuner: full (~1.3M
    combinations), reduced (~37k) and aggressive (~1.5k).
    """
    if aggressive:
        space = {
            "k_neighbors_frac": [1.5, 2.0, 2.5],
            "min_density": [2.0, 3.0, 4.0],
            "regular_threshold_factor": [0.8, 1.0, 1.2],
            "refinement_sigma": [1.0, 1.5],
            "min_cluster_size_frac": [0.3, 0.5],
            "merge_adjacent": [True],
            "use_mutual_knn": [False],
            "use_shared_neighbor_density": [False, True],
        }
    elif reduced:
        space = {
            "k_neighbors_frac": [1.0, 1.5, 2.0, 2.5, 3.0],
            "min_density": [1.5, 2.0, 3.0, 4.0, 5.0],
            "regular_threshold_factor": [0.7, 0.9, 1.0, 1.1, 1.3],
            "refinement_sigma": [0.5, 1.0, 1.5, 2.0],
            "min_cluster_size_frac": [0.2, 0.3, 0.5, 0.8],
            "merge_adjacent": [True, False],
            "use_mutual_knn": [False, True],
            "use_shared_neighbor_density": [False, True],
        }
    else:
        space = {
            "k_neighbors_frac": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
            "min_density": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
            "regular_threshold_factor": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4],
            "refinement_sigma": [0.5, 1.0, 1.5, 2.0, 2.5],
            "min_cluster_size_frac": [0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
            "merge_adjacent": [True, False],
            "use_mutual_knn": [False, True],
            "use_shared_neighbor_density": [False, True],
        }
    return {k: v[rng.integers(len(v))] for k, v in space.items()}


def tune_adagraph_unsupervised(X, n_trials=200, k_graph=15, gamma=1.5,
                               min_clusters=2, max_clusters=None,
                               use_engine_densities=True, reduced_search=False,
                               aggressive_search=False, random_state=42,
                               patience=None, verbose=False):
    """Tune AdaGraph without labels, scoring candidates with Graph-SCOPE.

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Data in native dimensionality.
    n_trials : int, default=200
        Random-search trials.
    k_graph : int, default=15
        Neighbours in the kNN graph used *for scoring*. Built once and reused
        across all trials, so this is cheap. Note this is separate from each
        candidate's own ``k_neighbors``, which the search varies.
    gamma : float, default=1.5
        Graph-SCOPE modularity resolution.
    min_clusters : int, default=2
        Reject candidates producing fewer clusters than this. A single cluster
        is degenerate and can score deceptively well on some components.
    max_clusters : int, optional
        Reject candidates producing more clusters than this. Useful when you
        know roughly how many states/regions to expect.
    use_engine_densities : bool, default=True
        Feed AdaGraph's own relative densities into Graph-SCOPE's noise
        component when available (the density-based C4). Falls back to graph
        cohesion otherwise.
    reduced_search, aggressive_search : bool
        Progressively smaller search spaces, for faster runs.
    patience : int, optional
        Stop after this many trials with no improvement.

    Returns
    -------
    labels : ndarray (n_samples,)
        Labels from the best configuration found.
    params : dict
        The winning parameters, plus ``graph_scope`` and ``n_clusters``.
    history : list of dict
        Per-trial record: parameters, score, cluster count.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)

    # Build the scoring graph once -- it is fixed for the dataset, so every
    # candidate is judged against the same topology.
    scoring_knn = precompute_knn(X, max_k=max(k_graph, 30))[1][:, :k_graph]

    best = {"score": -np.inf, "labels": None, "params": None}
    history = []
    since_improved = 0

    for trial in range(n_trials):
        params = _sample_param_space(rng, reduced_search, aggressive_search)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = AdaBoxGraph(verbose=False, **params)
                labels = np.asarray(model.fit_predict(X))
        except Exception:
            continue

        n_clusters = int(len(set(labels[labels >= 0])))
        if n_clusters < min_clusters:
            continue
        if max_clusters is not None and n_clusters > max_clusters:
            continue

        rd = None
        if use_engine_densities:
            rd = getattr(model, "relative_densities_", None)
            if rd is not None:
                rd = np.asarray(rd)
                if rd.shape[0] != len(X):
                    rd = None

        score = compute_graph_scope(scoring_knn, labels, gamma=gamma,
                                    relative_densities=rd)[0]
        history.append({"trial": trial, "graph_scope": float(score),
                        "n_clusters": n_clusters, **params})

        if score > best["score"]:
            best.update(score=float(score), labels=labels, params=dict(params))
            since_improved = 0
            if verbose:
                print(f"  trial {trial}: Graph-SCOPE={score:.4f} "
                      f"(k={n_clusters})")
        else:
            since_improved += 1
            if patience is not None and since_improved >= patience:
                break

    if best["labels"] is None:
        raise RuntimeError(
            "No valid clustering found. Try more trials, or relax "
            "min_clusters / max_clusters.")

    best["params"]["graph_scope"] = best["score"]
    best["params"]["n_clusters"] = int(
        len(set(best["labels"][best["labels"] >= 0])))
    return best["labels"], best["params"], history
