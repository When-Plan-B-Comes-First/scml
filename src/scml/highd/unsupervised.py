"""
Unsupervised tuning for AdaGraph -- no ground-truth labels required.

The supervised path (``AdaGraph().tune(X, y)``) scores candidates with SCOPE,
which needs ``y``. That is fine for benchmarking against labelled datasets, but
useless in the setting clustering is usually deployed in: you are clustering
precisely *because* you do not have labels.

This module scores candidates with **Graph-SCOPE**, which judges a clustering
from kNN-graph topology alone. Same engine, same parameters -- only the
objective changes:

    supervised:    SCOPE(X, y_true, labels)         needs y
    unsupervised:  Graph-SCOPE(knn, densities, labels)  needs nothing but data

Typical uses: reinforcement-learning state abstraction, exploratory analysis,
production pipelines where labels never exist.

This is the author's validated implementation, lifted unchanged from the
production API. Note in particular which parameters are deliberately FIXED
rather than searched -- ``use_shared_neighbor_density=False``,
``use_mutual_knn=False``, ``merge_adjacent=True``. Randomising those produces
degenerate high-noise clusterings that score well on Graph-SCOPE while
recovering little real structure. Do not "improve" this grid without
re-validating.

A caution worth stating plainly. Graph-SCOPE is a *selection signal*. When you
do have labels, judge with SCOPE or ARI; never report Graph-SCOPE as evidence
that a Graph-SCOPE-selected clustering is good, because that is circular.
"""

from __future__ import annotations

import inspect
import time

import numpy as np

from ._engine import AdaBoxGraph, precompute_knn
from .graph_scope import compute_graph_scope as _graph_scope_2arg


def _get_adabox_model_keys():
    """Parameter names accepted by the engine, so unknown keys are dropped."""
    return set(inspect.signature(AdaBoxGraph.__init__).parameters.keys()) - {"self"}


def compute_graph_scope(knn_indices, relative_densities, labels):
    """Graph-SCOPE with the density-based noise component (C4).

    Signature matches the production API: densities are supplied positionally.
    Delegates to :func:`scml.highd.graph_scope.compute_graph_scope`.
    """
    return _graph_scope_2arg(knn_indices, labels,
                             relative_densities=relative_densities)


def tune_adaboxgraph_graph_scope(X_sample, n_trials=400, random_state=42,
                                 patience=80, max_seconds=None):
    """
    Random-search tuning using unsupervised Graph-SCOPE as the objective.

    Why faster than silhouette:
      - Graph-SCOPE is O(n · k) — purely numpy ops on the kNN graph that
        AdaBoxGraph already built.  No pairwise distance matrix.
      - We can afford 400 trials (same budget as LLM-supervised SCOPE)
        without a timeout problem.

    patience : int
        Stop if this many consecutive trials produce no improvement.
    max_seconds : float, optional
        Wall-clock cap. Default None (no cap). The production API uses 240 to
        bound a web request; research runs on large data need longer.

    Returns
    -------
    (best_labels, best_params, best_score, best_components)
    """
    rng        = np.random.RandomState(random_state)
    model_keys = _get_adabox_model_keys()

    # Same reduced search grid as tune_adaboxgraph_random(reduced_search=True)
    k_frac_choices        = [2.0, 2.6, 3.3, 3.9]
    min_density_choices   = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    reg_thr_choices       = [0.5, 0.7, 1.0, 1.1]
    ref_sigma_choices     = [2.0, 2.5, 3.0]
    mcs_frac_choices      = [0.15, 0.5, 1.0]
    centroid_dist_choices = [1.0, 2.0, 3.0, 5.0]
    ks_alpha_choices      = [0.01, 0.05, 0.1, 0.2]
    seed_hops_choices     = [2, 3]
    ks_merge_frac_choices = [0.3, 0.4, 0.5, 0.6]

    # Precompute kNN ONCE — shared across all trials (same as LLM path).
    # CRITICAL: max_k must cover the largest k that any k_frac choice will request.
    # k_eff = k_frac × sqrt(n_sample).  With k_frac up to 3.9 and n=1000,
    # k_eff ≈ 123.  If max_k=30 (too small), fit_predict falls back to recomputing
    # kNN from scratch on EVERY trial → O(n²) per trial → 400 trials × slow = hang.
    k_frac_max  = max(k_frac_choices)
    max_k       = min(int(np.ceil(k_frac_max * np.sqrt(len(X_sample)))) + 5,
                      len(X_sample) - 1)
    precomputed = precompute_knn(X_sample, max_k=max_k)

    best_score      = -1.0
    best_labels     = None
    best_params     = None
    best_components = None
    no_improve      = 0
    t_start         = time.time()
    # The production API hardcodes 240s to protect a web request. As a library
    # the default is no cap, since research runs on large data legitimately
    # take longer; pass max_seconds to reinstate a wall-clock limit.

    for _ in range(n_trials):
        if no_improve >= patience:
            break
        if max_seconds is not None and time.time() - t_start > max_seconds:
            break

        params = {
            'k_neighbors_frac':            float(rng.choice(k_frac_choices)),
            'min_density':                 float(rng.choice(min_density_choices)),
            'regular_threshold_factor':    float(rng.choice(reg_thr_choices)),
            'refinement_sigma':            float(rng.choice(ref_sigma_choices)),
            'min_cluster_size_frac':       float(rng.choice(mcs_frac_choices)),
            'centroid_distance_threshold': float(rng.choice(centroid_dist_choices)),
            'ks_test_alpha':               float(rng.choice(ks_alpha_choices)),
            'seed_exclusion_hops':         int(rng.choice(seed_hops_choices)),
            'ks_merge_fraction':           float(rng.choice(ks_merge_frac_choices)),
            'merge_adjacent':              True,
            'use_mutual_knn':              False,
            'use_shared_neighbor_density': False,
            'verbose':                     False,
        }
        valid_params = {k: v for k, v in params.items() if k in model_keys}
        try:
            mdl    = AdaBoxGraph(**valid_params)
            labels = mdl.fit_predict(X_sample, precomputed_knn=precomputed)
        except Exception:
            no_improve += 1
            continue

        k_found   = len(set(labels.tolist()) - {-1})
        noise_pct = float(np.sum(labels == -1)) / len(labels)
        if k_found < 2 or k_found > 150 or noise_pct > 0.5:
            no_improve += 1
            continue

        try:
            # mdl.knn_indices_ and mdl.relative_densities_ are set by fit()
            score, components = compute_graph_scope(
                mdl.knn_indices_, mdl.relative_densities_, labels
            )
        except Exception:
            no_improve += 1
            continue

        if score > best_score:
            best_score      = score
            best_labels     = labels.copy()
            best_params     = dict(params)
            best_components = components
            no_improve      = 0
        else:
            no_improve += 1

    return best_labels, best_params, best_score, best_components
