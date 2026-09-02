"""
AdaGraph: the public clustering estimator for the high-D track.

AdaGraph clusters directly in the original high-dimensional space. Instead of
projecting to 2-D first (PCA/t-SNE/UMAP), which discards information, it builds
a k-nearest-neighbour graph and runs adaptive density clustering on that graph.
The premise: in high dimensions absolute distances become uninformative, but
*relative neighbourhoods* remain meaningful.

Pipeline (five stages, see the algorithm reference):
  1. Build the kNN neighbourhood graph and estimate relative density
  2. Seed initialisation at density peaks
  3. Iterative growth with graduation
  3.5 Fragment filtering
  4. Statistical cluster merging (KS test)
  5. Boundary refinement by kNN vote

Tuning is where the power lives, exactly as in the low-D track. Use ``tune``
when you have labels for a modest dataset, or :class:`scml.highd.SLCD` at
scale.

Example
-------
>>> from sklearn.datasets import make_blobs
>>> from scml.highd import AdaGraph
>>> X, y = make_blobs(n_samples=600, n_features=20, centers=5, random_state=0)
>>> labels = AdaGraph().fit_predict(X)
"""

from __future__ import annotations

import numpy as np

from ._engine import AdaBoxGraph, precompute_knn, tune_adaboxgraph_random


class AdaGraph:
    """Graph-native adaptive density clustering for high-dimensional data.

    Parameters
    ----------
    k_neighbors : int, default=15
        Neighbours per node in the kNN graph. Ignored when
        ``k_neighbors_frac`` is set.
    k_neighbors_frac : float, optional
        Scale-invariant alternative: ``k = round(frac * log2(n))``, clamped to
        ``[3, n-1]``. Preferred when dataset size varies, since it keeps the
        graph's connectivity comparable across scales.
    min_density : float, default=3.0
        Density threshold factor; the working threshold is
        ``mean_density * (min_density / 3.0)``.
    regular_threshold_factor : float, default=1.0
        Growth-phase density scaling.
    merge_adjacent : bool, default=True
        Enable statistical (KS-test) merging of adjacent clusters.
    refinement_sigma : float, default=1.5
        Boundary refinement aggressiveness.
    min_cluster_size : int, default=5
        Minimum points for a retained cluster. Ignored when
        ``min_cluster_size_frac`` is set.
    min_cluster_size_frac : float, optional
        Scale-invariant alternative: ``mcs = max(3, round(frac * sqrt(n)))``.
    use_mutual_knn : bool, default=False
        Require reciprocal neighbour agreement for an edge. More conservative;
        useful on noisy data.
    use_shared_neighbor_density : bool, default=False
        Use shared-neighbour (SNN) density instead of distance-based density.
        More robust when Euclidean distance degrades at very high dimension.
    **engine_kwargs
        Further engine parameters (``centroid_distance_threshold``,
        ``ks_test_alpha``, ``seed_exclusion_hops``, ``ks_merge_fraction``,
        ``min_growth_iterations``).

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels; ``-1`` marks noise.
    n_clusters_ : int
        Number of clusters found, excluding noise.
    best_params_ : dict
        Parameters selected by ``tune`` (None if ``tune`` was not used).
    """

    def __init__(self, k_neighbors=15, k_neighbors_frac=None, min_density=3.0,
                 regular_threshold_factor=1.0, merge_adjacent=True,
                 refinement_sigma=1.5, min_cluster_size=5,
                 min_cluster_size_frac=None, use_mutual_knn=False,
                 use_shared_neighbor_density=False, verbose=False,
                 **engine_kwargs):
        self.k_neighbors = k_neighbors
        self.k_neighbors_frac = k_neighbors_frac
        self.min_density = min_density
        self.regular_threshold_factor = regular_threshold_factor
        self.merge_adjacent = merge_adjacent
        self.refinement_sigma = refinement_sigma
        self.min_cluster_size = min_cluster_size
        self.min_cluster_size_frac = min_cluster_size_frac
        self.use_mutual_knn = use_mutual_knn
        self.use_shared_neighbor_density = use_shared_neighbor_density
        self.verbose = verbose
        self.engine_kwargs = engine_kwargs

        self.labels_ = None
        self.n_clusters_ = None
        self.best_params_ = None
        self.tuning_score_ = None
        self._engine = None
        self._X = None

    def _build(self, **overrides):
        params = dict(
            k_neighbors=self.k_neighbors,
            k_neighbors_frac=self.k_neighbors_frac,
            min_density=self.min_density,
            regular_threshold_factor=self.regular_threshold_factor,
            merge_adjacent=self.merge_adjacent,
            refinement_sigma=self.refinement_sigma,
            min_cluster_size=self.min_cluster_size,
            min_cluster_size_frac=self.min_cluster_size_frac,
            use_mutual_knn=self.use_mutual_knn,
            use_shared_neighbor_density=self.use_shared_neighbor_density,
            verbose=self.verbose,
        )
        params.update(self.engine_kwargs)
        params.update(overrides)
        return AdaBoxGraph(**params)

    def fit(self, X, y=None, precomputed_knn=None):
        """Cluster ``X`` with the current parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data in its native dimensionality -- no reduction required.
        y : ignored
            Present for API compatibility.
        precomputed_knn : tuple, optional
            ``(distances, indices)`` from :func:`precompute_knn`, to avoid
            recomputing the graph across repeated fits.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got {X.shape}.")
        self._X = X
        self._engine = self._build()
        self.labels_ = self._engine.fit_predict(X, precomputed_knn=precomputed_knn)
        self.n_clusters_ = int(len(set(self.labels_[self.labels_ >= 0])))
        return self

    def fit_predict(self, X, y=None, precomputed_knn=None):
        """Cluster ``X`` and return the labels."""
        self.fit(X, y, precomputed_knn=precomputed_knn)
        return self.labels_

    def tune(self, X, y, n_trials=400, expected_k=None, reduced_search=False,
             aggressive_search=False, n_jobs=-1, patience=100, verbose=False):
        """Random-search calibration on labelled data, scored by SCOPE.

        Twelve parameters are searched. The kNN graph is built once at
        ``max_k=30`` and reused across every trial, which is what makes a
        400-trial search affordable.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Ground-truth labels used to score candidates.
        n_trials : int, default=400
            Random search trials.
        expected_k : int, optional
            Hint for the number of clusters; penalises trials far from it.
        reduced_search, aggressive_search : bool
            Progressively smaller search spaces (36,864 and 1,458 combinations
            respectively, versus 1,327,104 for the full space).
        n_jobs : int, default=-1
            Parallel workers for trial execution.
        patience : int, default=100
            Stop after this many trials without improvement.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        result = tune_adaboxgraph_random(
            X, y, n_trials=n_trials, expected_k=expected_k, verbose=verbose,
            n_jobs=n_jobs, patience=patience, reduced_search=reduced_search,
            aggressive_search=aggressive_search)
        # engine returns (labels, params, score) -- or a 4th trial_log element
        labels, params, score = result[0], result[1], result[2]
        self.labels_ = np.asarray(labels)
        self.best_params_ = params
        self.tuning_score_ = float(score)
        self.n_clusters_ = int(len(set(self.labels_[self.labels_ >= 0])))
        self._X = X
        return self

    def score(self, X=None, y=None):
        """SCOPE Overall score against ground-truth labels ``y``."""
        from ..lowd.scope import compute_dice_metrics
        if y is None:
            raise ValueError("score() requires ground-truth labels y.")
        if self.labels_ is None:
            raise RuntimeError("Call fit() or tune() before score().")
        X = self._X if X is None else np.asarray(X, dtype=float)
        return float(compute_dice_metrics(X, np.asarray(y), self.labels_,
                                          verbose=False)["Overall_Score"])
