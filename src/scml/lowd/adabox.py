"""
AdaBox: the public, sklearn-style clustering estimator for the low-D track.

AdaBox discovers clusters of arbitrary shape from 2-D point data using adaptive
density boxes with Gaussian boundary refinement. It exposes a familiar
scikit-learn-style API (``fit`` / ``fit_predict`` / ``predict`` / ``score``).

IMPORTANT - tuning is where AdaBox's power lives
------------------------------------------------
AdaBox is a *calibrated* algorithm. Run with a single hand-picked parameter set
(plain ``fit``), it behaves like a generic density clusterer and will not match
DBSCAN/HDBSCAN. Its strength comes from the three-phase calibration (grid
search -> min_cluster_size -> anti-fragmentation), reached via ``tune`` here or
via the full Sample -> Label -> Calibrate -> Deploy workflow in :class:`scml.lowd.SLCD`.

Recommended (tuned) usage, when you have labels to calibrate against:
>>> from sklearn.datasets import make_blobs
>>> from scml.lowd import AdaBox
>>> X, y = make_blobs(n_samples=800, centers=4, cluster_std=1.0, random_state=0)
>>> model = AdaBox().tune(X, y)          # three-phase calibration
>>> labels = model.labels_               # strong result

Raw-engine usage (no calibration) is available via ``fit`` for ablations and
for cases where you already know good parameters:
>>> labels = AdaBox(n_boxes=12, min_density=5).fit_predict(X)
"""

from __future__ import annotations

import numpy as np

from ._engine import EADBBC2D_BR
from .scope import compute_dice_metrics


class AdaBox:
    """Adaptive density-based box clustering (low-D track).

    Parameters
    ----------
    n_boxes : int, default=20
        Grid resolution per axis. The 2-D feature space is divided into an
        ``n_boxes x n_boxes`` grid for density estimation.
    min_density : int, default=3
        Minimum points-per-box used as the absolute density threshold when
        ``use_relative_density=False``.
    min_growth_iterations : int, default=1
        Number of successful expansion steps a seed must complete before it
        graduates into a cluster. Lower values recover compact clusters; higher
        values demand more contiguous growth.
    merge_adjacent : bool, default=True
        Whether to merge adjacent box-clusters that fail a separation test.
    boundary_refinement : bool, default=True
        Whether to reassign noise points near cluster boundaries using a smooth
        density field (Gaussian filtering).
    refinement_sigma : float, default=1.5
        Smoothing scale for boundary refinement.
    use_relative_density : bool, default=False
        If True, density thresholds adapt to the global average density via
        ``relative_density_param`` instead of the absolute ``min_density``.
    relative_density_param : float, default=1.0
        Multiplier on the global average density when
        ``use_relative_density=True``.
    random_state : int, optional
        Reserved for reproducibility of any stochastic steps.
    **engine_kwargs
        Additional advanced parameters forwarded to the underlying engine
        (e.g. ``initial_threshold_factor``, ``regular_threshold_factor``,
        ``centroid_distance_threshold``).

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point after ``fit``. Noise is labelled ``-1``.
    n_clusters_ : int
        Number of clusters discovered (excluding noise).
    """

    def __init__(
        self,
        n_boxes=20,
        min_density=3,
        min_growth_iterations=1,
        merge_adjacent=True,
        boundary_refinement=True,
        refinement_sigma=1.5,
        use_relative_density=False,
        relative_density_param=1.0,
        random_state=None,
        **engine_kwargs,
    ):
        self.n_boxes = n_boxes
        self.min_density = min_density
        self.min_growth_iterations = min_growth_iterations
        self.merge_adjacent = merge_adjacent
        self.boundary_refinement = boundary_refinement
        self.refinement_sigma = refinement_sigma
        self.use_relative_density = use_relative_density
        self.relative_density_param = relative_density_param
        self.random_state = random_state
        self.engine_kwargs = engine_kwargs

        self.labels_ = None
        self.n_clusters_ = None
        self.best_params_ = None
        self._engine = None
        self._X = None

    def _build_engine(self):
        return EADBBC2D_BR(
            n_boxes=self.n_boxes,
            min_density=self.min_density,
            min_growth_iterations=self.min_growth_iterations,
            merge_adjacent=self.merge_adjacent,
            boundary_refinement=self.boundary_refinement,
            refinement_sigma=self.refinement_sigma,
            use_relative_density=self.use_relative_density,
            relative_density_param=self.relative_density_param,
            **self.engine_kwargs,
        )

    def fit(self, X, y=None):
        """Discover clusters in ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            2-D point data.
        y : ignored
            Present for API compatibility.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("AdaBox expects 2-D input of shape (n_samples, 2).")
        self._X = X
        self._engine = self._build_engine()
        self._engine.fit(X)
        self.labels_ = self._engine.predict_labels(X)
        self.n_clusters_ = int(len(set(self.labels_)) - (1 if -1 in self.labels_ else 0))
        return self

    def predict(self, X):
        """Assign labels to ``X`` using the fitted box-cluster grid."""
        if self._engine is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        return self._engine.predict_labels(X)

    def fit_predict(self, X, y=None):
        """Fit the model and return cluster labels for ``X``."""
        self.fit(X, y)
        return self.labels_

    def tune(self, X, y, direct_gs_threshold=5000):
        """Calibrate AdaBox with the three-phase optimization, then fit.

        Selects the path by dataset size:

        - **Up to ``direct_gs_threshold`` points (default 5,000): direct full-
          data tuning.** Exhaustive Grid Search with absolute density on the
          full dataset -- the exact protocol of the published benchmarks, which
          this implementation reproduces to three decimals. No SLCD: with no
          sampling there is no transfer, so relative density adds nothing.
        - **Beyond the threshold: use** :class:`scml.lowd.SLCD` **instead.**
          SLCD samples, calibrates with Random Search + relative density, and
          deploys; it exists precisely because full-data tuning becomes
          expensive at scale. ``tune`` on larger data falls back to Random
          Search + relative density on the full data for convenience, but SLCD
          is the recommended path.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            2-D point data.
        y : array-like of shape (n_samples,)
            Ground-truth labels used to score candidates during tuning.
        direct_gs_threshold : int, default=5000
            Upper bound for direct full-data Grid Search with absolute density.
            Above it, prefer SLCD.

        Returns
        -------
        self
            With ``labels_``, ``n_clusters_`` and ``best_params_`` populated.
        """
        from ._tuning import slcd_calibrate_adabox

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if len(X) <= direct_gs_threshold:
            search_method, mode = "GS", "OLD"      # exhaustive GS, absolute density
            n_trials = 0
        else:
            # Beyond the direct-GS range, SLCD is the recommended path; this
            # fallback runs RS + relative density on the full data.
            search_method, mode = "RS", "NEW"
            n_trials = 200 if len(X) > 20000 else 100

        labels, params = slcd_calibrate_adabox(
            X, y, tuning_metric="SCOPE", tuning_stages=3, mode=mode,
            search_method=search_method, n_trials=n_trials)
        if params is None:
            raise RuntimeError("Tuning failed to find valid parameters.")
        self.best_params_ = params
        self.labels_ = labels
        self.n_clusters_ = int(len(set(labels)) - (1 if -1 in labels else 0))
        self._X = X
        return self

    def score(self, X=None, y=None):
        """Return the SCOPE Overall score against ground-truth labels ``y``.

        Higher is better; 1.0 is a perfect structural match. Requires ground
        truth, mirroring scikit-learn's supervised ``score`` convention.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2), optional
            Defaults to the data passed to ``fit``.
        y : array-like of shape (n_samples,)
            Ground-truth cluster labels (``-1`` for noise).

        Returns
        -------
        float
            SCOPE Overall score in [0, 1].
        """
        if y is None:
            raise ValueError("score() requires ground-truth labels y.")
        if self.labels_ is None:
            raise RuntimeError("Call fit() before score().")
        X = self._X if X is None else np.asarray(X, dtype=float)
        result = compute_dice_metrics(X, np.asarray(y), self.labels_)
        return float(result["Overall_Score"])
