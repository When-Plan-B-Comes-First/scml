"""
SLCD (high-D track): Sample -> Learn -> Classify -> Deploy.

SLCD names a family, not one algorithm. Both tracks share the invariant that
gives it value -- **the full dataset is never tuned** -- but they reach it by
different mechanisms, and the two implementations are not interchangeable:

  * Low-D  (scml.lowd): Sample -> Label -> Calibrate -> Deploy.
    "Deploy" is true *parameter transfer*: parameters tuned on the sample are
    used to cluster the full dataset.
  * High-D (this module): Sample -> Learn -> Classify -> Deploy.
    "Deploy" is *point assignment*: AdaGraph clusters the sample, then the
    remaining points are classified into the sample's clusters by a two-pass
    kNN vote.

The high-D sample is drawn by density-aware sampling rather than stratified
sampling: points hill-climb to their local density peak, and each resulting
mode contributes proportionally with a guaranteed minimum. This keeps rare
density modes represented, which uniform random sampling can miss entirely.
"""

from __future__ import annotations

import numpy as np

from ._engine import (adaptive_deploy, density_aware_sample, prototype_deploy,
                      tune_adaboxgraph_random)


def default_sample_size(n_points):
    """Size-adaptive sample for high-D SLCD.

    The sample must be large enough for AdaGraph to see the structure, but
    small enough that tuning stays cheap. The author's benchmarks use 1,000
    points at 10k-50k scale; these tiers extend that behaviour.
    """
    if n_points <= 2_000:
        return n_points          # small enough to cluster directly
    if n_points < 20_000:
        return 1_000
    if n_points < 200_000:
        return 2_000
    return 5_000


class SLCD:
    """Sample -> Learn -> Classify -> Deploy workflow for AdaGraph (high-D).

    Parameters
    ----------
    sample_size : int, optional
        Points drawn for the learning sample. Default picks by dataset size
        (see :func:`default_sample_size`).
    n_trials : int, default=400
        Random Search trials during the Learn stage. AdaGraph has 12
        parameters and reuses one precomputed kNN graph across trials, so a
        400-trial search is affordable.
    expected_k : int, optional
        Hint for the number of clusters.
    deploy_method : {"prototype", "adaptive"}, default="prototype"
        How the Classify/Deploy stage assigns the remaining points.
        ``"prototype"`` runs the two-pass kNN vote using the sample's labelled
        points as prototypes -- pass 1 establishes bulk structure, pass 2
        re-votes low-confidence boundary points against a much larger
        prototype set. ``"adaptive"`` instead re-runs AdaGraph on the full
        data with a scaled ``k_neighbors``.
    k_vote : int, default=7
        Neighbours consulted per point in the prototype vote.
    reduced_search, aggressive_search : bool
        Use progressively smaller search spaces during Learn.
    min_per_mode : int, default=20
        Minimum points sampled from each discovered density mode.
    n_jobs : int, default=-1
        Parallel workers for tuning trials.
    random_state : int, default=42

    Attributes
    ----------
    labels_ : ndarray
        Labels for the full dataset after deployment.
    sample_indices_ : ndarray
        Indices of the points used for learning.
    sample_labels_ : ndarray
        AdaGraph's labels on the sample.
    best_params_ : dict
        Parameters selected during Learn.
    sample_size_, n_trials_ : int
        Values actually used.
    """

    def __init__(self, sample_size=None, n_trials=400, expected_k=None,
                 deploy_method="prototype", k_vote=7, reduced_search=False,
                 aggressive_search=False, min_per_mode=20, n_jobs=-1,
                 random_state=42, verbose=False):
        self.sample_size = sample_size
        self.n_trials = n_trials
        self.expected_k = expected_k
        self.deploy_method = deploy_method
        self.k_vote = k_vote
        self.reduced_search = reduced_search
        self.aggressive_search = aggressive_search
        self.min_per_mode = min_per_mode
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.labels_ = None
        self.sample_indices_ = None
        self.sample_labels_ = None
        self.best_params_ = None
        self.sample_size_ = None
        self.n_trials_ = None
        self.sample_modes_ = None
        self.sample_info_ = None
        self.deploy_info_ = None
        self.learn_score_ = None

    def sample(self, X):
        """Stage 1 -- density-aware sample preserving rare density modes."""
        X = np.asarray(X, dtype=float)
        self.sample_size_ = (self.sample_size if self.sample_size is not None
                             else default_sample_size(len(X)))
        self.sample_size_ = min(self.sample_size_, len(X))
        # density_aware_sample returns (indices, mode_labels, info)
        idx, mode_labels, info = density_aware_sample(
            X, self.sample_size_, min_per_mode=self.min_per_mode,
            verbose=self.verbose)
        self.sample_indices_ = np.asarray(idx)
        self.sample_modes_ = np.asarray(mode_labels)
        self.sample_info_ = info
        return self.sample_indices_

    def learn(self, X, y):
        """Stage 2 -- tune AdaGraph on the sample, scored by SCOPE."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.sample_indices_ is None:
            self.sample(X)
        Xs = X[self.sample_indices_]
        ys = y[self.sample_indices_]
        self.n_trials_ = self.n_trials
        labels, params, score = tune_adaboxgraph_random(
            Xs, ys, n_trials=self.n_trials_, expected_k=self.expected_k,
            verbose=self.verbose, n_jobs=self.n_jobs,
            reduced_search=self.reduced_search,
            aggressive_search=self.aggressive_search)[:3]
        self.sample_labels_ = np.asarray(labels)
        self.best_params_ = params
        self.learn_score_ = float(score)
        return self

    def deploy(self, X):
        """Stages 3-4 -- classify the remaining points and deploy."""
        if self.sample_labels_ is None:
            raise RuntimeError("Call learn() before deploy().")
        X = np.asarray(X, dtype=float)

        if self.deploy_method == "adaptive":
            target_k = int(len(set(self.sample_labels_[self.sample_labels_ >= 0])))
            # adaptive_deploy returns (final_labels, info)
            final_labels, self.deploy_info_ = adaptive_deploy(
                X, self.best_params_, target_k, verbose=self.verbose)
            self.labels_ = np.asarray(final_labels)
        else:
            Xs = X[self.sample_indices_]
            # prototype_deploy returns (full_labels, info)
            full_labels, self.deploy_info_ = prototype_deploy(
                X, Xs, self.sample_labels_, k_vote=self.k_vote,
                verbose=self.verbose)
            self.labels_ = np.asarray(full_labels)
        return self.labels_

    def fit_predict(self, X, y):
        """Run the full Sample -> Learn -> Classify -> Deploy pipeline.

        ``y`` is used only on the sample, during Learn.
        """
        X = np.asarray(X, dtype=float)
        self.sample(X)
        self.learn(X, y)
        return self.deploy(X)
