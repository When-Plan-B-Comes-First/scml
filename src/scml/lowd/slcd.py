"""
SLCD (low-D track): Sample -> Label -> Calibrate -> Deploy.

SLCD is the workflow that delivers AdaBox's full power at scale. It draws a
small, structure-preserving *sample* from a large dataset, *calibrates* AdaBox
on that sample with the three cascading tuning stages (grid search ->
min_cluster_size -> anti-fragmentation), then *deploys* the calibrated
parameters to the full dataset.

Calibration is where the algorithm's strength lives: an untuned AdaBox is just
another density clusterer, but the three-phase calibration is what makes it
competitive with and often better than DBSCAN/HDBSCAN.

SLCD names a family, not a single algorithm. Each track instantiates it with
its own mechanism, and the two are not interchangeable:

  * Low-D  (this module): Sample -> Label -> Calibrate -> Deploy.
    "Deploy" means true *parameter transfer* -- the parameters tuned on the
    sample are used to cluster the full dataset.
  * High-D (scml.highd):  Sample -> Learn -> Classify -> Deploy.
    "Deploy" means AdaGraph clusters the sample, then the remaining points are
    assigned to those clusters by a voting mechanism.

What the two share is the invariant that gives SLCD its value: the full dataset
is never tuned.
"""

import numpy as np

from ._engine import EADBBC2D_BR
from ._tuning import slcd_calibrate_adabox, evaluate_params, corrected_ari
from .metrics import scope_score


def stratified_sample(X, y, n_samples=500, random_state=42):
    """
    Create a stratified sample preserving cluster proportions.
    """
    n_total = len(X)
    
    if n_total <= n_samples:
        return X.copy(), y.copy(), np.arange(n_total)
    
    sample_ratio = n_samples / n_total
    
    # Handle noise points separately
    noise_mask = (y == -1)
    cluster_mask = ~noise_mask
    
    unique_clusters = np.unique(y[cluster_mask])
    cluster_sizes = {c: np.sum(y == c) for c in unique_clusters}
    n_noise = np.sum(noise_mask)
    
    # Allocate samples proportionally
    samples_per_cluster = {}
    total_cluster_points = sum(cluster_sizes.values())
    n_cluster_samples = int(n_samples * (total_cluster_points / n_total))
    n_noise_samples = n_samples - n_cluster_samples
    
    # Ensure at least 2 points per cluster
    remaining = n_cluster_samples
    for c in unique_clusters:
        proportion = cluster_sizes[c] / total_cluster_points
        samples_per_cluster[c] = max(2, int(n_cluster_samples * proportion))
        remaining -= samples_per_cluster[c]
    
    # Distribute remaining samples
    if remaining > 0:
        sorted_clusters = sorted(unique_clusters, key=lambda c: cluster_sizes[c], reverse=True)
        for i, c in enumerate(sorted_clusters):
            if remaining <= 0:
                break
            samples_per_cluster[c] += 1
            remaining -= 1
    
    # Sample from each cluster
    rng = np.random.RandomState(random_state)
    sample_indices = []
    
    for c in unique_clusters:
        cluster_indices = np.where(y == c)[0]
        n_to_sample = min(samples_per_cluster[c], len(cluster_indices))
        sampled = rng.choice(cluster_indices, size=n_to_sample, replace=False)
        sample_indices.extend(sampled)
    
    # Sample noise points
    if n_noise > 0 and n_noise_samples > 0:
        noise_indices = np.where(noise_mask)[0]
        n_noise_to_sample = min(n_noise_samples, len(noise_indices))
        sampled_noise = rng.choice(noise_indices, size=n_noise_to_sample, replace=False)
        sample_indices.extend(sampled_noise)
    
    sample_indices = np.array(sample_indices)
    rng.shuffle(sample_indices)
    
    X_sample = X[sample_indices]
    y_sample = y[sample_indices]
    
    return X_sample, y_sample, sample_indices



def stratified_sample_excluding(X, y, n_samples, exclude_indices, random_state=42):
    """
    Create a stratified sample from X, EXCLUDING specified indices.
    Used in Protocol B to draw Sample₂ independently of Sample₁.
    
    Parameters:
    -----------
    X : array - Full dataset
    y : array - Full labels
    n_samples : int - Number of points to sample
    exclude_indices : array - Indices to exclude (e.g., Sample₁ indices)
    random_state : int - Random seed
    
    Returns:
    --------
    X_sample, y_sample, sample_indices (relative to full dataset)
    """
    # Create mask for available points
    all_indices = np.arange(len(X))
    exclude_set = set(exclude_indices)
    available_mask = np.array([i not in exclude_set for i in all_indices])
    available_indices = all_indices[available_mask]
    
    n_available = len(available_indices)
    
    if n_available <= n_samples:
        return X[available_indices].copy(), y[available_indices].copy(), available_indices
    
    # Get labels for available points
    y_available = y[available_indices]
    
    # Handle noise points separately
    noise_mask = (y_available == -1)
    cluster_mask = ~noise_mask
    
    unique_clusters = np.unique(y_available[cluster_mask])
    cluster_sizes = {c: np.sum(y_available == c) for c in unique_clusters}
    n_noise = np.sum(noise_mask)
    
    # Allocate samples proportionally
    total_cluster_points = sum(cluster_sizes.values())
    n_cluster_samples = int(n_samples * (total_cluster_points / n_available))
    n_noise_samples = n_samples - n_cluster_samples
    
    samples_per_cluster = {}
    for c in unique_clusters:
        proportion = cluster_sizes[c] / total_cluster_points
        samples_per_cluster[c] = max(2, int(n_cluster_samples * proportion))
    
    # Distribute remaining
    remaining = n_cluster_samples - sum(samples_per_cluster.values())
    if remaining > 0:
        sorted_clusters = sorted(unique_clusters, key=lambda c: cluster_sizes[c], reverse=True)
        for c in sorted_clusters:
            if remaining <= 0:
                break
            samples_per_cluster[c] += 1
            remaining -= 1
    
    # Sample from each cluster
    rng = np.random.RandomState(random_state)
    sample_indices_local = []
    
    for c in unique_clusters:
        cluster_local_indices = np.where(y_available == c)[0]
        n_to_sample = min(samples_per_cluster[c], len(cluster_local_indices))
        sampled = rng.choice(cluster_local_indices, size=n_to_sample, replace=False)
        sample_indices_local.extend(sampled)
    
    # Sample noise points
    if n_noise > 0 and n_noise_samples > 0:
        noise_local_indices = np.where(noise_mask)[0]
        n_noise_to_sample = min(n_noise_samples, len(noise_local_indices))
        sampled_noise = rng.choice(noise_local_indices, size=n_noise_to_sample, replace=False)
        sample_indices_local.extend(sampled_noise)
    
    sample_indices_local = np.array(sample_indices_local)
    rng.shuffle(sample_indices_local)
    
    # Convert local indices back to global indices
    sample_indices_global = available_indices[sample_indices_local]
    
    X_sample = X[sample_indices_global]
    y_sample = y[sample_indices_global]
    
    return X_sample, y_sample, sample_indices_global


def default_sample_size(n_points, n_clusters=None):
    """Size-adaptive calibration sample size, aware of cluster count.

    Two forces set the sample size:

    1. **Dataset size** -- the sample grows with deployment scale so coverage
       stays adequate (the paper scales sample size to hold the coverage ratio
       roughly constant from standard scale up to 100M):

           < 50,000   -> 200      < 500,000  -> 800
           < 2,000,000-> 1,500    >= 2,000,000 -> 5,000

    2. **Cluster count** -- small samples cannot represent many clusters, so the
       sample must give enough points per cluster (the paper notes >15 clusters
       need larger samples; experiments show ~80 points/cluster transfers well).
       When ``n_clusters`` is known, the floor ``80 * n_clusters`` is enforced
       for datasets with more than 5 clusters.

    The returned size is ``max(size_tier, cluster_floor)``, capped so it never
    exceeds the dataset itself.

    Parameters
    ----------
    n_points : int
        Size of the full dataset.
    n_clusters : int, optional
        Number of clusters (e.g. distinct non-noise labels). If given and > 5,
        raises the sample size to ensure adequate per-cluster representation.
    """
    if n_points < 50_000:
        size = 200
    elif n_points < 500_000:
        size = 800
    elif n_points < 2_000_000:
        size = 1_500
    else:
        size = 5_000

    if n_clusters is not None and n_clusters > 5:
        size = max(size, 80 * int(n_clusters))

    return int(min(size, n_points))


def _fit_with_params(X, params):
    """Fit an AdaBox engine on X using a calibrated parameter dict and return
    (model, labels). Mirrors the deployment path used during calibration."""
    from .scope import compute_dice_metrics  # noqa: F401 (kept for parity)
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        X2d = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        X2d = X
    model = EADBBC2D_BR(
        n_boxes=params["n_boxes"],
        min_density=params["min_density"],
        min_growth_iterations=params.get("min_growth_iterations", 1),
        initial_threshold_factor=params.get("initial_threshold_factor", 1.0),
        regular_threshold_factor=params["regular_threshold_factor"],
        merge_adjacent=params.get("merge_adjacent", True),
        refinement_sigma=params["refinement_sigma"],
        refinement_threshold=params.get("refinement_threshold", 0.1),
        use_relative_density=params.get("use_relative_density", False),
        relative_density_param=params.get("relative_density_param", 5.5),
        verbose=False,
    )
    model.fit(X2d)
    labels = model.predict_labels(X2d, min_cluster_size=params.get("min_cluster_size", 1))
    return model, labels


class SLCD:
    """Sample -> Label -> Calibrate -> Deploy workflow for AdaBox (low-D).

    SLCD tunes AdaBox on a small stratified sample with **Random Search** and
    **relative (scale-invariant) density** -- the configuration that makes
    sample-tuned parameters transfer to the full dataset -- optionally validates
    them through a cascade of larger, non-overlapping samples (retuning only on
    detected drift), then deploys the frozen parameters. Matches the SLCD paper.

    Parameters
    ----------
    sample_size : int, optional
        Size of the Stage 1 calibration sample. If None, chosen by dataset size
        (see :func:`default_sample_size`): 200 (<50k), 800 (<500k), 1,500
        (<2M), 5,000 (>=2M). The sample grows with the data so coverage stays
        adequate at extreme scale.
    n_trials : int, optional
        Random Search trials in the Label stage. If None: 100 for < 20,000
        points, 200 for > 20,000 (author defaults).
    cascade_stages : int, optional
        Number of cascade (Calibrate) stages, including Stage 1. If None: 1 for
        < 20,000 points, 2 for > 20,000 (author defaults; the paper recommends
        stopping at 2).
    tuning_metric : {"SCOPE", "ARI"}, default="SCOPE"
        Objective used during calibration.
    fragmentation_threshold : float, default=1.1
        Anti-fragmentation triggers above n_true_clusters * this factor
        (SLCD-era default; the earlier benchmark code used 1.5).
    degradation_threshold : float, default=0.01
        A cascade stage retunes (neighbourhood RS) if deploying the current
        parameters to the new sample loses more than this in ARI, or if the
        deployed ARI falls below the 0.4 quality floor (paper trigger rule).
    retune_trials : int, default=100
        Neighbourhood Random Search trials when a cascade stage retunes.
    neighborhood_scale : float, default=0.3
        How far neighbourhood RS explores around the current best (paper value).
    random_state : int, default=42
        Seed for sampling and Random Search.

    Attributes
    ----------
    best_params_ : dict
        Calibrated AdaBox parameters (with stage bookkeeping).
    calibration_score_ : float
        SCOPE score on the final calibration sample.
    sample_size_, n_trials_, cascade_stages_ : int
        Values actually used after size-based defaulting.
    stage_history_ : list of dict
        Per-stage record: sample size, score, and whether a retune fired.
    labels_ : ndarray
        Labels from deploying the calibrated parameters on the full data.
    """

    def __init__(self, sample_size=None, n_trials=None, cascade_stages=None,
                 tuning_metric="SCOPE", fragmentation_threshold=1.1,
                 degradation_threshold=0.01, retune_trials=100,
                 neighborhood_scale=0.3, random_state=42):
        self.sample_size = sample_size
        self.n_trials = n_trials
        self.cascade_stages = cascade_stages
        self.tuning_metric = tuning_metric
        self.fragmentation_threshold = fragmentation_threshold
        self.degradation_threshold = degradation_threshold
        self.retune_trials = retune_trials
        self.neighborhood_scale = neighborhood_scale
        self.random_state = random_state

        self.best_params_ = None
        self.calibration_score_ = None
        self.sample_size_ = None
        self.n_trials_ = None
        self.cascade_stages_ = None
        self.stage_history_ = None
        self.labels_ = None
        self._model = None

    def calibrate(self, X, y):
        """Run the Label + Calibrate cascade and select transfer parameters.

        Stage 1 tunes on a stratified sample (Random Search, relative density).
        Each later stage draws a fresh, non-overlapping sample twice the size,
        deploys the current parameters to it, and retunes via neighbourhood
        Random Search only if SCOPE drops by more than ``degradation_threshold``.
        """
        from ._tuning import slcd_retune_adabox_rs

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        big = len(X) > 20000
        n_clusters = len(set(np.asarray(y).tolist()) - {-1})
        self.sample_size_ = (self.sample_size if self.sample_size is not None
                             else default_sample_size(len(X), n_clusters))
        self.n_trials_ = (self.n_trials if self.n_trials is not None
                          else (200 if big else 100))
        self.cascade_stages_ = (self.cascade_stages if self.cascade_stages is not None
                                else (2 if big else 1))

        # ---- Stage 1: Label on the first stratified sample ----
        Xs, ys, used_idx = stratified_sample(X, y, self.sample_size_, self.random_state)
        labels, params = slcd_calibrate_adabox(
            Xs, ys, tuning_metric=self.tuning_metric,
            fragmentation_threshold=self.fragmentation_threshold,
            tuning_stages=3, mode="NEW", search_method="RS",
            n_trials=self.n_trials_, random_state=self.random_state)
        if params is None:
            raise RuntimeError("Calibration failed to find valid parameters.")

        cur_score = float(scope_score(Xs, ys, labels))
        cur_ari = float(corrected_ari(ys, labels))
        used_idx = list(used_idx)
        history = [{"stage": 1, "sample_size": len(Xs), "score": cur_score,
                    "ari": cur_ari, "retuned": False}]

        # ---- Stages 2..k: Calibrate (validate, retune only on drift) ----
        # Drift detection follows the paper/notebook exactly: degradation is
        # measured in ARI; retune triggers if ARI drop > degradation_threshold
        # (default 0.01) OR the deployed ARI falls below the 0.4 floor.
        stage_size = self.sample_size_
        for stage in range(2, self.cascade_stages_ + 1):
            stage_size *= 2
            Xc, yc, idx = stratified_sample_excluding(
                X, y, stage_size, used_idx, self.random_state + stage)
            if len(Xc) < 10:
                break
            used_idx += list(idx)

            # deploy current params to the new sample; measure degradation (ARI)
            _, deployed_labels = _fit_with_params(Xc, params)
            deployed_score = float(scope_score(Xc, yc, deployed_labels))
            deployed_ari = float(corrected_ari(yc, deployed_labels))
            degradation = cur_ari - deployed_ari
            retuned = False
            if (degradation > self.degradation_threshold) or (deployed_ari < 0.4):
                new_labels, new_params = slcd_retune_adabox_rs(
                    Xc, yc, params, tuning_metric=self.tuning_metric,
                    n_trials=self.retune_trials,
                    neighborhood_scale=self.neighborhood_scale,
                    fragmentation_threshold=self.fragmentation_threshold,
                    random_state=self.random_state + stage)
                if new_params is not None and new_labels is not None:
                    new_ari = float(corrected_ari(yc, new_labels))
                    # accept retune only if ARI strictly improves (paper rule)
                    if new_ari > deployed_ari:
                        params, retuned = new_params, True
                        cur_score = float(scope_score(Xc, yc, new_labels))
                        cur_ari = new_ari
                    else:
                        cur_score, cur_ari = deployed_score, deployed_ari
                else:
                    cur_score, cur_ari = deployed_score, deployed_ari
            else:
                cur_score, cur_ari = deployed_score, deployed_ari
            history.append({"stage": stage, "sample_size": len(Xc),
                            "score": cur_score, "ari": cur_ari,
                            "retuned": retuned})

        self.best_params_ = params
        self.calibration_score_ = cur_score
        self.stage_history_ = history
        return self

    def deploy(self, X_full):
        """Deploy the calibrated parameters to the full dataset."""
        if self.best_params_ is None:
            raise RuntimeError("Call calibrate() before deploy().")
        X_full = np.asarray(X_full, dtype=float)
        self._model, self.labels_ = _fit_with_params(X_full, self.best_params_)
        return self.labels_

    def fit_predict(self, X, y):
        """Run the full Sample -> Label -> Calibrate -> Deploy pipeline.

        Returns cluster labels for the full dataset ``X``. ``y`` is used only on
        the calibration samples.
        """
        self.calibrate(X, y)
        return self.deploy(X)
