"""
AdaBox three-phase tuning (the core of the algorithm's power).

This module implements the calibration that makes AdaBox more than a generic
density clusterer. Calibration runs three cascading stages on a labelled
sample:

  Stage 1 - Grid search over the AdaBox parameter grid (n_boxes, density,
            threshold factor, merge, refinement sigma), scored by SCOPE.
  Stage 2 - min_cluster_size optimization around the Stage 1 winner.
  Stage 3 - Anti-fragmentation: if the result has far more clusters than the
            ground truth, adjacent over-split clusters are merged back.

Each candidate is fit, iteratively refined, optionally smart-merged, and scored
with SCOPE. These functions are adapted from the SLCD-paper notebooks and are
free of the notebook-level globals used there.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from itertools import product

from ._engine import EADBBC2D_BR
from .scope import compute_dice_metrics


def get_tuning_score(X, y_true, y_pred, metric="SCOPE"):
    """Score a candidate clustering. SCOPE Overall by default; ARI otherwise."""
    try:
        if metric == "SCOPE":
            return compute_dice_metrics(X, y_true, y_pred, verbose=False)["Overall_Score"]
        return corrected_ari(y_true, y_pred)
    except Exception:
        return 0.0


def corrected_ari(y_true, y_pred):
    """
    Adjusted Rand Index with degenerate solution detection.
    """
    unique_pred = set(y_pred) - {-1}
    n_clusters_pred = len(unique_pred)
    unique_true = set(y_true) - {-1}
    n_clusters_true = len(unique_true)
    n_noise = int(np.sum(y_pred == -1))
    n_total = len(y_pred)
    noise_ratio = n_noise / n_total if n_total > 0 else 1.0

    if n_clusters_pred == 0:
        return 0.0
    if noise_ratio > 0.8:
        return 0.0
    if n_clusters_true > 0 and n_clusters_pred < n_clusters_true * 0.5:
        return 0.0
    return adjusted_rand_score(y_true, y_pred)

def apply_iterative_refinement(model, X, y_true, initial_labels, max_iterations=20):
    """Iteratively refine cluster boundaries until convergence."""
    best_labels = initial_labels.copy()
    best_score = get_tuning_score(X, y_true, initial_labels)
    current_labels = initial_labels.copy()

    for iteration in range(1, max_iterations + 1):
        if hasattr(model, '_refine_boundaries_with_smoothing'):
            refined_labels = model._refine_boundaries_with_smoothing(X, current_labels)
        else:
            break
        n_reassigned = int(np.sum(refined_labels != current_labels))
        if n_reassigned == 0:
            break
        score = get_tuning_score(X, y_true, refined_labels)
        if score > best_score:
            best_score = score
            best_labels = refined_labels.copy()
            current_labels = refined_labels.copy()
        else:
            break
    return best_labels

def apply_smart_merging(model, X, regular_threshold, max_extra_iterations=10):
    """Smart cluster merging with extra growth iterations."""
    if not hasattr(model, 'final_clusters_boxes_') or len(model.final_clusters_boxes_) <= 1:
        return getattr(model, 'final_clusters_boxes_', [])

    relaxed_threshold = regular_threshold * 0.5
    current_clusters = [c.copy() for c in model.final_clusters_boxes_]

    for iteration in range(max_extra_iterations):
        expanded_clusters = []
        for cluster_boxes in current_clusters:
            expanded = cluster_boxes.copy()
            offsets = [(di, dj) for di in [-1, 0, 1] for dj in [-1, 0, 1] if not (di == 0 and dj == 0)]
            candidates = set()
            for (bi, bj) in cluster_boxes:
                for di, dj in offsets:
                    neighbor = (bi + di, bj + dj)
                    if hasattr(model, 'n_boxes') and 0 <= neighbor[0] < model.n_boxes and 0 <= neighbor[1] < model.n_boxes:
                        if neighbor not in expanded:
                            candidates.add(neighbor)
            if hasattr(model, 'density_map_'):
                for candidate in candidates:
                    if model.density_map_[candidate[0], candidate[1]] >= relaxed_threshold:
                        expanded.add(candidate)
            expanded_clusters.append(expanded)

        n_clusters = len(expanded_clusters)
        parent = list(range(n_clusters))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                cluster_i = expanded_clusters[i]
                cluster_j = expanded_clusters[j]
                overlap = cluster_i & cluster_j
                if len(overlap) > 0:
                    union(i, j)
                    continue
                adjacent = False
                offsets = [(di, dj) for di in [-1, 0, 1] for dj in [-1, 0, 1] if not (di == 0 and dj == 0)]
                for (bi, bj) in cluster_i:
                    for di, dj in offsets:
                        neighbor = (bi + di, bj + dj)
                        if neighbor in cluster_j:
                            adjacent = True
                            break
                    if adjacent:
                        break
                if adjacent:
                    union(i, j)

        groups = {}
        for i in range(n_clusters):
            root = find(i)
            if root not in groups:
                groups[root] = set()
            groups[root].update(current_clusters[i])

        new_clusters = list(groups.values())
        if len(new_clusters) == len(current_clusters):
            break
        current_clusters = new_clusters

    return current_clusters


# =============================================================================
# STAGE 3: ANTI-FRAGMENTATION FUNCTIONS
# =============================================================================

def compute_cluster_precision_s3(y_true, y_pred):
    """
    Compute Cluster Precision for Stage 3: fraction of predicted cluster points correctly assigned.
    For each predicted cluster, find the best-matching true cluster.
    """
    pred_clusters = set(y_pred) - {-1}
    true_clusters = set(y_true) - {-1}
    
    if len(pred_clusters) == 0:
        return 1.0
    
    total_correctly_assigned = 0
    total_points_in_clusters = 0
    
    for pred_c in pred_clusters:
        pred_mask = (y_pred == pred_c)
        points_in_pred = np.sum(pred_mask)
        total_points_in_clusters += points_in_pred
        
        if len(true_clusters) == 0:
            continue
            
        best_match_count = 0
        for true_c in true_clusters:
            true_mask = (y_true == true_c)
            overlap = np.sum(pred_mask & true_mask)
            if overlap > best_match_count:
                best_match_count = overlap
        
        total_correctly_assigned += best_match_count
    
    if total_points_in_clusters == 0:
        return 1.0
    
    return total_correctly_assigned / total_points_in_clusters

def compute_count_accuracy_s3(y_true, y_pred):
    """
    Compute Count Accuracy for Stage 3: how close predicted cluster count is to true count.
    Count_Accuracy = 1 - |n_pred - n_true| / max(n_pred, n_true)
    """
    n_pred_clusters = len(set(y_pred) - {-1})
    n_true_clusters = len(set(y_true) - {-1})
    
    if n_pred_clusters == 0 and n_true_clusters == 0:
        return 1.0
    if n_pred_clusters == 0 or n_true_clusters == 0:
        return 0.0
    
    return 1.0 - abs(n_pred_clusters - n_true_clusters) / max(n_pred_clusters, n_true_clusters)

def compute_stage3_merge_score(y_true, y_pred):
    """
    Compute the Stage 3 optimization score.
    Uses weighted combination: 0.67 * Cluster_Precision + 0.33 * Count_Accuracy
    (Maintains 2:1 ratio from SCOPE weights: 20% vs 10%)
    """
    precision = compute_cluster_precision_s3(y_true, y_pred)
    count_acc = compute_count_accuracy_s3(y_true, y_pred)
    return 0.67 * precision + 0.33 * count_acc, precision, count_acc

def find_adjacent_cluster_pairs_s3(cluster_boxes_list, n_boxes):
    """
    Find all pairs of clusters that have adjacent boxes (8-connectivity).
    Returns list of (cluster_i, cluster_j) pairs where i < j.
    """
    adjacent_pairs = []
    n_clusters = len(cluster_boxes_list)
    
    # 8-connectivity offsets (including diagonals)
    offsets = [(di, dj) for di in [-1, 0, 1] for dj in [-1, 0, 1] if not (di == 0 and dj == 0)]
    
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            boxes_i = cluster_boxes_list[i]
            boxes_j = cluster_boxes_list[j]
            
            # Check if any box in cluster i is adjacent to any box in cluster j
            is_adjacent = False
            for (bi, bj) in boxes_i:
                for di, dj in offsets:
                    neighbor = (bi + di, bj + dj)
                    if neighbor in boxes_j:
                        is_adjacent = True
                        break
                if is_adjacent:
                    break
            
            if is_adjacent:
                adjacent_pairs.append((i, j))
    
    return adjacent_pairs

def merge_cluster_labels_s3(labels, cluster_id_a, cluster_id_b):
    """
    Merge two clusters by relabeling cluster_id_b to cluster_id_a.
    Returns new labels array with consecutive cluster IDs.
    """
    new_labels = labels.copy()
    
    # Merge b into a
    new_labels[new_labels == cluster_id_b] = cluster_id_a
    
    # Renumber to consecutive IDs
    unique_clusters = sorted(set(new_labels) - {-1})
    mapping = {old: new for new, old in enumerate(unique_clusters)}
    
    final_labels = np.full_like(new_labels, -1)
    for idx in range(len(new_labels)):
        if new_labels[idx] != -1:
            final_labels[idx] = mapping[new_labels[idx]]
    
    return final_labels

def apply_stage3_anti_fragmentation(labels, y_true, cluster_boxes_list, n_boxes, max_rounds=15):
    """
    Stage 3: Iteratively merge adjacent cluster pairs to reduce fragmentation.
    
    This is called AFTER Stage 1 and Stage 2 tuning when fragmentation is detected.
    
    Algorithm:
    1. Find all adjacent cluster pairs
    2. For each pair, test if merging improves Stage3 score
    3. Keep the best merge if it improves the score
    4. Repeat until no improvement or max_rounds reached
    
    Returns:
    --------
    labels : ndarray - Optimized cluster labels
    merge_count : int - Number of merges performed
    """
    current_labels = labels.copy()
    current_boxes = [set(boxes) for boxes in cluster_boxes_list]
    merge_count = 0
    
    # Compute initial score
    initial_score, _, _ = compute_stage3_merge_score(y_true, current_labels)
    
    # Iterative merge rounds
    for round_num in range(1, max_rounds + 1):
        # Find adjacent pairs for current clustering
        adjacent_pairs = find_adjacent_cluster_pairs_s3(current_boxes, n_boxes)
        
        if len(adjacent_pairs) == 0:
            break
        
        # Compute current score
        current_score, _, _ = compute_stage3_merge_score(y_true, current_labels)
        
        # Test each pair and find the best merge
        best_improvement = 0
        best_merge = None
        best_new_labels = None
        best_new_boxes = None
        
        for (idx_a, idx_b) in adjacent_pairs:
            # Get actual cluster IDs from labels
            cluster_ids = sorted(set(current_labels) - {-1})
            if idx_a >= len(cluster_ids) or idx_b >= len(cluster_ids):
                continue
                
            cid_a = cluster_ids[idx_a]
            cid_b = cluster_ids[idx_b]
            
            # Test merge
            test_labels = merge_cluster_labels_s3(current_labels, cid_a, cid_b)
            test_score, _, _ = compute_stage3_merge_score(y_true, test_labels)
            
            improvement = test_score - current_score
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_merge = (idx_a, idx_b, cid_a, cid_b)
                best_new_labels = test_labels
                
                # Merge boxes too
                new_boxes = []
                for i, boxes in enumerate(current_boxes):
                    if i == idx_a:
                        new_boxes.append(boxes | current_boxes[idx_b])
                    elif i == idx_b:
                        continue  # Skip merged cluster
                    else:
                        new_boxes.append(boxes.copy())
                best_new_boxes = new_boxes
        
        # Apply best merge if found
        if best_merge is not None and best_improvement > 0:
            current_labels = best_new_labels
            current_boxes = best_new_boxes
            merge_count += 1
        else:
            break
    
    return current_labels, merge_count

def evaluate_params(X, y_true, n_boxes, min_density, reg_thr, merge_adj, ref_sigma, mcs=1, 
                   use_relative_density=False, relative_density_param=5.5):
    """
    Evaluate a single parameter combination and return the score.
    Used by all tuning strategies.
    
    NOW SUPPORTS BOTH OLD AND NEW MODES:
    - OLD mode: use_relative_density=False, tunes min_density
    - NEW mode: use_relative_density=True, tunes relative_density_param
    
    Automatically handles high-dimensional data by reducing to 2D using PCA.
    Also returns the model for Stage 3 anti-fragmentation.
    """
    try:
        # EADBBC2D requires 2D input - reduce dimensionality if needed
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
        else:
            X_2d = X
        
        model = EADBBC2D_BR(
            n_boxes=n_boxes,
            min_density=min_density,
            min_growth_iterations=1,
            initial_threshold_factor=1.0,  # Use full threshold for seeding
            regular_threshold_factor=reg_thr,
            merge_adjacent=False,  # Handle manually
            refinement_sigma=ref_sigma,
            refinement_threshold=0.1,
            use_relative_density=use_relative_density,  # NEW: Support both modes
            relative_density_param=relative_density_param,  # NEW: Tune this in NEW mode
            verbose=False
        )
        model.fit(X_2d)
        labels = model.predict_labels(X_2d, min_cluster_size=1)
        labels = apply_iterative_refinement(model, X_2d, y_true, labels, max_iterations=20)
        
        if merge_adj:
            if use_relative_density:
                # NEW mode: Use relative threshold
                reg_threshold = relative_density_param * model.global_avg_density_ * reg_thr
            else:
                # OLD mode: Use absolute threshold
                reg_threshold = min_density * reg_thr
            merged_clusters = apply_smart_merging(model, X_2d, reg_threshold, max_extra_iterations=10)
            model.final_clusters_boxes_ = merged_clusters
            labels = model.predict_labels(X_2d, min_cluster_size=mcs)
        else:
            labels = model.predict_labels(X_2d, min_cluster_size=mcs)
        
        # Use original X for SCOPE metric computation (preserves full geometry)
        score = get_tuning_score(X, y_true, labels)
        
        params = {
            'n_boxes': n_boxes,
            'min_density': min_density,
            'regular_threshold_factor': reg_thr,
            'merge_adjacent': merge_adj,
            'refinement_sigma': ref_sigma,
            'min_cluster_size': mcs,
            'min_growth_iterations': 1,
            'initial_threshold_factor': 1.0,
            'refinement_threshold': 0.1,
            'dim_reduction': 'PCA' if X.shape[1] > 2 else 'None',
            'use_relative_density': use_relative_density,
            'relative_density_param': relative_density_param,
        }
        
        return score, labels, params, model
    except Exception as e:
        return 0.0, None, None, None


# ============================================================
# GRID SEARCH (Full exhaustive search - adapts to ADABOX_TEST_MODE)
# ============================================================


def slcd_calibrate_adabox(X, y_true, tuning_metric="SCOPE",
                          fragmentation_threshold=1.5, tuning_stages=3,
                          mode="NEW", search_method="RS", n_trials=100,
                          random_state=42):
    """Calibrate AdaBox via the three-phase internal optimization.

    Implements the Label stage of SLCD (paper section 4.2): Phase 1 parameter
    search (Grid Search OR Random Search), Phase 2 post-hoc min_cluster_size x
    merge refinement, Phase 3 anti-fragmentation. The ``tuning_stages`` argument
    controls how many of the three phases run (3 = all).

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Data to tune on (>2-D is reduced to 2-D with PCA inside evaluation).
    y_true : ndarray (n_samples,)
        Ground-truth labels.
    tuning_metric : {"SCOPE", "ARI"}
        Objective optimized during tuning. SCOPE is the paper default.
    fragmentation_threshold : float
        Phase 3 triggers when predicted clusters exceed
        ``n_true_clusters * fragmentation_threshold``.
    tuning_stages : int in {1, 2, 3}
        How many of the three internal phases to run.
    mode : {"NEW", "OLD"}
        "NEW" tunes ``relative_density_param`` (scale-invariant relative
        density; the paper default for SLCD). "OLD" tunes absolute
        ``min_density`` (used for small-data, no-transfer clustering).
    search_method : {"RS", "GS"}
        Phase 1 search. "RS" (Random Search) is the paper default for SLCD
        samples; "GS" (Grid Search) evaluates the full Cartesian product and is
        used for small datasets where no sampling occurs.
    n_trials : int
        Number of Random Search trials (ignored for GS).
    random_state : int
        Seed for Random Search trial sampling.

    Returns
    -------
    best_labels : ndarray
        Labels under the best parameters found.
    best_params : dict
        The selected AdaBox parameters (plus stage bookkeeping).
    """
    n_true_clusters = len(set(y_true) - {-1})

    grid = {
        "n_boxes": [15, 20, 25, 30, 35, 40, 45, 50, 60],
        "regular_threshold_factor": [0.7, 1.1],
        "merge_adjacent": [True, False],
        "refinement_sigma": [0.5, 1.0, 1.5, 2.0, 2.5],
    }
    if mode == "NEW":
        grid["min_density"] = [3]
        grid["use_relative_density"] = [True]
        grid["relative_density_param"] = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
    else:
        grid["min_density"] = [1, 2, 3, 4, 5]
        grid["use_relative_density"] = [False]
        grid["relative_density_param"] = [5.5]

    best = {"score": -1, "labels": None, "params": None, "model": None}

    # ---- Phase 1: parameter search (GS = exhaustive, RS = sampled) ----
    all_combos = list(product(
        grid["n_boxes"], grid["min_density"],
        grid["regular_threshold_factor"], grid["merge_adjacent"],
        grid["refinement_sigma"], grid["use_relative_density"],
        grid["relative_density_param"]))
    if search_method == "GS":
        candidates = all_combos
    else:
        # Random Search: sample n_trials combos uniformly WITHOUT replacement
        # from the enumerated grid (matches the original implementation, which
        # uses np.random.RandomState(seed).choice(..., replace=False)).
        rng_rs = np.random.RandomState(random_state)
        n_to_try = min(n_trials, len(all_combos))
        selected_idx = rng_rs.choice(len(all_combos), size=n_to_try, replace=False)
        candidates = [all_combos[i] for i in selected_idx]

    for (nb, md, reg, mrg, sig, ur, rp) in candidates:
        score, labels, params, model = evaluate_params(
            X, y_true, nb, md, reg, mrg, sig, mcs=1,
            use_relative_density=ur, relative_density_param=rp)
        if labels is not None and score > best["score"]:
            best.update(score=score, labels=labels, params=params, model=model)

    # ---- Stage 2: min_cluster_size optimization ----
    if best["params"] is not None and tuning_stages >= 2:
        bp = best["params"]
        for mcs in [1, 5, 10]:
            for mrg in [True, False]:
                score, labels, params, model = evaluate_params(
                    X, y_true, bp["n_boxes"], bp["min_density"],
                    bp["regular_threshold_factor"], mrg,
                    bp["refinement_sigma"], mcs,
                    bp.get("use_relative_density", False),
                    bp.get("relative_density_param", 5.5))
                if labels is not None and score > best["score"]:
                    best.update(score=score, labels=labels, params=params, model=model)

    # ---- Stage 3: anti-fragmentation ----
    if best["params"] is not None:
        best["params"]["stage3_applied"] = False
        best["params"]["stage3_merges"] = 0
        best["params"]["tuning_stages_used"] = tuning_stages
        n_pred = len(set(best["labels"]) - {-1})
        fragmented = n_pred > (n_true_clusters * fragmentation_threshold)
        model = best["model"]
        if (tuning_stages >= 3 and fragmented and model is not None
                and getattr(model, "final_clusters_boxes_", None)):
            s3_labels, merges = apply_stage3_anti_fragmentation(
                labels=best["labels"], y_true=y_true,
                cluster_boxes_list=model.final_clusters_boxes_,
                n_boxes=model.n_boxes, max_rounds=15)
            s3_score = get_tuning_score(X, y_true, s3_labels, metric=tuning_metric)
            if s3_score >= best["score"]:
                best.update(score=s3_score, labels=s3_labels)
                best["params"]["stage3_applied"] = True
                best["params"]["stage3_merges"] = merges

    return best["labels"], best["params"]


def get_slcd_tuning_score(X, y_true, y_pred, metric="SCOPE"):
    """Alias of get_tuning_score, used by the cascade retune path."""
    return get_tuning_score(X, y_true, y_pred, metric=metric)


def _perturb_adabox_params(base_params, rng, scale=0.3):
    """
    Generate a random parameter set in the neighborhood of base_params.
    
    Parameters:
    -----------
    base_params : dict - Best params from Stage 1 (Sample₁ GS)
    rng : np.random.RandomState - Random number generator
    scale : float - How far to explore (fraction of range)
    
    Returns:
    --------
    dict of perturbed parameters
    """
    # Define valid ranges for each parameter
    n_boxes_options = [15, 20, 25, 30, 35, 40, 45, 50, 60]
    reg_thr_options = [0.7, 1.1]
    ref_sigma_options = [0.5, 1.0, 1.5, 2.0, 2.5]
    mcs_options = [1, 5, 10]
    
    base_n_boxes = base_params.get('n_boxes', 30)
    
    # Perturb n_boxes: pick from nearby values
    base_idx = n_boxes_options.index(base_n_boxes) if base_n_boxes in n_boxes_options else len(n_boxes_options) // 2
    n_nearby = max(1, int(len(n_boxes_options) * scale))
    low_idx = max(0, base_idx - n_nearby)
    high_idx = min(len(n_boxes_options), base_idx + n_nearby + 1)
    n_boxes = rng.choice(n_boxes_options[low_idx:high_idx])
    
    # Perturb threshold factor: randomly pick from options
    reg_thr = rng.choice(reg_thr_options)
    
    # Perturb merge_adjacent: with 30% chance flip from base
    merge_adj = base_params.get('merge_adjacent', True)
    if rng.random() < scale:
        merge_adj = not merge_adj
    
    # Perturb refinement_sigma: pick from nearby values
    base_ref_sigma = base_params.get('refinement_sigma', 1.5)
    base_rs_idx = ref_sigma_options.index(base_ref_sigma) if base_ref_sigma in ref_sigma_options else 2
    n_nearby_rs = max(1, int(len(ref_sigma_options) * scale))
    low_rs = max(0, base_rs_idx - n_nearby_rs)
    high_rs = min(len(ref_sigma_options), base_rs_idx + n_nearby_rs + 1)
    ref_sigma = rng.choice(ref_sigma_options[low_rs:high_rs])
    
    # Perturb min_cluster_size
    mcs = rng.choice(mcs_options)
    
    # Mode-specific density parameters
    use_rel = base_params.get('use_relative_density', True)
    
    if use_rel:
        # NEW Ada: perturb relative_density_param in neighborhood
        rel_param_options = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
        base_rel = base_params.get('relative_density_param', 5.5)
        # Find nearest option
        base_rp_idx = min(range(len(rel_param_options)), key=lambda i: abs(rel_param_options[i] - base_rel))
        n_nearby_rp = max(1, int(len(rel_param_options) * scale))
        low_rp = max(0, base_rp_idx - n_nearby_rp)
        high_rp = min(len(rel_param_options), base_rp_idx + n_nearby_rp + 1)
        rel_param = rng.choice(rel_param_options[low_rp:high_rp])
        min_density = base_params.get('min_density', 3)
    else:
        # OLD Ada: perturb min_density
        md_options = [1, 2, 3, 4, 5]
        base_md = base_params.get('min_density', 2)
        base_md_idx = md_options.index(base_md) if base_md in md_options else 1
        n_nearby_md = max(1, int(len(md_options) * scale))
        low_md = max(0, base_md_idx - n_nearby_md)
        high_md = min(len(md_options), base_md_idx + n_nearby_md + 1)
        min_density = rng.choice(md_options[low_md:high_md])
        rel_param = base_params.get('relative_density_param', 5.5)
    
    return {
        'n_boxes': n_boxes,
        'min_density': min_density,
        'regular_threshold_factor': reg_thr,
        'merge_adjacent': merge_adj,
        'refinement_sigma': ref_sigma,
        'min_cluster_size': mcs,
        'use_relative_density': use_rel,
        'relative_density_param': rel_param,
    }


def slcd_retune_adabox_rs(X, y_true, base_params, tuning_metric='SCOPE',
                          n_trials=50, neighborhood_scale=0.3, 
                          fragmentation_threshold=1.5, random_state=42):
    """
    Re-tune AdaBox on Sample₂ via Random Search around Stage 1 best params.
    
    Parameters:
    -----------
    X, y_true : Sample₂ data and labels
    base_params : dict - Best params from Stage 1 (calibrated on Sample₁)
    tuning_metric : str - 'SCOPE' or 'ARI'
    n_trials : int - Number of random search trials
    neighborhood_scale : float - How far to explore from base_params
    fragmentation_threshold : float - For Stage 3 anti-fragmentation
    random_state : int
    
    Returns:
    --------
    best_labels, best_params, retune_time
    """
    rng = np.random.RandomState(random_state + 1000)  # Different seed from Stage 1
    
    n_true_clusters = len(set(y_true) - {-1})
    
    # Start with base_params as current best (evaluate on Sample₂)
    base_score_s2, base_labels_s2, _, base_model_s2 = evaluate_params(
        X, y_true,
        base_params.get('n_boxes', 30),
        base_params.get('min_density', 3),
        base_params.get('regular_threshold_factor', 0.7),
        base_params.get('merge_adjacent', True),
        base_params.get('refinement_sigma', 1.5),
        base_params.get('min_cluster_size', 1),
        use_relative_density=base_params.get('use_relative_density', True),
        relative_density_param=base_params.get('relative_density_param', 5.5)
    )
    
    best_score = base_score_s2 if base_score_s2 is not None else -1
    best_labels = base_labels_s2
    best_params = base_params.copy()
    best_model = base_model_s2
    
    # Random Search: try n_trials random perturbations
    for trial in range(n_trials):
        trial_params = _perturb_adabox_params(base_params, rng, scale=neighborhood_scale)
        
        try:
            score, labels, params, model = evaluate_params(
                X, y_true,
                trial_params['n_boxes'],
                trial_params['min_density'],
                trial_params['regular_threshold_factor'],
                trial_params['merge_adjacent'],
                trial_params['refinement_sigma'],
                trial_params['min_cluster_size'],
                use_relative_density=trial_params['use_relative_density'],
                relative_density_param=trial_params['relative_density_param']
            )
            
            if score is not None and score > best_score:
                best_score = score
                best_labels = labels
                best_params = params
                best_model = model
        except Exception:
            continue
    
    # Stage 3: Anti-Fragmentation on best RS result
    if best_params is not None and best_labels is not None:
        best_params['stage3_applied'] = False
        best_params['stage3_merges'] = 0
        best_params['tuning_stages_used'] = '3 (RS re-tune)'
        
        n_pred_clusters = len(set(best_labels) - {-1})
        fragmented = n_pred_clusters > (n_true_clusters * fragmentation_threshold)
        
        if fragmented and best_model is not None and hasattr(best_model, 'final_clusters_boxes_') and len(best_model.final_clusters_boxes_) > 0:
            stage3_labels, merge_count = apply_stage3_anti_fragmentation(
                labels=best_labels,
                y_true=y_true,
                cluster_boxes_list=best_model.final_clusters_boxes_,
                n_boxes=best_model.n_boxes,
                max_rounds=15
            )
            
            stage3_score = get_slcd_tuning_score(X, y_true, stage3_labels, metric=tuning_metric)
            
            if stage3_score >= best_score:
                best_labels = stage3_labels
                best_params['stage3_applied'] = True
                best_params['stage3_merges'] = merge_count
    
    return best_labels, best_params


