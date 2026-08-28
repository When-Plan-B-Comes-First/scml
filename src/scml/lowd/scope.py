"""
SCOPE: Structured Clustering Optimization via Performance Evaluation.

SCOPE is a structure-aware *optimization objective* for clustering, not only a
report-time score. It decomposes clustering quality into five interpretable
components (core purity, boundary recall, cluster precision, noise F1,
cluster-count accuracy) and combines them into a single value in [0, 1].

Because it rewards correct recovery of cluster *structure* rather than exact
label matching, SCOPE gives a smoother, more informative signal than ARI when
used to drive hyperparameter search -- which is how AdaBox is tuned. Its value
is therefore twofold: a better metric to report, and a better objective to
optimize against.
"""
import numpy as np

# ======================================================
# SCOPE: Structured Clustering Optimization via Performance Evaluation
# ======================================================
# Complete implementation with all 5 components
# Version: 5.0 (k-NN Density-based core identification)
# ======================================================

def compute_dice_metrics(X, y_true, y_pred, alpha=0.25, verbose=False):
    """
    Compute SCOPE clustering quality metrics.
    
    SCOPE decomposes clustering quality into 5 interpretable components:
    1. Core Purity (25%): Fraction of cores assigned to CORRECT matched cluster (Hungarian)
    2. Boundary Recall (25%): Fraction of boundaries assigned to CORRECT matched cluster
    3. Cluster Precision (20%): Weighted purity of predicted clusters (micro-averaged)
    4. Noise F1 (20%): Harmonic mean of noise precision/recall
    5. Cluster Count Accuracy (10%): Penalty for wrong number of clusters
    
    Core/Boundary identification uses k-NN local density (Definition 3.1-3.2):
    - k = ln(n) for robustness
    - ρ(x) = 1 / d_k(x) where d_k is distance to k-th nearest neighbor
    - Core = points with ρ(x) ≥ Q_{1-α}(cluster densities) = densest α fraction
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    y_true : array-like, shape (n_samples,)
        Ground truth labels (-1 for noise)
    y_pred : array-like, shape (n_samples,)
        Predicted labels (-1 for noise)
    alpha : float
        Core percentile threshold (default: 0.25 = densest 25% are cores)
    verbose : bool
        Print detailed component scores
        
    Returns:
    --------
    dict : Dictionary containing:
        - Core_Purity: float in [0, 1]
        - Boundary_Recall: float in [0, 1]
        - Cluster_Precision: float in [0, 1]
        - Noise_F1: float in [0, 1]
        - Cluster_Count_Accuracy: float in [0, 1]
        - Overall_Score: Weighted average of all components
        - N_True_Clusters: int
        - N_Pred_Clusters: int
    """
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import f1_score
    from sklearn.neighbors import NearestNeighbors
    
    n_points = len(X)
    
    # ============================================================
    # STEP 0: Compute Global k-NN Distances for Local Density
    # ============================================================
    # k = ln(n) as per Definition 3.1
    k = max(2, int(np.log(n_points)))
    
    # Compute k-NN distances for all points
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    distances_knn, _ = nbrs.kneighbors(X)
    
    # d_k(x) = distance to k-th nearest neighbor (index k, since index 0 is self)
    d_k = distances_knn[:, k]
    
    # Local density: ρ(x) = 1 / d_k(x)
    # Add small epsilon to avoid division by zero
    local_density = 1.0 / (d_k + 1e-10)
    
    # ============================================================
    # COMPONENT 1: Core Purity (25% weight)
    # ============================================================
    # Identify cores using k-NN local density (Definition 3.2)
    # Core = points with density ≥ Q_{1-α} within their cluster
    # This is robust to non-convex shapes (moons, rings, spirals)
    
    cores_true = np.zeros(n_points, dtype=bool)
    
    for cluster_id in set(y_true):
        if cluster_id == -1:  # Skip noise
            continue
        
        cluster_mask = (y_true == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        
        # Edge case: very small clusters (all points are cores)
        if len(cluster_indices) < 2:
            cores_true[cluster_indices] = True
            continue
        
        # Get local densities for this cluster
        cluster_densities = local_density[cluster_indices]
        
        # Core = densest α fraction (density ≥ Q_{1-α})
        # Q_{1-α} means the (1-α) quantile, so top α% have density above this
        density_threshold = np.percentile(cluster_densities, (1 - alpha) * 100)
        is_core_local = cluster_densities >= density_threshold
        
        cores_true[cluster_indices[is_core_local]] = True
    
    # Calculate Core Purity using Hungarian algorithm for optimal cluster matching
    core_purity = 0.0
    cluster_mapping = {}  # Store optimal true→pred cluster mapping for later use
    
    if cores_true.sum() > 0:
        true_core_labels = y_true[cores_true]
        pred_core_labels = y_pred[cores_true]
        
        true_clusters = set(true_core_labels) - {-1}
        pred_clusters = set(pred_core_labels) - {-1}
        
        if len(true_clusters) > 0 and len(pred_clusters) > 0:
            # Build cost matrix: negative overlap counts for maximization
            cost_matrix = np.zeros((len(true_clusters), len(pred_clusters)))
            true_cluster_list = sorted(true_clusters)
            pred_cluster_list = sorted(pred_clusters)
            
            for i, tc in enumerate(true_cluster_list):
                for j, pc in enumerate(pred_cluster_list):
                    overlap = np.sum((true_core_labels == tc) & (pred_core_labels == pc))
                    cost_matrix[i, j] = -overlap  # Negative for maximization
            
            # Optimal assignment using Hungarian algorithm (maximize core overlap)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_matched = -cost_matrix[row_ind, col_ind].sum()
            core_purity = total_matched / cores_true.sum()
            
            # Store the optimal mapping: true_cluster → pred_cluster
            for i, j in zip(row_ind, col_ind):
                cluster_mapping[true_cluster_list[i]] = pred_cluster_list[j]
        elif len(true_clusters) == 0:
            core_purity = 1.0  # No cores to match
        else:
            core_purity = 0.0  # Algorithm found no clusters
    else:
        core_purity = 1.0  # Perfect (no cores in ground truth)
    
    # ============================================================
    # COMPONENT 2: Boundary Recall (25% weight)
    # ============================================================
    # Measure how many boundary points are assigned to their CORRECT matched cluster
    # (not just any cluster, but specifically the one matched via Hungarian algorithm)
    # This prevents random labeling from getting artificially high scores
    
    boundaries_true = (y_true != -1) & (~cores_true)  # Non-core clustered points
    
    boundary_recall = 0.0
    
    if boundaries_true.sum() > 0:
        # Check if boundary points are assigned to their CORRECT matched cluster
        correctly_assigned_boundaries = 0
        
        for idx in np.where(boundaries_true)[0]:
            true_cluster = y_true[idx]
            pred_cluster = y_pred[idx]
            
            # Check if predicted cluster matches the optimal mapping from Hungarian
            if true_cluster in cluster_mapping:
                expected_pred = cluster_mapping[true_cluster]
                if pred_cluster == expected_pred:
                    correctly_assigned_boundaries += 1
        
        boundary_recall = correctly_assigned_boundaries / boundaries_true.sum()
    else:
        boundary_recall = 1.0  # Perfect (no boundaries to capture)
    
    # ============================================================
    # COMPONENT 3: Cluster Precision (20% weight)
    # ============================================================
    # Measure weighted purity: for each predicted cluster, what fraction
    # belongs to its dominant true cluster (micro-averaged across all points)
    # This prevents random scattered assignments from getting high scores
    
    cluster_precision = 0.0
    
    pred_clusters = set(y_pred) - {-1}
    
    if len(pred_clusters) > 0:
        total_points_in_clusters = 0
        total_correctly_assigned = 0
        
        for pred_cluster_id in pred_clusters:
            # Get all points in this predicted cluster
            pred_cluster_mask = (y_pred == pred_cluster_id)
            pred_cluster_size = pred_cluster_mask.sum()
            
            # Find the dominant true cluster (majority vote)
            true_labels_in_pred = y_true[pred_cluster_mask]
            true_labels_in_pred_no_noise = true_labels_in_pred[true_labels_in_pred != -1]
            
            if len(true_labels_in_pred_no_noise) > 0:
                # Count points belonging to the dominant true cluster
                unique, counts = np.unique(true_labels_in_pred_no_noise, return_counts=True)
                dominant_true_cluster = unique[np.argmax(counts)]
                n_dominant = np.max(counts)
                
                total_correctly_assigned += n_dominant
                total_points_in_clusters += pred_cluster_size
            else:
                # All points in this predicted cluster are noise in ground truth
                total_points_in_clusters += pred_cluster_size
        
        if total_points_in_clusters > 0:
            cluster_precision = total_correctly_assigned / total_points_in_clusters
        else:
            cluster_precision = 1.0
    else:
        cluster_precision = 1.0  # Perfect (no clusters predicted, so no impurity)
    
    # ============================================================
    # COMPONENT 4: Noise F1 Score (20% weight)
    # ============================================================
    # Measure noise detection accuracy using F1 score
    
    true_noise = (y_true == -1).astype(int)
    pred_noise = (y_pred == -1).astype(int)
    
    if true_noise.sum() > 0 or pred_noise.sum() > 0:
        noise_f1 = f1_score(true_noise, pred_noise, zero_division=0.0)
    else:
        noise_f1 = 1.0  # Perfect (no noise in ground truth or predictions)
    
    # ============================================================
    # COMPONENT 5: Cluster Count Accuracy (10% weight)
    # ============================================================
    # Penalize over-fragmentation or under-clustering
    
    n_true_clusters = len(set(y_true) - {-1})
    n_pred_clusters = len(set(y_pred) - {-1})
    
    if n_true_clusters == 0 and n_pred_clusters == 0:
        cluster_count_accuracy = 1.0  # Both found no clusters (edge case)
    elif n_true_clusters == 0 or n_pred_clusters == 0:
        cluster_count_accuracy = 0.0  # One found 0, other found >0 (mismatch)
    else:
        # Proportional penalty based on mismatch
        cluster_count_accuracy = 1.0 - abs(n_pred_clusters - n_true_clusters) / max(n_pred_clusters, n_true_clusters)
    
    # ============================================================
    # OVERALL SCOPE SCORE (Weighted Average)
    # ============================================================
    overall_score = (
        0.25 * core_purity +
        0.25 * boundary_recall +
        0.20 * cluster_precision +
        0.20 * noise_f1 +
        0.10 * cluster_count_accuracy
    )
    
    # ============================================================
    # Verbose Output (Optional)
    # ============================================================
    if verbose:
        print("=" * 60)
        print("SCOPE Component Scores:")
        print("-" * 60)
        print(f"  Core Purity (25%):           {core_purity:.3f}")
        print(f"  Boundary Recall (25%):       {boundary_recall:.3f}")
        print(f"  Cluster Precision (20%):     {cluster_precision:.3f}")
        print(f"  Noise F1 (20%):              {noise_f1:.3f}")
        print(f"  Cluster Count Accuracy (10%): {cluster_count_accuracy:.3f}")
        print("-" * 60)
        print(f"  Overall SCOPE Score:          {overall_score:.3f}")
        print("=" * 60)
        print(f"  True Clusters: {n_true_clusters}, Predicted: {n_pred_clusters}")
        print("=" * 60)
    
    # ============================================================
    # Return Results Dictionary
    # ============================================================
    return {
        'Core_Purity': core_purity,
        'Boundary_Recall': boundary_recall,
        'Cluster_Precision': cluster_precision,
        'Noise_F1': noise_f1,
        'Cluster_Count_Accuracy': cluster_count_accuracy,
        'Overall_Score': overall_score,
        'N_True_Clusters': n_true_clusters,
        'N_Pred_Clusters': n_pred_clusters
    }


# ======================================================
# INITIALIZATION CONFIRMATION
# ======================================================
