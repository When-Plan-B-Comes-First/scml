"""
AdaGraph core engine (high-D track).

Expansion-Adaptive Density-Based Clustering via Graph Neighborhoods in Native
Dimensions. Clusters directly in the original high-dimensional space by
building a kNN neighborhood graph rather than reducing to 2-D first.

Lifted from the author's validated implementation. The internal class name
AdaBoxGraph is retained; the public estimator is scml.highd.AdaGraph.
"""

# ======================================================
# AdaHD Graph Engine -- Strategy C (OPTIMIZED)
# ======================================================
# Key optimizations over v1:
#   1. Precompute kNN at max-k once, reuse across trials
#   2. Sparse adjacency matrix (scipy CSR) for fast lookups
#   3. Numpy boolean masks for growth (no Python set loops)
#   4. Vectorized boundary detection for merging
#   5. kNN-vote refinement (no per-cluster sklearn models)
# ======================================================

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
from ..lowd.scope import compute_dice_metrics


# ======================================================
# PRECOMPUTE kNN (call once, reuse across all trials)
# ======================================================

def precompute_knn(X, max_k=30):
    """
    Precompute kNN distances and indices at max_k.
    Call ONCE before tuning loop. Each trial slices to its own k.

    Returns: (distances, indices) — both shape (n, max_k), self excluded.
    """
    n = len(X)
    k = min(max_k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
    dists, indices = nbrs.kneighbors(X)
    return dists[:, 1:], indices[:, 1:]  # skip self (column 0)


# ======================================================
# AdaBoxGraph — Optimized
# ======================================================

class AdaBoxGraph:
    """
    AdaBox on a kNN graph. Optimized with sparse matrices + numpy vectorization.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbors for graph. Default: 15.
    min_density : float
        Density threshold factor. Default: 3.0.
        Threshold = mean_density * (min_density / 3.0).
    regular_threshold_factor : float
        Growth threshold multiplier. Default: 1.0.
    merge_adjacent : bool
        Enable statistical merging. Default: True.
    refinement_sigma : float
        Controls refinement aggressiveness. Default: 1.5.
    min_cluster_size : int
        Minimum cluster size. Default: 5.
    """

    def __init__(self, k_neighbors=15, min_density=3.0,
                 regular_threshold_factor=1.0, merge_adjacent=True,
                 refinement_sigma=1.5, min_cluster_size=5,
                 min_growth_iterations=1,
                 centroid_distance_threshold=2.0, ks_test_alpha=0.05,
                 seed_exclusion_hops=1, use_mutual_knn=False,
                 use_shared_neighbor_density=False,
                 ks_merge_fraction=0.5,
                 k_neighbors_frac=None, min_cluster_size_frac=None,
                 verbose=False):
        self.k_neighbors = int(k_neighbors)
        self.min_density = float(min_density)
        self.regular_threshold_factor = float(regular_threshold_factor)
        self.merge_adjacent = bool(merge_adjacent)
        self.refinement_sigma = float(refinement_sigma)
        self.min_cluster_size = int(min_cluster_size)
        self.min_growth_iterations = int(min_growth_iterations)
        self.centroid_distance_threshold = float(centroid_distance_threshold)
        self.ks_test_alpha = float(ks_test_alpha)
        self.seed_exclusion_hops = int(seed_exclusion_hops)
        self.use_mutual_knn = bool(use_mutual_knn)
        self.use_shared_neighbor_density = bool(use_shared_neighbor_density)
        self.ks_merge_fraction = float(ks_merge_fraction)
        # Scale-invariant fractional params (override counts if set)
        self.k_neighbors_frac = k_neighbors_frac
        self.min_cluster_size_frac = min_cluster_size_frac
        self.verbose = verbose

    def fit(self, X, precomputed_knn=None):
        """
        Fit the model. Optionally accept precomputed kNN to avoid recomputation.

        Parameters
        ----------
        X : array (n, d)
        precomputed_knn : tuple (dists, indices) from precompute_knn(), optional
        """
        self.X_ = X
        n, d = X.shape

        # -- Resolve scale-invariant fractions to counts --
        if self.k_neighbors_frac is not None:
            self.k_neighbors = max(3, min(round(self.k_neighbors_frac * np.log2(n)), n - 1))
        if self.min_cluster_size_frac is not None:
            self.min_cluster_size = max(3, round(self.min_cluster_size_frac * np.sqrt(n)))

        if self.verbose:
            print(f"  AdaBoxGraph: n={n}, d={d}, k={self.k_neighbors}"
                  f"{' (frac='+str(self.k_neighbors_frac)+')' if self.k_neighbors_frac else ''}")

        # -- Stage 1: Build kNN graph --
        self._build_graph(X, precomputed_knn)

        if self.verbose:
            n_edges = self.adj_matrix_.nnz // 2
            mean_d = np.mean(self.densities_)
            thr = mean_d * (self.min_density / 3.0)
            n_dense = int(np.sum(self.densities_ >= thr))
            print(f"  Stage 1: graph built. {n_edges} edges, "
                  f"{n_dense}/{n} dense nodes")

        # -- Stage 2: Seed initialization --
        seed_origins = self._seed_initialization()

        if self.verbose:
            print(f"  Stage 2: {len(seed_origins)} seeds")

        # -- Stage 3: Growth --
        clusters = self._iterative_growth(seed_origins)

        if self.verbose:
            total = sum(len(c) for c in clusters)
            print(f"  Stage 3: {len(clusters)} clusters ({total}/{n} pts)")

        # -- Stage 3.5: Filter small clusters BEFORE merge --
        # Kill fragments early so merge sees only substantial clusters.
        # Filtered points become noise; refinement recovers them later.
        min_size = max(self.min_cluster_size, 7)
        pre_filter = len(clusters)
        clusters = [c for c in clusters if len(c) >= min_size]
        if self.verbose and pre_filter != len(clusters):
            print(f"  Stage 3.5: {pre_filter} -> {len(clusters)} clusters "
                  f"(removed {pre_filter - len(clusters)} small fragments)")

        # -- Stage 4: Merging --
        if self.merge_adjacent and len(clusters) > 1:
            clusters = self._merge_clusters(clusters)
            if self.verbose:
                print(f"  Stage 4: {len(clusters)} after merging")

        self.clusters_ = clusters
        return self

    def predict_labels(self, X=None):
        """Assign labels including Stage 5 refinement."""
        if X is None:
            X = self.X_
        n = len(X)

        # Initialize labels based on self.clusters_
        labels = np.full(n, -1, dtype=int)
        for i, cluster_pts in enumerate(self.clusters_):
            labels[list(cluster_pts)] = i

        # Stage 5: Refinement
        labels = self._refine_boundaries(labels)

        # Filter small clusters
        min_size = max(self.min_cluster_size, 7)
        for cid in set(labels) - {-1}:
            if np.sum(labels == cid) < min_size:
                labels[labels == cid] = -1

        # Remap consecutive
        remaining = sorted(set(labels) - {-1})
        if remaining:
            mp = {old: new for new, old in enumerate(remaining)}
            for i in range(n):
                if labels[i] != -1:
                    labels[i] = mp[labels[i]]

        self.labels_ = labels
        return labels

    def fit_predict(self, X, precomputed_knn=None, min_cluster_size=None,
                    target_k=None):
        """Fit and predict.  If target_k is given (from sample tuning),
        apply unsupervised post-merge to bring cluster count down to target_k
        using shared-neighbor similarity (no ground truth needed)."""
        if min_cluster_size is not None:
            self.min_cluster_size = min_cluster_size
        self.fit(X, precomputed_knn=precomputed_knn)
        labels = self.predict_labels(X)

        if target_k is not None:
            k_now = len(set(labels) - {-1})
            if k_now > target_k:
                knn_idx = self.knn_indices_
                labels, n_merges = _unsupervised_post_merge(
                    labels, knn_idx, target_k, verbose=self.verbose
                )
                self.labels_ = labels
        return labels

    # ----------------------------------------------------------
    # STAGE 1: Build Graph (sparse matrix)
    # ----------------------------------------------------------

    def _build_graph(self, X, precomputed_knn=None):
        n = len(X)

        if precomputed_knn is not None:
            all_dists, all_indices = precomputed_knn
            k = min(self.k_neighbors, all_indices.shape[1])
            self.knn_dists_ = all_dists[:, :k]
            self.knn_indices_ = all_indices[:, :k]
        else:
            k = min(self.k_neighbors, n - 1)
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
            dists, indices = nbrs.kneighbors(X)
            self.knn_dists_ = dists[:, 1:]
            self.knn_indices_ = indices[:, 1:]

        # Density computation — two options:
        #   (a) Distance-based: 1 / mean_kNN_dist  (default)
        #   (b) Shared-neighbor: for each point, sum of shared neighbors
        #       with its k neighbors. More robust in HD where Euclidean
        #       distances concentrate, because it measures topological
        #       coherence rather than metric distance.
        if self.use_shared_neighbor_density:
            # SNN density: for point i, sum over j in kNN(i) of |kNN(i) ∩ kNN(j)|
            # Vectorized via neighbor-set membership lookup.
            knn_sets = [set(row) for row in self.knn_indices_]
            snn_scores = np.zeros(n, dtype=np.float64)
            for i in range(n):
                s = 0
                for j in self.knn_indices_[i]:
                    s += len(knn_sets[i] & knn_sets[j])
                snn_scores[i] = s
            # Normalize to [0, max] then convert to density-like scale
            k_used = self.knn_indices_.shape[1]
            self.densities_ = snn_scores / (k_used + 1e-10)
        else:
            mean_dists = np.mean(self.knn_dists_, axis=1)
            self.densities_ = 1.0 / (mean_dists + 1e-10)

        # Relative density: point's density / mean density of its neighbors
        # This makes the threshold ADAPTIVE — a point in a sparse region
        # can still be a seed if it's denser than its local neighborhood.
        nbr_densities = self.densities_[self.knn_indices_]  # (n, k)
        nbr_mean = np.mean(nbr_densities, axis=1)           # (n,)
        self.relative_densities_ = self.densities_ / (nbr_mean + 1e-10)

        # Build SPARSE adjacency (vectorized, no Python loops)
        # mutual kNN: only keep edge (i,j) if both i∈kNN(j) AND j∈kNN(i)
        # This prunes noisy asymmetric connections, especially in HD.
        k_actual = self.knn_indices_.shape[1]
        rows = np.repeat(np.arange(n), k_actual)
        cols = self.knn_indices_.ravel()
        data = np.ones(len(rows), dtype=np.int8)
        adj_forward = csr_matrix((data, (rows, cols)), shape=(n, n))
        if self.use_mutual_knn:
            # Mutual: intersection — edge only if both directions present
            self.adj_matrix_ = adj_forward.multiply(adj_forward.T).astype(np.int8)
        else:
            # Symmetric: union — edge if either direction present
            self.adj_matrix_ = ((adj_forward + adj_forward.T) > 0).astype(np.int8)

    # ----------------------------------------------------------
    # STAGE 2: Seed Initialization (uses RELATIVE density)
    # ----------------------------------------------------------

    def _seed_initialization(self):
        n = len(self.densities_)
        # Use relative density: a point is a seed candidate if it is
        # denser than its local neighborhood, scaled by min_density.
        # relative_density > 1.0 means "denser than neighbors".
        # Threshold: min_density/3.0 acts as a scaling factor.
        rel_threshold = self.min_density / 3.0  # e.g. 1.0 for min_density=3

        sorted_idx = np.argsort(-self.relative_densities_)
        occupied = np.zeros(n, dtype=bool)
        origins = []

        for idx_raw in sorted_idx:
            idx = int(idx_raw)
            if self.relative_densities_[idx] < rel_threshold:
                break
            if occupied[idx]:
                continue

            origins.append(idx)

            # Mark multi-hop neighborhood occupied.
            # seed_exclusion_hops controls seed spacing independently of k.
            # hops=1: only direct neighbors (original behavior)
            # hops=2+: expands exclusion zone, reducing over-seeding in HD
            occupied[idx] = True
            frontier = {idx}
            for _hop in range(self.seed_exclusion_hops):
                next_frontier = set()
                for node in frontier:
                    nbrs = self.adj_matrix_[node].nonzero()[1]
                    next_frontier.update(nbrs[~occupied[nbrs]])
                occupied[list(next_frontier)] = True
                frontier = next_frontier
                if not frontier:
                    break

        return origins

    # ----------------------------------------------------------
    # STAGE 3: Growth (numpy boolean masks)
    # ----------------------------------------------------------

    def _iterative_growth(self, seed_origins, max_iterations=50):
        n = len(self.X_)
        d = self.X_.shape[1]
        n_seeds = len(seed_origins)
        if n_seeds == 0:
            return []

        # Growth uses RELATIVE density: a point can be absorbed if it is
        # locally dense relative to its neighborhood. This is adaptive and
        # works in high dimensions where absolute densities concentrate.
        rel_growth_thr = (self.min_density / 3.0) * self.regular_threshold_factor

        # Boolean mask per seed: which points belong to it
        seed_masks = np.zeros((n_seeds, n), dtype=bool)
        for i, origin in enumerate(seed_origins):
            seed_masks[i, origin] = True

        owned = np.any(seed_masks, axis=0)
        dense = self.relative_densities_ >= rel_growth_thr
        active = np.ones(n_seeds, dtype=bool)
        alive = np.ones(n_seeds, dtype=bool)   # survives graduation
        growth_counts = np.zeros(n_seeds, dtype=int)

        # Graduation checkpoint: evaluate & kill weak seeds at this iteration
        grad_checkpoint = max(3, int(np.log2(max(n / 100, 1))))
        graduation_done = False

        for iteration in range(max_iterations):

            # --- Adaptive graduation at checkpoint ---
            if (not graduation_done and iteration == grad_checkpoint
                    and n_seeds > 1):
                graduation_done = True
                seed_sizes = seed_masks.sum(axis=1)
                alive_sizes = seed_sizes[alive]

                if len(alive_sizes) > 0 and alive_sizes.max() > 0:
                    best_size = alive_sizes.max()
                    median_size = max(np.median(alive_sizes), 1)
                    skew_ratio = best_size / median_size

                    # Adaptive persistence threshold based on density skew
                    # Balanced data (skew ~1-3): use full 50% of best
                    # Skewed data (skew >5): soften proportionally
                    if skew_ratio > 5.0:
                        persist_frac = 0.5 / (skew_ratio / 5.0)
                        persist_frac = max(persist_frac, 0.05)
                    else:
                        persist_frac = 0.5

                    for i in range(n_seeds):
                        if not alive[i]:
                            continue

                        # Test 1: Persistence — size >= persist_frac of best
                        persist_ok = seed_sizes[i] >= persist_frac * best_size

                        # Test 2: Sustained rate — grew in >= 60% of rounds
                        sustain_ok = (growth_counts[i] >=
                                      0.6 * grad_checkpoint)

                        if not (persist_ok and sustain_ok):
                            # Kill this seed, release its points
                            pts = np.where(seed_masks[i])[0]
                            owned[pts] = False
                            seed_masks[i, :] = False
                            alive[i] = False
                            active[i] = False

                    if self.verbose:
                        n_killed = int((~alive).sum())
                        n_alive = int(alive.sum())
                        print(f"    Graduation @iter {grad_checkpoint}: "
                              f"killed {n_killed}/{n_seeds}, "
                              f"{n_alive} survive "
                              f"(skew={skew_ratio:.1f}, "
                              f"persist={persist_frac:.2f})")

            # Precompute centroids for active clusters (used for competition)
            centroids = np.zeros((n_seeds, d))
            for i in range(n_seeds):
                if active[i] and alive[i]:
                    centroids[i] = np.mean(self.X_[seed_masks[i]], axis=0)

            # Phase 1: Collect expansion proposals from ALL clusters.
            proposals = {}  # node_id -> list of competing cluster_ids

            for i in range(n_seeds):
                if not active[i] or not alive[i]:
                    continue

                cluster_pts = np.where(seed_masks[i])[0]
                nbr_cols = self.adj_matrix_[cluster_pts].nonzero()[1]
                unique_nbrs = np.unique(nbr_cols)

                valid = (~owned[unique_nbrs] & dense[unique_nbrs])
                new_pts = unique_nbrs[valid]

                for pt in new_pts:
                    if pt not in proposals:
                        proposals[pt] = []
                    proposals[pt].append(i)

            # Phase 2: Resolve — uncontested nodes assigned directly,
            # contested nodes go to the cluster with nearest centroid.
            round_gains = np.zeros(n_seeds, dtype=int)

            for node, competing in proposals.items():
                if len(competing) == 1:
                    winner = competing[0]
                else:
                    dists = [np.linalg.norm(self.X_[node] - centroids[cid])
                             for cid in competing]
                    winner = competing[int(np.argmin(dists))]

                seed_masks[winner, node] = True
                owned[node] = True
                round_gains[winner] += 1

            any_growth = len(proposals) > 0

            # Deactivate seeds that gained nothing; count successful rounds
            for i in range(n_seeds):
                if active[i] and alive[i]:
                    if round_gains[i] > 0:
                        growth_counts[i] += 1
                    else:
                        active[i] = False

            if not any_growth:
                break

        # Collect clusters from alive seeds that grew
        clusters = []
        for i in range(n_seeds):
            if alive[i] and growth_counts[i] >= 1:
                pts = set(np.where(seed_masks[i])[0].tolist())
                if len(pts) > 1:  # at least seed + 1 absorbed point
                    clusters.append(pts)

        return clusters

    # ----------------------------------------------------------
    # STAGE 4: Merging (vectorized boundary detection)
    # ----------------------------------------------------------

    def _merge_clusters(self, clusters):
        """Iterative merge: repeat single-pass merging until no progress."""
        for _round in range(20):
            n_before = len(clusters)
            clusters = self._single_merge_pass(clusters)
            if len(clusters) >= n_before:
                break
        return clusters

    def _single_merge_pass(self, clusters):
        if len(clusters) <= 1:
            return clusters

        n_clusters = len(clusters)
        n_points = len(self.X_)

        # Point-to-cluster mapping
        pt_to_cl = np.full(n_points, -1, dtype=int)
        for i, cl in enumerate(clusters):
            pts = np.array(list(cl))
            pt_to_cl[pts] = i

        # Build cluster adjacency (vectorized: which clusters touch?)
        cluster_adj = {i: set() for i in range(n_clusters)}
        for i in range(n_clusters):
            pts_i = np.array(list(clusters[i]))
            nbr_cols = self.adj_matrix_[pts_i].nonzero()[1]
            adj_cls = set(pt_to_cl[nbr_cols].tolist()) - {-1, i}
            cluster_adj[i] = adj_cls

        # Union-Find
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

        # Only check ADJACENT cluster pairs
        checked = set()
        for i in range(n_clusters):
            for j in cluster_adj[i]:
                a, b = min(i, j), max(i, j)
                if (a, b) in checked:
                    continue
                checked.add((a, b))

                if find(a) == find(b):
                    continue

                # --- Criterion 1: Boundary density (vectorized) ---
                pts_a = np.array(list(clusters[a]))
                mask_b = np.zeros(n_points, dtype=bool)
                mask_b[list(clusters[b])] = True

                _, cols_a = self.adj_matrix_[pts_a].nonzero()
                boundary_a = pts_a[np.unique(
                    self.adj_matrix_[pts_a].nonzero()[0][mask_b[cols_a]]
                )] if np.any(mask_b[cols_a]) else np.array([], dtype=int)

                if len(boundary_a) == 0:
                    continue

                pts_b = np.array(list(clusters[b]))
                mask_a = np.zeros(n_points, dtype=bool)
                mask_a[list(clusters[a])] = True
                _, cols_b = self.adj_matrix_[pts_b].nonzero()
                boundary_b = pts_b[np.unique(
                    self.adj_matrix_[pts_b].nonzero()[0][mask_a[cols_b]]
                )] if np.any(mask_a[cols_b]) else np.array([], dtype=int)

                all_boundary = np.concatenate([boundary_a, boundary_b])
                avg_bd = np.mean(self.densities_[all_boundary])
                mean_density = np.mean(self.densities_)
                if avg_bd < mean_density * (self.min_density / 3.0):
                    continue

                # --- Criterion 2: Centroid distance ---
                data_a = self.X_[pts_a]
                data_b = self.X_[pts_b]

                if len(data_a) < 5 or len(data_b) < 5:
                    union(a, b)
                    continue

                centroid_a = data_a.mean(0)
                centroid_b = data_b.mean(0)
                dist = np.linalg.norm(centroid_a - centroid_b)
                radius_a = np.mean(np.linalg.norm(data_a - centroid_a, axis=1))
                radius_b = np.mean(np.linalg.norm(data_b - centroid_b, axis=1))
                norm_dist = dist / (radius_a + radius_b + 1e-10)

                if norm_dist > self.centroid_distance_threshold:
                    continue

                # --- Criterion 3: KS test — fraction of dimensions passing ---
                # In HD, requiring ALL dimensions to pass is too strict
                # (0.95^d → near-zero for large d). Instead, require that
                # a sufficient fraction of dimensions show compatible
                # distributions (p >= alpha).
                n_dims = self.X_.shape[1]
                n_pass = 0
                for dim in range(n_dims):
                    try:
                        _, p = ks_2samp(data_a[:, dim], data_b[:, dim])
                        if p >= self.ks_test_alpha:
                            n_pass += 1
                    except Exception:
                        n_pass += 1  # benefit of the doubt on error
                # Require >= ks_merge_fraction of dimensions to pass
                ks_fraction = n_pass / n_dims if n_dims > 0 else 1.0
                if ks_fraction < self.ks_merge_fraction:
                    continue

                union(a, b)

        # Collect merged clusters
        groups = {}
        for i in range(n_clusters):
            r = find(i)
            if r not in groups:
                groups[r] = set()
            groups[r].update(clusters[i])

        return list(groups.values())

    # ----------------------------------------------------------
    # STAGE 5: Refinement (kNN vote, no per-cluster models)
    # ----------------------------------------------------------

    def _refine_boundaries(self, labels):
        """Assign noise points by majority vote among kNN neighbors."""
        refined = labels.copy()
        n_clusters = len(set(labels) - {-1})
        if n_clusters == 0:
            return refined

        # min_votes scales with refinement_sigma (higher sigma = fewer votes needed)
        min_votes = max(1, int(self.k_neighbors / (self.refinement_sigma * 2 + 1)))

        for iteration in range(10):
            noise_idx = np.where(refined == -1)[0]
            if len(noise_idx) == 0:
                break

            # Get neighbor labels for all noise points at once
            nbr_labels = refined[self.knn_indices_[noise_idx]]  # (n_noise, k)

            changed = False
            for i in range(len(noise_idx)):
                non_noise = nbr_labels[i][nbr_labels[i] != -1]
                if len(non_noise) >= min_votes:
                    counts = np.bincount(non_noise, minlength=n_clusters)
                    best = int(np.argmax(counts))
                    if counts[best] >= min_votes:
                        refined[noise_idx[i]] = best
                        changed = True

            if not changed:
                break

        return refined


# ======================================================
# TUNING FUNCTION (optimized: precomputes kNN once)
# + Parallel trial execution via joblib
# + Early stopping (patience-based)
# ======================================================

def _run_single_trial(X, y_true, params, precomputed, expected_k):
    """Execute one tuning trial. Returns (score, labels, params, n_clusters) or None.

    This is a module-level function so joblib can pickle it for parallel execution.
    The per-trial computation is IDENTICAL to the original sequential loop body.
    """
    try:
        model = AdaBoxGraph(
            k_neighbors=params['k_neighbors'],
            min_density=params['min_density'],
            regular_threshold_factor=params['regular_threshold_factor'],
            merge_adjacent=params['merge_adjacent'],
            refinement_sigma=params['refinement_sigma'],
            min_cluster_size=params['min_cluster_size'],
            centroid_distance_threshold=params['centroid_distance_threshold'],
            ks_test_alpha=params['ks_test_alpha'],
            seed_exclusion_hops=params['seed_exclusion_hops'],
            use_mutual_knn=params['use_mutual_knn'],
            use_shared_neighbor_density=params['use_shared_neighbor_density'],
            ks_merge_fraction=params['ks_merge_fraction'],
            verbose=False
        )
        labels = model.fit_predict(X, precomputed_knn=precomputed)

        n_clusters = len(set(labels) - {-1})
        if n_clusters == 0:
            return None
        noise_ratio = np.sum(labels == -1) / len(labels)
        if noise_ratio > 0.8:
            return None

        try:
            scope_result = compute_dice_metrics(X, y_true, labels, verbose=False)
            tuning_score = scope_result['Overall_Score']
        except Exception:
            tuning_score = 0.0

        if expected_k is not None and n_clusters > 0:
            ratio = max(n_clusters / expected_k, expected_k / n_clusters)
            if ratio >= 1.5:
                penalty = min(0.25, 0.15 * (ratio - 1.0))
                tuning_score = tuning_score * (1.0 - penalty)

        return (tuning_score, labels, params, n_clusters)
    except Exception:
        return None


# ======================================================
# GREEDY POST-MERGE (anti-fragmentation)
# ======================================================

def _find_knn_adjacent_pairs(labels, knn_indices):
    """Find adjacent cluster pairs via kNN graph edges (vectorized)."""
    k = knn_indices.shape[1]
    nbr_labels = labels[knn_indices]                # (n, k)
    src = np.repeat(labels, k)                      # (n*k,)
    dst = nbr_labels.ravel()                        # (n*k,)

    mask = (src != -1) & (dst != -1) & (src != dst)
    a, b = src[mask], dst[mask]
    if len(a) == 0:
        return []

    pairs = np.column_stack([np.minimum(a, b), np.maximum(a, b)])
    unique = np.unique(pairs, axis=0)
    return [tuple(p) for p in unique]


def _compute_pair_similarity(labels, knn_indices, cid_a, cid_b):
    """Shared-neighbor overlap between two clusters — fully unsupervised.

    For points in cluster A that have kNN neighbors in cluster B (and vice versa),
    compute the fraction of cross-boundary edges relative to each cluster's size.
    Higher = more intertwined = should merge.
    """
    mask_a = (labels == cid_a)
    mask_b = (labels == cid_b)
    n_a = mask_a.sum()
    n_b = mask_b.sum()
    if n_a == 0 or n_b == 0:
        return 0.0

    # Count kNN edges from A→B and B→A
    nbr_labels_a = labels[knn_indices[mask_a]]   # (n_a, k)
    cross_ab = (nbr_labels_a == cid_b).sum()     # edges from A pointing into B

    nbr_labels_b = labels[knn_indices[mask_b]]   # (n_b, k)
    cross_ba = (nbr_labels_b == cid_a).sum()     # edges from B pointing into A

    k = knn_indices.shape[1]
    # Jaccard-style: cross edges / total possible edges
    similarity = (cross_ab + cross_ba) / (k * (n_a + n_b) + 1e-10)
    return similarity


def _merge_labels(labels, cid_a, cid_b):
    """Merge cid_b into cid_a, renumber to consecutive IDs."""
    out = labels.copy()
    out[out == cid_b] = cid_a
    remaining = np.unique(out[out != -1])
    if len(remaining) > 0:
        mp = np.full(remaining.max() + 1, -1, dtype=int)
        for new_id, old_id in enumerate(remaining):
            mp[old_id] = new_id
        non_noise = out != -1
        out[non_noise] = mp[out[non_noise]]
    return out


def _unsupervised_post_merge(labels, knn_indices, target_k, verbose=False):
    """Unsupervised greedy post-merge guided by sample-derived target_k.

    Merge order: most similar pair first (shared-neighbor overlap).
    Stop condition: number of clusters reaches target_k.

    This bridges the gap between supervised tuning (sample) and unsupervised
    deployment (full data) — the sample k is the ONLY supervised signal used.
    """
    current_labels = labels.copy()
    merge_count = 0
    k_now = len(set(current_labels) - {-1})

    if k_now <= target_k:
        if verbose:
            print(f"      Unsupervised post-merge: already at k={k_now} <= target={target_k}, skip")
        return current_labels, 0

    if verbose:
        print(f"      Unsupervised post-merge: k={k_now} → target={target_k}")

    max_rounds = k_now - target_k + 5  # safety margin
    for round_num in range(1, max_rounds + 1):
        k_now = len(set(current_labels) - {-1})
        if k_now <= target_k:
            break

        pairs = _find_knn_adjacent_pairs(current_labels, knn_indices)
        if not pairs:
            if verbose:
                print(f"      Round {round_num}: no adjacent pairs left, stop at k={k_now}")
            break

        # Score all pairs by shared-neighbor similarity
        scored = []
        for (ca, cb) in pairs:
            sim = _compute_pair_similarity(current_labels, knn_indices, ca, cb)
            scored.append((sim, ca, cb))

        # Merge the most similar pair
        scored.sort(reverse=True)
        best_sim, best_a, best_b = scored[0]
        current_labels = _merge_labels(current_labels, best_a, best_b)
        merge_count += 1

        if verbose:
            k_after = len(set(current_labels) - {-1})
            print(f"      Round {round_num}: merge C{best_a}+C{best_b} "
                  f"(sim={best_sim:.4f}), k={k_after}")

    if verbose:
        k_final = len(set(current_labels) - {-1})
        print(f"      Unsupervised post-merge done: {merge_count} merges, k={k_final}")

    return current_labels, merge_count


def adaptive_deploy(X, sample_params, target_k, verbose=False):
    """Deploy AdaHD on full data with adaptive graph resolution.

    Instead of blindly using the sample's k_frac (which under-resolves at scale),
    try several k_neighbors values and pick the one whose Stage-4 output is
    closest to a good starting point for the unsupervised post-merge.

    Strategy:
      1. Resolve the sample k_frac to a base k for this n.
      2. Try k_base and several multiples: [k_base, 1.5×, 2×, 3×, 4×].
      3. For each k, run full pipeline (Stages 1-5) — cheap, single run, no tuning.
      4. Pick the k where post-Stage-5 cluster count is in [target_k, 3× target_k].
         (close enough that post-merge only needs a few high-quality merges)
      5. Run unsupervised post-merge on the best candidate down to target_k.

    Parameters
    ----------
    X : array (n, d)
    sample_params : dict from sample tuning (contains fracs + other params)
    target_k : int — cluster count from sample (supervised signal)
    verbose : bool

    Returns
    -------
    labels : array of cluster labels
    info : dict with deployment details
    """
    n = len(X)

    # Resolve base k from fraction
    k_frac = sample_params.get('k_neighbors_frac', 2.0)
    mcs_frac = sample_params.get('min_cluster_size_frac', 0.5)
    base_k = max(3, min(round(k_frac * np.log2(n)), n - 1))

    # Candidate k values: base and upward multiples
    k_multipliers = [1.0, 1.5, 2.0, 3.0, 4.0]
    k_candidates = sorted(set(
        max(3, min(round(base_k * m), n - 1)) for m in k_multipliers
    ))

    if verbose:
        print(f"    Adaptive deploy: n={n}, base_k={base_k}, "
              f"target_k={target_k}, candidates={k_candidates}")

    # Extract non-graph params from sample
    other_params = {k: v for k, v in sample_params.items()
                    if k not in ('k_neighbors', 'k_neighbors_frac',
                                 'min_cluster_size', 'min_cluster_size_frac',
                                 'post_merge_count')}

    # Ideal: post-stage-5 gives ~1.5-3× target_k so post-merge does few merges
    ideal_low = target_k
    ideal_high = max(target_k * 3, target_k + 5)

    best_labels = None
    best_k_val = None
    best_cluster_count = None
    best_distance = float('inf')  # distance from ideal range

    candidate_results = []

    for k_val in k_candidates:
        model = AdaBoxGraph(
            k_neighbors=k_val,
            min_cluster_size_frac=mcs_frac,
            min_density=other_params.get('min_density', 3.0),
            regular_threshold_factor=other_params.get('regular_threshold_factor', 1.0),
            merge_adjacent=other_params.get('merge_adjacent', True),
            refinement_sigma=other_params.get('refinement_sigma', 2.0),
            centroid_distance_threshold=other_params.get('centroid_distance_threshold', 2.0),
            ks_test_alpha=other_params.get('ks_test_alpha', 0.05),
            seed_exclusion_hops=other_params.get('seed_exclusion_hops', 2),
            use_mutual_knn=other_params.get('use_mutual_knn', False),
            use_shared_neighbor_density=other_params.get('use_shared_neighbor_density', False),
            ks_merge_fraction=other_params.get('ks_merge_fraction', 0.5),
            verbose=False
        )
        labels = model.fit_predict(X)
        k_found = len(set(labels) - {-1})

        # Distance to ideal range
        if k_found < ideal_low:
            dist = ideal_low - k_found  # under-clustered (bad — can't split)
            dist += 100  # heavy penalty: we can merge down but can't split up
        elif k_found <= ideal_high:
            dist = 0  # in the sweet spot
        else:
            dist = k_found - ideal_high  # over-clustered (post-merge can fix)

        candidate_results.append((k_val, k_found, dist))

        if verbose:
            tag = " ◀ BEST" if dist < best_distance else ""
            print(f"      k={k_val:3d} → {k_found:3d} clusters "
                  f"(ideal={ideal_low}-{ideal_high}, dist={dist}){tag}")

        if dist < best_distance:
            best_distance = dist
            best_labels = labels
            best_k_val = k_val
            best_cluster_count = k_found
            best_model = model

    if verbose:
        print(f"    Selected k={best_k_val} → {best_cluster_count} clusters")

    # Apply unsupervised post-merge if needed
    final_labels = best_labels
    n_merges = 0
    if best_cluster_count > target_k:
        knn_idx = best_model.knn_indices_
        final_labels, n_merges = _unsupervised_post_merge(
            best_labels, knn_idx, target_k, verbose=verbose
        )

    final_k = len(set(final_labels) - {-1})
    info = {
        'k_tested': k_candidates,
        'k_selected': best_k_val,
        'k_before_merge': best_cluster_count,
        'k_final': final_k,
        'n_merges': n_merges,
        'mcs_resolved': best_model.min_cluster_size,
        'candidate_results': candidate_results,
    }

    if verbose:
        print(f"    Final: k={final_k} (target={target_k}), "
              f"{n_merges} merges, selected_k_neighbors={best_k_val}")

    return final_labels, info


def density_aware_sample(X, sample_size, k_density='auto', min_per_mode=20,
                         min_mode_size=None, max_density_points=20000,
                         random_state=42, verbose=False):
    """
    Density-aware sampling: analyze the full data's density landscape,
    detect density modes, and sample proportionally from each mode
    with a guaranteed minimum per mode.

    Algorithm:
    1. If n > max_density_points, run density analysis on a random subset,
       then assign all n points to nearest mode centroid (fast path)
    2. Compute kNN density (adaptive k = max(15, sqrt(n_analysis)))
    3. Hill-climb to local density maxima to find modes
    4. Merge small modes into their nearest large neighbor
    5. Sample proportionally from each mode with minimum floor

    Parameters
    ----------
    X : array (n, d) — full dataset (scaled)
    sample_size : int — desired sample size
    k_density : int or 'auto' — k for kNN density; 'auto' = max(15, sqrt(n_analysis))
    min_per_mode : int — minimum sample points per density mode (default 20)
    min_mode_size : int or None — modes smaller than this are merged (default sqrt(n_analysis))
    max_density_points : int — max points for density analysis; larger datasets use
                         a random subset for mode discovery, then assign all points
                         to nearest mode centroid (default 20000)
    random_state : int — random seed
    verbose : bool

    Returns
    -------
    indices : array of int — indices into X for the sample
    mode_labels : array (n,) — mode assignment for ALL full-data points
    info : dict with details (n_modes, mode_sizes, etc.)
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.RandomState(random_state)
    n = len(X)
    sample_size = min(sample_size, n)

    # ── Fast path: subsample for density analysis if dataset is large ──
    use_fast_path = n > max_density_points
    if use_fast_path:
        sub_idx = rng.choice(n, max_density_points, replace=False)
        X_analysis = X[sub_idx]
        n_analysis = max_density_points
        if verbose:
            print(f"    Fast path: density analysis on {n_analysis:,} / {n:,} points")
    else:
        X_analysis = X
        n_analysis = n
        sub_idx = None

    # ── Step 1: Adaptive kNN density estimation (on analysis subset) ──
    if k_density == 'auto':
        k_density = max(15, int(np.sqrt(n_analysis)))
    k_use = min(k_density, n_analysis - 1)

    nn = NearestNeighbors(n_neighbors=k_use + 1, algorithm='auto', metric='euclidean')
    nn.fit(X_analysis)
    distances, knn_indices = nn.kneighbors(X_analysis)

    # Density = 1 / mean distance to k neighbors (skip self at index 0)
    mean_dist = distances[:, 1:].mean(axis=1)
    density = 1.0 / (mean_dist + 1e-10)

    if verbose:
        print(f"    Density (k={k_use}): min={density.min():.3f}, "
              f"max={density.max():.3f}, median={np.median(density):.3f}")

    # ── Step 2: Hill-climb to density maxima ──
    neighbor_dens = density[knn_indices[:, 1:]]  # (n_analysis, k_use)
    best_nb_local = np.argmax(neighbor_dens, axis=1)
    climb_target = knn_indices[np.arange(n_analysis), best_nb_local + 1]

    self_is_max = density >= density[climb_target]
    climb_target[self_is_max] = np.arange(n_analysis)[self_is_max]

    assignment = np.arange(n_analysis, dtype=int)
    for step in range(200):
        next_step = climb_target[assignment]
        if np.array_equal(next_step, assignment):
            break
        assignment = next_step

    unique_peaks_raw = np.unique(assignment)
    n_raw = len(unique_peaks_raw)
    if verbose:
        print(f"    Raw modes after hill-climb: {n_raw}")

    # ── Step 3: Merge small modes into nearest large mode ──
    if min_mode_size is None:
        min_mode_size = max(10, int(np.sqrt(n_analysis)))

    # Build initial mode labels (on analysis subset)
    peak_to_mode = {p: i for i, p in enumerate(unique_peaks_raw)}
    mode_labels = np.array([peak_to_mode[assignment[i]] for i in range(n_analysis)])
    n_modes_raw = len(unique_peaks_raw)

    mode_sizes_raw = np.bincount(mode_labels, minlength=n_modes_raw)
    large_modes = np.where(mode_sizes_raw >= min_mode_size)[0]
    small_modes = np.where(mode_sizes_raw < min_mode_size)[0]

    if len(large_modes) == 0:
        # All modes are small — keep top modes by size
        top_n = max(2, sample_size // min_per_mode)
        large_modes = np.argsort(-mode_sizes_raw)[:top_n]
        small_modes = np.setdiff1d(np.arange(n_modes_raw), large_modes)

    if len(small_modes) > 0 and len(large_modes) > 0:
        # Compute centroids for large modes (from analysis subset)
        large_centroids = np.array([X_analysis[mode_labels == m].mean(axis=0) for m in large_modes])

        # Merge each small mode into nearest large mode
        merge_map = {}  # old_mode_id -> new_mode_id (among large_modes)
        for sm in small_modes:
            sm_pts = np.where(mode_labels == sm)[0]
            if len(sm_pts) == 0:
                continue
            sm_centroid = X_analysis[sm_pts].mean(axis=0)
            dists = np.linalg.norm(large_centroids - sm_centroid, axis=1)
            nearest_large = large_modes[np.argmin(dists)]
            merge_map[sm] = nearest_large

        # Apply merges
        for sm, target in merge_map.items():
            mode_labels[mode_labels == sm] = target

    # Re-number modes to be contiguous 0..n_modes-1
    unique_final = np.unique(mode_labels)
    remap = {old: new for new, old in enumerate(unique_final)}
    mode_labels = np.array([remap[m] for m in mode_labels])

    n_modes = len(unique_final)
    mode_sizes = np.bincount(mode_labels, minlength=n_modes)

    if verbose:
        print(f"    After merging (min_mode_size={min_mode_size}): "
              f"{n_modes} modes (merged {n_raw - n_modes})")
        print(f"    Mode sizes: min={mode_sizes.min()}, max={mode_sizes.max()}, "
              f"median={np.median(mode_sizes):.0f}")

    # ── Fast path: assign ALL n points to nearest mode centroid ──
    if use_fast_path:
        # Compute centroids for each final mode from analysis subset
        mode_centroids = np.array([X_analysis[mode_labels == m].mean(axis=0)
                                   for m in range(n_modes)])
        # Assign every point in full X to nearest mode centroid
        # Process in chunks to avoid memory blow-up
        chunk_size = 50000
        full_mode_labels = np.empty(n, dtype=int)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dists = np.linalg.norm(X[start:end, np.newaxis, :] - mode_centroids[np.newaxis, :, :], axis=2)
            full_mode_labels[start:end] = np.argmin(dists, axis=1)

        mode_labels = full_mode_labels
        mode_sizes = np.bincount(mode_labels, minlength=n_modes)

        if verbose:
            print(f"    Fast path: assigned {n:,} points to {n_modes} modes")
            print(f"    Full-data mode sizes: min={mode_sizes.min()}, max={mode_sizes.max()}, "
                  f"median={np.median(mode_sizes):.0f}")

    # ── Step 4: Proportional sampling with minimum floor ──
    n_per_mode = np.zeros(n_modes, dtype=int)

    # Floor allocation
    for m in range(n_modes):
        n_per_mode[m] = min(min_per_mode, mode_sizes[m])

    remaining = sample_size - n_per_mode.sum()
    if remaining > 0:
        eligible = mode_sizes - n_per_mode
        weights = eligible.astype(float)
        if weights.sum() > 0:
            weights /= weights.sum()
            extra = (weights * remaining).astype(int)
            leftover = remaining - extra.sum()
            if leftover > 0:
                top_modes = np.argsort(-weights)[:leftover]
                extra[top_modes] += 1
            n_per_mode += np.minimum(extra, eligible)

    # Safety: fill deficit from largest modes
    total = n_per_mode.sum()
    if total < sample_size:
        deficit = sample_size - total
        for m in np.argsort(-mode_sizes):
            can_add = mode_sizes[m] - n_per_mode[m]
            add = min(can_add, deficit)
            n_per_mode[m] += add
            deficit -= add
            if deficit <= 0:
                break

    # ── Actual sampling ──
    indices = []
    for m in range(n_modes):
        mode_pts = np.where(mode_labels == m)[0]
        n_take = min(n_per_mode[m], len(mode_pts))
        if n_take > 0:
            chosen = rng.choice(mode_pts, n_take, replace=False)
            indices.extend(chosen)

    indices = np.array(indices)
    rng.shuffle(indices)

    if len(indices) > sample_size:
        indices = indices[:sample_size]

    if verbose:
        print(f"    Sampled {len(indices)} points from {n_modes} modes")
        for m in range(min(n_modes, 15)):
            print(f"      Mode {m}: size={mode_sizes[m]:5d}, sampled={n_per_mode[m]:4d}")
        if n_modes > 15:
            print(f"      ... and {n_modes - 15} more modes")

    info = {
        'n_modes': n_modes,
        'mode_sizes': mode_sizes,
        'n_per_mode': n_per_mode,
        'density': density,
        'mode_labels': mode_labels,
        'peak_indices': unique_final,
        'fallback': False,
    }

    return indices, mode_labels, info


def prototype_deploy(X_full, X_sample, sample_labels, k_vote=7,
                     conf_threshold=0.75, k_vote_pass2=15, verbose=False):
    """
    Two-pass prototype deployment: transfer sample clustering to full data
    using kNN voting with label propagation.

    Pass 1: kNN vote using sample's labeled points as prototypes.
    Pass 2: High-confidence points from Pass 1 become additional prototypes.
             Low-confidence boundary points are re-voted with massively
             denser prototype coverage (e.g. 8000+ prototypes instead of 1000).

    Parameters
    ----------
    X_full : array (n_full, d) — full dataset (scaled)
    X_sample : array (n_sample, d) — sample points (scaled, subset of X_full)
    sample_labels : array (n_sample,) — cluster labels from sample tuning
    k_vote : int — number of nearest neighbors for Pass 1 (default 7)
    conf_threshold : float — confidence cutoff for Pass 2 re-voting (default 0.75)
    k_vote_pass2 : int — number of nearest neighbors for Pass 2 (default 15)
    verbose : bool

    Returns
    -------
    full_labels : array (n_full,) — cluster labels for all points
    info : dict with deployment details
    """
    from sklearn.neighbors import NearestNeighbors

    n_full = len(X_full)
    n_sample = len(X_sample)

    # Filter out noise points from sample — only use clustered points as prototypes
    valid_mask = sample_labels != -1
    X_proto = X_sample[valid_mask]
    y_proto = sample_labels[valid_mask]
    n_proto = len(X_proto)

    if n_proto == 0:
        if verbose:
            print("    WARNING: No valid prototypes (all noise). Returning all noise.")
        return np.full(n_full, -1, dtype=int), {'method': 'prototype', 'n_proto': 0}

    k_eff = min(k_vote, n_proto)

    if verbose:
        n_clusters = len(set(y_proto))
        print(f"    Pass 1: {n_proto} prototypes from {n_sample} sample pts "
              f"({n_clusters} clusters), k_vote={k_eff}")

    # ── Pass 1: kNN vote from sample prototypes ──
    def _knn_vote(X_query, X_ref, y_ref, k):
        """Weighted kNN vote. Returns labels and confidence for each query point."""
        k_use = min(k, len(X_ref))
        nn = NearestNeighbors(n_neighbors=k_use, algorithm='auto', metric='euclidean')
        nn.fit(X_ref)
        distances, indices = nn.kneighbors(X_query)
        neighbor_labels = y_ref[indices]

        n_q = len(X_query)
        labels_out = np.empty(n_q, dtype=int)
        conf_out = np.empty(n_q, dtype=float)

        for i in range(n_q):
            dists_i = distances[i]
            labs_i = neighbor_labels[i]

            if dists_i[0] < 1e-12:
                labels_out[i] = labs_i[0]
                conf_out[i] = 1.0
                continue

            weights = 1.0 / (dists_i + 1e-10)
            unique_labs = np.unique(labs_i)
            weighted_votes = np.zeros(len(unique_labs))
            for j, lab in enumerate(unique_labs):
                weighted_votes[j] = weights[labs_i == lab].sum()

            winner_idx = np.argmax(weighted_votes)
            labels_out[i] = unique_labs[winner_idx]
            conf_out[i] = weighted_votes[winner_idx] / weights.sum()

        return labels_out, conf_out

    full_labels, confidence = _knn_vote(X_full, X_proto, y_proto, k_eff)

    # ── Pass 2: Re-vote low-confidence points with augmented prototypes ──
    low_conf_mask = confidence < conf_threshold
    n_low = low_conf_mask.sum()

    if n_low > 0 and n_low < n_full:
        high_conf_mask = ~low_conf_mask

        # Augmented prototype set = original sample prototypes + high-confidence full points
        X_proto2 = np.vstack([X_proto, X_full[high_conf_mask]])
        y_proto2 = np.concatenate([y_proto, full_labels[high_conf_mask]])
        n_proto2 = len(X_proto2)

        k_eff2 = min(k_vote_pass2, n_proto2)

        if verbose:
            print(f"    Pass 2: {n_low} low-conf points ({100*n_low/n_full:.1f}%), "
                  f"re-voting with {n_proto2} prototypes (was {n_proto}), k_vote={k_eff2}")

        new_labels, new_conf = _knn_vote(X_full[low_conf_mask], X_proto2, y_proto2, k_eff2)
        full_labels[low_conf_mask] = new_labels
        confidence[low_conf_mask] = new_conf
    elif verbose:
        print(f"    Pass 2: skipped (0 low-conf points)")

    # Renumber labels to be consecutive starting from 0
    unique_final = np.unique(full_labels[full_labels != -1])
    if len(unique_final) > 0:
        remap = {old: new for new, old in enumerate(unique_final)}
        for i in range(n_full):
            if full_labels[i] != -1:
                full_labels[i] = remap[full_labels[i]]

    final_k = len(set(full_labels) - {-1})
    mean_conf = np.mean(confidence)
    low_conf_final = np.sum(confidence < 0.5) / n_full * 100

    if verbose:
        print(f"    Final: k={final_k}, mean_confidence={mean_conf:.3f}, "
              f"low_confidence(<50%)={low_conf_final:.1f}%")

    info = {
        'method': 'prototype_2pass',
        'n_proto_pass1': n_proto,
        'n_proto_pass2': n_proto2 if n_low > 0 and n_low < n_full else n_proto,
        'n_low_conf': int(n_low),
        'k_vote': k_eff,
        'k_vote_pass2': k_eff2 if n_low > 0 and n_low < n_full else 0,
        'k_final': final_k,
        'mean_confidence': mean_conf,
        'low_confidence_pct': low_conf_final,
    }

    return full_labels, info

    """Merge cid_b into cid_a, renumber to consecutive IDs."""
    out = labels.copy()
    out[out == cid_b] = cid_a
    remaining = np.unique(out[out != -1])
    if len(remaining) > 0:
        mp = np.full(remaining.max() + 1, -1, dtype=int)
        for new_id, old_id in enumerate(remaining):
            mp[old_id] = new_id
        non_noise = out != -1
        out[non_noise] = mp[out[non_noise]]
    return out


def _greedy_post_merge(X, y_true, labels, knn_indices, max_rounds=15, verbose=False):
    """
    Greedy post-merge: iteratively merge adjacent cluster pairs to reduce
    fragmentation.  After each round the single best merge (highest SCOPE
    improvement) is applied.  Stops when no merge improves SCOPE or
    max_rounds reached.

    Smart sampling: when k > 10, randomly sample pairs instead of testing
    all combinations (much faster for highly fragmented results).
    """
    import random as _rnd

    current_labels = labels.copy()
    merge_count = 0

    try:
        result = compute_dice_metrics(X, y_true, current_labels, verbose=False)
        current_score = result['Overall_Score']
    except Exception:
        return current_labels, 0

    for round_num in range(1, max_rounds + 1):
        pairs = _find_knn_adjacent_pairs(current_labels, knn_indices)
        if not pairs:
            break

        k_now = len(set(current_labels) - {-1})

        # Smart sampling: exhaustive when few clusters, random when many
        if k_now <= 10 or len(pairs) <= 20:
            test_pairs = pairs
        else:
            # Cap at 30 pairs per round to keep speed reasonable
            n_sample = min(len(pairs), 30)
            test_pairs = _rnd.sample(pairs, n_sample)

        best_improvement = 0
        best_new_labels = None
        best_score_new = current_score

        for (cid_a, cid_b) in test_pairs:
            test_labels = _merge_labels(current_labels, cid_a, cid_b)
            try:
                test_result = compute_dice_metrics(X, y_true, test_labels, verbose=False)
                test_score = test_result['Overall_Score']
            except Exception:
                continue

            improvement = test_score - current_score
            if improvement > best_improvement:
                best_improvement = improvement
                best_new_labels = test_labels
                best_score_new = test_score

        if best_new_labels is not None and best_improvement > 0:
            current_labels = best_new_labels
            current_score = best_score_new
            merge_count += 1
            if verbose:
                k_now = len(set(current_labels) - {-1})
                sampled = f" (sampled {len(test_pairs)}/{len(pairs)} pairs)" if len(test_pairs) < len(pairs) else ""
                print(f"      Post-merge round {round_num}: k={k_now}, "
                      f"SCOPE={current_score:.4f} (+{best_improvement:.4f}){sampled}")
        else:
            break

    return current_labels, merge_count


def tune_adaboxgraph_random(X, y_true, n_trials=100, expected_k=None, verbose=False,
                            n_jobs=-1, patience=100, return_all_trials=False,
                            reduced_search=False, aggressive_search=False):
    """
    Random search tuning with precomputed kNN graph.
    The kNN is built ONCE at max_k=30, then reused across all trials.

    Speed features (do NOT touch per-trial computation):
      - n_jobs:   parallel trial execution via joblib (-1 = all cores, 1 = sequential)
      - patience: early stopping — stop if no improvement in this many consecutive trials

    Parameters
    ----------
    X          : array (n, d)
    y_true     : true labels (for ARI scoring)
    n_trials   : number of random trials
    expected_k : optional hint for number of clusters
    verbose    : print progress
    n_jobs     : number of parallel workers (-1 = all cores, 1 = sequential)
    patience   : stop after this many trials without improvement (None = disabled)
    return_all_trials : bool
        If True, return a 4th element: list of (score, params, n_clusters) for all valid trials.
    reduced_search : bool
        If True, use reduced search space based on sensitivity analysis:
        FIX merge_adjacent=True, use_mutual_knn=False, use_shared_neighbor_density=False
        NARROW k_neighbors, seed_exclusion_hops, refinement_sigma
    aggressive_search : bool
        If True, use aggressive search space (1,458 combos):
        FIX 5 params (merge_adjacent, min_cluster_size, seed_exclusion_hops,
        use_mutual_knn, use_shared_neighbor_density)
        NARROW remaining 7 params to tightest ranges.
        Overrides reduced_search if both are True.
    """
    import random as rnd

    n_samples = len(X)
    log2_n = np.log2(max(n_samples, 2))
    sqrt_n = np.sqrt(n_samples)
    max_k = min(30, n_samples - 1)

    # === PRECOMPUTE kNN ONCE ===
    if verbose:
        print(f"    Precomputing kNN (k={max_k})...")
    precomputed = precompute_knn(X, max_k=max_k)
    if verbose:
        mode = "AGGRESSIVE" if aggressive_search else ("REDUCED" if reduced_search else "FULL")
        print(f"    Done. Starting {n_trials} trials ({mode}, n_jobs={n_jobs}, patience={patience})...")

    # ── Scale-invariant fractional search spaces ──
    # k_neighbors_frac:  k = round(frac * log2(n)), clamped to [3, n-1]
    #   At n=400 (log2=8.6): fracs [2.0, 2.6, 3.3, 3.9] → k ≈ [17, 22, 28, 34]
    #   At n=10K  (log2=13.3): same fracs → k ≈ [27, 35, 44, 52]
    # min_cluster_size_frac:  mcs = max(3, round(frac * sqrt(n)))  [sublinear!]
    #   At n=400 (sqrt=20): fracs [0.15, 0.5, 1.0] → mcs = [3, 10, 20]
    #   At n=10K (sqrt=100): same fracs → mcs = [15, 50, 100]

    if aggressive_search:
        k_frac_choices = [2.6, 3.3, 3.9]
        min_density_choices = [1.0, 2.0, 3.0]
        reg_thr_choices = [0.5, 0.7, 1.0]
        merge_adj_choices = [True]
        ref_sigma_choices = [2.0, 3.0]
        mcs_frac_choices = [0.5]
        centroid_dist_choices = [2.0, 3.0, 5.0]
        ks_alpha_choices = [0.05, 0.1, 0.2]
        seed_hops_choices = [3]
        mutual_knn_choices = [False]
        snn_density_choices = [False]
        ks_merge_fraction_choices = [0.3, 0.4, 0.5]
    elif reduced_search:
        k_frac_choices = [2.0, 2.6, 3.3, 3.9]
        min_density_choices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        reg_thr_choices = [0.5, 0.7, 1.0, 1.1]
        merge_adj_choices = [True]
        ref_sigma_choices = [2.0, 2.5, 3.0]
        mcs_frac_choices = [0.15, 0.5, 1.0]
        centroid_dist_choices = [1.0, 2.0, 3.0, 5.0]
        ks_alpha_choices = [0.01, 0.05, 0.1, 0.2]
        seed_hops_choices = [2, 3]
        mutual_knn_choices = [False]
        snn_density_choices = [False]
        ks_merge_fraction_choices = [0.3, 0.4, 0.5, 0.6]
    else:
        k_frac_choices = [0.4, 0.5, 0.7, 0.9, 1.3, 2.0, 2.6, 3.3, 3.9]
        min_density_choices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        reg_thr_choices = [0.5, 0.7, 1.0, 1.1]
        merge_adj_choices = [True, False]
        ref_sigma_choices = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        mcs_frac_choices = [0.15, 0.5, 1.0]
        centroid_dist_choices = [1.0, 2.0, 3.0, 5.0]
        ks_alpha_choices = [0.01, 0.05, 0.1, 0.2]
        seed_hops_choices = [1, 2, 3]
        mutual_knn_choices = [True, False]
        snn_density_choices = [True, False]
        ks_merge_fraction_choices = [0.3, 0.4, 0.5, 0.6]

    # Pre-generate ALL parameter sets to preserve random seed sequence
    all_params = []
    for _ in range(n_trials):
        k_frac = rnd.choice(k_frac_choices)
        mcs_frac = rnd.choice(mcs_frac_choices)
        # Convert fractions → counts for this n (used during tuning)
        k_count = max(3, min(round(k_frac * log2_n), n_samples - 1))
        mcs_count = max(3, round(mcs_frac * sqrt_n))
        params = {
            'k_neighbors': min(k_count, max_k),  # clamp to precomputed cache
            'k_neighbors_frac': k_frac,
            'min_density': rnd.choice(min_density_choices),
            'regular_threshold_factor': rnd.choice(reg_thr_choices),
            'merge_adjacent': rnd.choice(merge_adj_choices),
            'refinement_sigma': rnd.choice(ref_sigma_choices),
            'min_cluster_size': mcs_count,
            'min_cluster_size_frac': mcs_frac,
            'centroid_distance_threshold': rnd.choice(centroid_dist_choices),
            'ks_test_alpha': rnd.choice(ks_alpha_choices),
            'seed_exclusion_hops': rnd.choice(seed_hops_choices),
            'use_mutual_knn': rnd.choice(mutual_knn_choices),
            'use_shared_neighbor_density': rnd.choice(snn_density_choices),
            'ks_merge_fraction': rnd.choice(ks_merge_fraction_choices),
        }
        all_params.append(params)

    # Determine effective parallelism
    # Skip parallel overhead for small/medium datasets
    # Parallel only helps when per-trial cost is high (n >= 5000)
    try:
        from joblib import Parallel, delayed, cpu_count
        eff_jobs = cpu_count() if n_jobs == -1 else max(n_jobs, 1)
        if eff_jobs is None:
            eff_jobs = 1
        # Auto: only parallelize large datasets where overhead is justified
        if n_jobs == -1 and n_samples < 5000:
            eff_jobs = 1
        use_parallel = eff_jobs > 1
    except ImportError:
        eff_jobs = 1
        use_parallel = False

    if verbose:
        if use_parallel:
            print(f"    Using {eff_jobs} parallel workers")
        else:
            print(f"    Sequential mode")

    best_score = -1
    best_labels = None
    best_params = None
    trial_log = []  # collect (score, params, n_clusters) for analysis

    if use_parallel:
        # PARALLEL: submit ALL trials in one Parallel() call to minimize overhead.
        # One call = one pool startup. No batching overhead.
        all_results = Parallel(n_jobs=eff_jobs, backend='loky')(
            delayed(_run_single_trial)(X, y_true, p, precomputed, expected_k)
            for p in all_params
        )
        for res in all_results:
            if res is not None:
                tuning_score, labels, params, n_clusters = res
                if return_all_trials:
                    trial_log.append((tuning_score, params.copy(), n_clusters))
                if tuning_score > best_score:
                    best_score = tuning_score
                    best_labels = labels.copy()
                    best_params = params.copy()
        if verbose:
            print(f"    All {n_trials} trials complete: best SCOPE={best_score:.3f}")
    else:
        # SEQUENTIAL: batch execution with early stopping
        trials_since_improvement = 0
        batch_size = 50
        for batch_start in range(0, n_trials, batch_size):
            batch_end = min(batch_start + batch_size, n_trials)
            batch_params = all_params[batch_start:batch_end]

            batch_results = [
                _run_single_trial(X, y_true, p, precomputed, expected_k)
                for p in batch_params
            ]

            improved = False
            for res in batch_results:
                if res is not None:
                    tuning_score, labels, params, n_clusters = res
                    if return_all_trials:
                        trial_log.append((tuning_score, params.copy(), n_clusters))
                    if tuning_score > best_score:
                        best_score = tuning_score
                        best_labels = labels.copy()
                        best_params = params.copy()
                        improved = True

            if improved:
                trials_since_improvement = 0
            else:
                trials_since_improvement += len(batch_params)

            if verbose:
                print(f"    Trials {batch_start+1}-{batch_end}/{n_trials}: "
                      f"best SCOPE={best_score:.3f}")

            if patience is not None and trials_since_improvement >= patience:
                if verbose:
                    print(f"    Early stop at trial {batch_end}/{n_trials} "
                          f"(no improvement in {trials_since_improvement} trials)")
                break

    if best_labels is None:
        best_labels = np.full(len(X), -1, dtype=int)
        best_params = {}
        best_score = 0.0

    # === GREEDY POST-MERGE (anti-fragmentation) ===
    if best_labels is not None and len(set(best_labels) - {-1}) > 1:
        best_k = best_params.get('k_neighbors', 15)
        knn_idx = precomputed[1][:, :best_k]
        k_before = len(set(best_labels) - {-1})
        # Allow enough rounds for large k to converge (cheap with sampling)
        post_max_rounds = max(15, k_before)
        if verbose:
            print(f"    Post-merge starting (k={k_before}, max_rounds={post_max_rounds})...")
        merged_labels, n_merges = _greedy_post_merge(
            X, y_true, best_labels, knn_idx, max_rounds=post_max_rounds, verbose=verbose
        )
        if n_merges > 0:
            try:
                merged_result = compute_dice_metrics(X, y_true, merged_labels, verbose=False)
                merged_score = merged_result['Overall_Score']
            except Exception:
                merged_score = 0.0
            if merged_score >= best_score:
                k_after = len(set(merged_labels) - {-1})
                best_labels = merged_labels
                best_score = merged_score
                best_params['post_merge_count'] = n_merges
                if verbose:
                    print(f"    Post-merge accepted: {n_merges} merges, "
                          f"k: {k_before} -> {k_after}, SCOPE={merged_score:.3f}")
            elif verbose:
                print(f"    Post-merge rejected (SCOPE would decrease)")
        elif verbose:
            print(f"    Post-merge: no beneficial merges found")

    if return_all_trials:
        return best_labels, best_params, best_score, trial_log
    return best_labels, best_params, best_score


print("[OK] AdaBoxGraph engine loaded (optimized: sparse matrix + numpy vectorization)")
