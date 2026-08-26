"""
AdaBox core engine (low-D track).

Expansion-Adaptive Density-Based Box Clustering: grid-based density
estimation with adaptive iterative box expansion and Gaussian boundary
refinement. This is the engine; the public sklearn-style wrapper lives
in scml.lowd.adabox (class AdaBox).
"""
import numpy as np

# ======================================================
# COMPLETE ORIGINAL EADBBC2D and EADBBC2D_BR CODE
# ======================================================
# Extracted from: Fully_tested_and_working_notebook_for_the_3_datasets_groups_11_9_2025.ipynb
# This is the EXACT code that was performing well with two-stage tuning
# NO MODIFICATIONS from the original working version

from scipy.ndimage import gaussian_filter

class EADBBC2D:
    """
    Expansion-Adaptive Density-Based Box Clustering (base implementation).
    Uses grid-based density estimation with adaptive iterative expansion.
    """
    def __init__(self, n_boxes=20, min_density=3, min_growth_iterations=3,
                 initial_threshold_factor=0.6, regular_threshold_factor=1.0,
                 merge_adjacent=True, verbose=False,
                 centroid_distance_threshold=2.0, ks_test_alpha=0.05,
                 use_relative_density=True, relative_density_param=5.5,
                 global_weight=0.3, local_weight=0.7, min_boxes_for_local=5):
        """
        Parameters:
        -----------
        use_relative_density : bool, default=True (NEW MODE DEFAULT in this testing notebook)
            If True, uses relative density thresholds instead of absolute min_density.
            Set to False to test OLD mode behavior.
        relative_density_param : float, default=5.5
            Multiplier for density threshold (0.1-10.0 range).
            5.5 ≈ equivalent to min_density=3 for typical sparse data (avg_density ~0.5-0.6)
        global_weight : float, default=0.3
            Weight for global average density (n/b²) in combined threshold.
        local_weight : float, default=0.7
            Weight for local cluster density (n_c/b_c) in combined threshold.
        min_boxes_for_local : int, default=5
            Minimum cluster size (in boxes) before using local density.
        """
        self.n_boxes = int(n_boxes)
        self.min_density = int(min_density)
        self.min_growth_iterations = int(min_growth_iterations)
        self.initial_threshold_factor = float(initial_threshold_factor)
        self.regular_threshold_factor = float(regular_threshold_factor)
        self.merge_adjacent = bool(merge_adjacent)
        self.verbose = verbose
        self.centroid_distance_threshold = float(centroid_distance_threshold)
        self.ks_test_alpha = float(ks_test_alpha)
        
        # NEW: Relative density parameters
        self.use_relative_density = bool(use_relative_density)
        self.relative_density_param = float(relative_density_param)
        self.global_weight = float(global_weight)
        self.local_weight = float(local_weight)
        self.min_boxes_for_local = int(min_boxes_for_local)
        self.global_avg_density_ = None
        
        self.reset()

    def reset(self):
        self.X_ = None
        self.grid_bounds_ = None
        self.box_size_ = None
        self.density_map_ = None
        self.clusters_boxes_ = []
        self.final_clusters_boxes_ = []
        self.labels_ = None
        self.cluster_gd_ = []
        self.cluster_growth_sequences_ = []
        self._cluster_gd_at_grad = []
        self._cluster_growth_sequences_at_grad = []

    def _create_grid(self, X):
        if X.shape[1] != 2:
            raise ValueError("EADBBC2D expects 2D input")
        self.grid_bounds_ = {}
        self.box_size_ = {}
        eps = 1e-9
        for d in range(2):
            mn, mx = X[:, d].min(), X[:, d].max()
            pad = (mx - mn) * 0.05
            self.grid_bounds_[f'min_{d}'] = mn - pad
            self.grid_bounds_[f'max_{d}'] = mx + pad
            eff = self.grid_bounds_[f'max_{d}'] - self.grid_bounds_[f'min_{d}']
            self.box_size_[f'size_{d}'] = eff / self.n_boxes if eff > 0 else eps
        
        # NEW: Calculate global average density for relative density mode
        if self.use_relative_density:
            n_points = len(X)
            total_boxes = self.n_boxes * self.n_boxes
            self.global_avg_density_ = n_points / total_boxes if total_boxes > 0 else 1.0
            if self.verbose:
                print(f"Global average density: {self.global_avg_density_:.2f} points/box")

    def _calculate_density(self, X):
        self.density_map_ = np.zeros((self.n_boxes, self.n_boxes), dtype=int)
        for p in X:
            i = int((p[0] - self.grid_bounds_['min_0']) / self.box_size_['size_0'])
            j = int((p[1] - self.grid_bounds_['min_1']) / self.box_size_['size_1'])
            i = max(0, min(i, self.n_boxes-1))
            j = max(0, min(j, self.n_boxes-1))
            self.density_map_[i, j] += 1

    def _liberal_seeding(self, initial_threshold):
        candidates = []
        for i in range(self.n_boxes):
            for j in range(self.n_boxes):
                if self.density_map_[i, j] >= initial_threshold:
                    candidates.append(((i, j), int(self.density_map_[i, j])))
        candidates.sort(key=lambda x: (-x[1], x[0][0], x[0][1]))
        seeds = []
        occupied = set()

        for (i, j), dens in candidates:
            if (i, j) in occupied:
                continue
            seed = {
                'center': (i, j),
                'boxes': {(i, j)},
                'iteration_count': 0,
                'successful_growth_count': 0,
                'growth_sequence': [],
                'graduated_at': None,
                'active': True,
                'num_points': 0,  # NEW: Track points for relative density
            }
            seeds.append(seed)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.n_boxes and 0 <= nj < self.n_boxes:
                        occupied.add((ni, nj))
        return seeds

    def _neighbors8(self, i, j):
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.n_boxes and 0 <= nj < self.n_boxes:
                    yield (ni, nj)

    def _compute_dynamic_threshold(self, seed, base_factor):
        """
        Compute dynamic threshold for relative density mode.
        Uses weighted combination of global and local density.
        """
        if not self.use_relative_density:
            # Legacy mode: use absolute min_density
            return self.min_density * base_factor
        
        # Relative density mode
        num_boxes = len(seed['boxes'])
        
        if num_boxes < self.min_boxes_for_local:
            # Cold start: use global average only
            threshold = self.relative_density_param * self.global_avg_density_ * base_factor
        else:
            # Compute local cluster density
            num_points = sum(self.density_map_[i, j] for (i, j) in seed['boxes'])
            local_avg_density = num_points / num_boxes if num_boxes > 0 else self.global_avg_density_
            
            # Weighted combination
            combined_density = (self.global_weight * self.global_avg_density_ + 
                              self.local_weight * local_avg_density)
            threshold = self.relative_density_param * combined_density * base_factor
        
        return max(threshold, 1.0)  # Ensure threshold is at least 1

    def _attempt_growth_once(self, seed, threshold, used_boxes, added_this_iter):
        current = seed['boxes'].copy()
        candidates = set()

        for (i, j) in current:
            for (ni, nj) in self._neighbors8(i, j):
                if (ni, nj) not in current and (ni, nj) not in used_boxes and (ni, nj) not in added_this_iter:
                    candidates.add((ni, nj))

        newly_added = set()
        for (ci, cj) in candidates:
            if self.density_map_[ci, cj] >= threshold:
                seed['boxes'].add((ci, cj))
                newly_added.add((ci, cj))
        return newly_added

    def _iterative_growth(self, seeds, regular_threshold, max_iterations=50):
        active = [s for s in seeds if s['active']]
        used_boxes = set().union(*[s['boxes'] for s in active]) if active else set()
        final_clusters = []
        gd_at_grad_list = []
        seq_at_grad_list = []

        iteration = 1
        while active and iteration <= max_iterations:
            still = []
            added_this_iter_global = set()
            for seed in active:
                # NEW: Compute dynamic threshold per cluster if using relative density
                if self.use_relative_density:
                    current_threshold = self._compute_dynamic_threshold(seed, self.regular_threshold_factor)
                else:
                    current_threshold = regular_threshold
                
                added = self._attempt_growth_once(seed, current_threshold, used_boxes, added_this_iter_global)
                seed['iteration_count'] += 1
                delta = len(added)
                seed['growth_sequence'].append(delta)
                if delta > 0:
                    seed['successful_growth_count'] += 1
                    if seed['graduated_at'] is None and seed['successful_growth_count'] >= self.min_growth_iterations:
                        seed['graduated_at'] = len(seed['growth_sequence'])
                        final_clusters.append(seed['boxes'].copy())
                        gd_at_grad_list.append(seed['successful_growth_count'])
                        seq_at_grad_list.append(seed['growth_sequence'][:])
                    still.append(seed)
                    added_this_iter_global.update(added)
                else:
                    if seed['graduated_at'] is None:
                        seed['active'] = False
                    else:
                        seed['active'] = False
            used_boxes.update(added_this_iter_global)
            active = [s for s in still if s['active']]
            iteration += 1

        graduated_seeds = [s for s in seeds if s.get('graduated_at') is not None]
        full_gd_list = []
        full_seq_list = []
        for s in graduated_seeds:
            seq_full = s['growth_sequence'][:]
            full_gd_list.append(int(np.sum(np.array(seq_full) > 0)))
            full_seq_list.append(seq_full)

        self.clusters_boxes_ = final_clusters
        self.cluster_gd_ = full_gd_list
        self.cluster_growth_sequences_ = full_seq_list
        self._cluster_gd_at_grad = gd_at_grad_list
        self._cluster_growth_sequences_at_grad = seq_at_grad_list
        return final_clusters

    def _merge_adjacent_clusters(self, clusters, min_boundary_density=None):
        if not clusters:
            return []

        if min_boundary_density is None:
            min_boundary_density = self.min_density

        parent = list(range(len(clusters)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        box_to_cluster = {}
        for idx, c in enumerate(clusters):
            for box in c:
                box_to_cluster.setdefault(box, []).append(idx)

        offsets = [(di, dj) for di in [-1, 0, 1] for dj in [-1, 0, 1] if not (di == 0 and dj == 0)]
        seen_pairs = set()

        for i, c in enumerate(clusters):
            for (bi, bj) in c:
                for di, dj in offsets:
                    nb = (bi + di, bj + dj)
                    if nb in box_to_cluster:
                        for j in box_to_cluster[nb]:
                            if i != j:
                                a, b = sorted((i, j))
                                if (a, b) not in seen_pairs:
                                    seen_pairs.add((a, b))

                                    should_merge = self._should_merge_clusters(
                                        clusters[a], clusters[b], min_boundary_density
                                    )

                                    if should_merge:
                                        union(a, b)

        groups = {}
        for i in range(len(clusters)):
            r = find(i)
            groups.setdefault(r, set()).update(clusters[i])
        return list(groups.values())

    def _should_merge_clusters(self, cluster_a, cluster_b, min_boundary_density):
        from scipy.stats import ks_2samp

        offsets = [(di, dj) for di in [-1, 0, 1] for dj in [-1, 0, 1] if not (di == 0 and dj == 0)]

        boundary_pairs = []
        for (bi, bj) in cluster_a:
            for di, dj in offsets:
                neighbor = (bi + di, bj + dj)
                if neighbor in cluster_b:
                    boundary_pairs.append(((bi, bj), neighbor))

        if not boundary_pairs:
            return False

        boundary_densities = []
        for (box_a, box_b) in boundary_pairs:
            density_a = self.density_map_[box_a[0], box_a[1]]
            density_b = self.density_map_[box_b[0], box_b[1]]
            avg_density = (density_a + density_b) / 2.0
            boundary_densities.append(avg_density)

        avg_boundary_density = np.mean(boundary_densities)

        if avg_boundary_density < min_boundary_density:
            return False

        def get_points_in_cluster(cluster_boxes):
            points = []
            for idx, p in enumerate(self.X_):
                i = int((p[0] - self.grid_bounds_['min_0']) / self.box_size_['size_0'])
                j = int((p[1] - self.grid_bounds_['min_1']) / self.box_size_['size_1'])
                i = max(0, min(i, self.n_boxes - 1))
                j = max(0, min(j, self.n_boxes - 1))
                if (i, j) in cluster_boxes:
                    points.append(p)
            return np.array(points) if points else np.array([]).reshape(0, 2)

        points_a = get_points_in_cluster(cluster_a)
        points_b = get_points_in_cluster(cluster_b)

        if len(points_a) < 5 or len(points_b) < 5:
            return True

        centroid_a = points_a.mean(axis=0)
        centroid_b = points_b.mean(axis=0)
        distance = np.linalg.norm(centroid_a - centroid_b)

        std_a = points_a.std()
        std_b = points_b.std()
        avg_std = (std_a + std_b) / 2.0

        normalized_distance = distance / avg_std if avg_std > 1e-6 else 0.0

        if normalized_distance > self.centroid_distance_threshold:
            return False

        for dim in range(self.X_.shape[1]):
            try:
                stat, p_value = ks_2samp(points_a[:, dim], points_b[:, dim])

                if p_value < self.ks_test_alpha:
                    return False
            except:
                pass

        return True

    def fit(self, X):
        self.reset()
        self.X_ = X
        self._create_grid(X)
        self._calculate_density(X)

        # Compute initial thresholds
        if self.use_relative_density:
            init_thr = max(1.0, self.relative_density_param * self.global_avg_density_ * self.initial_threshold_factor)
            reg_thr = max(1.0, self.relative_density_param * self.global_avg_density_ * self.regular_threshold_factor)
            if self.verbose:
                print(f"Relative density mode: init_thr={init_thr:.2f}, reg_thr={reg_thr:.2f}")
        else:
            init_thr = self.min_density * self.initial_threshold_factor
            reg_thr = self.min_density * self.regular_threshold_factor

        seeds = self._liberal_seeding(init_thr)
        premerge = self._iterative_growth(seeds, reg_thr)
        clusters = premerge

        if self.merge_adjacent and len(clusters) > 1:
            clusters = self._merge_adjacent_clusters(clusters)

        self.final_clusters_boxes_ = clusters
        return self

    def predict_labels(self, X, min_cluster_size=1):
        labels = np.full(len(X), -1, dtype=int)

        box_to_cluster = {}
        for cid, boxes in enumerate(self.final_clusters_boxes_):
            for b in boxes:
                box_to_cluster[b] = cid

        tmp = np.full(len(X), -1, dtype=int)
        for idx, p in enumerate(X):
            i = int((p[0] - self.grid_bounds_['min_0']) / self.box_size_['size_0'])
            j = int((p[1] - self.grid_bounds_['min_1']) / self.box_size_['size_1'])
            i = max(0, min(i, self.n_boxes - 1))
            j = max(0, min(j, self.n_boxes - 1))
            b = (i, j)

            if b in box_to_cluster:
                tmp[idx] = box_to_cluster[b]

        if min_cluster_size <= 1:
            return tmp

        keep = {}
        for cid in set(tmp) - {-1}:
            if int(np.sum(tmp == cid)) >= min_cluster_size:
                keep[cid] = True

        mapping = {old: new for new, old in enumerate(sorted(keep.keys()))}
        for i in range(len(tmp)):
            labels[i] = mapping.get(tmp[i], -1)
        return labels

    def get_metrics(self):
        gd = self.cluster_gd_
        seq = self.cluster_growth_sequences_
        ab = []
        for s in seq:
            arr = np.array(s, dtype=float)
            m = arr.mean() if arr.size > 0 else 0.0
            ab.append(0.0 if (arr.size <= 1 or m == 0) else float(arr.std(ddof=0) / m))
        return gd, ab


class EADBBC2D_BR(EADBBC2D):
    """
    EADBBC with Gaussian smoothing boundary refinement.
    Extends base algorithm with post-processing refinement using smooth density fields.
    """

    def __init__(self, n_boxes=20, min_density=3, min_growth_iterations=3,
                 initial_threshold_factor=0.6, regular_threshold_factor=1.0,
                 merge_adjacent=True, boundary_refinement=True,
                 refinement_sigma=1.5, refinement_threshold=0.15, verbose=False,
                 centroid_distance_threshold=2.0, ks_test_alpha=0.05,
                 use_relative_density=False, relative_density_param=1.0,
                 global_weight=0.3, local_weight=0.7, min_boxes_for_local=5):
        super().__init__(n_boxes, min_density, min_growth_iterations,
                        initial_threshold_factor, regular_threshold_factor,
                        merge_adjacent, verbose,
                        centroid_distance_threshold, ks_test_alpha,
                        use_relative_density, relative_density_param,
                        global_weight, local_weight, min_boxes_for_local)
        self.boundary_refinement = boundary_refinement
        self.refinement_sigma = refinement_sigma
        self.refinement_threshold = refinement_threshold
        self.reclassified_points_ = []

    def _refine_boundaries_with_smoothing(self, X, initial_labels):
        refined_labels = initial_labels.copy()
        noise_indices = np.where(initial_labels == -1)[0]

        if len(noise_indices) == 0 or len(self.final_clusters_boxes_) == 0:
            return refined_labels

        cluster_fields = []
        for cluster_boxes in self.final_clusters_boxes_:
            cluster_density = np.zeros((self.n_boxes, self.n_boxes))
            for box in cluster_boxes:
                cluster_density[box] = self.density_map_[box]

            cluster_density_enhanced = np.power(cluster_density, 0.6)
            cluster_density_smooth = gaussian_filter(cluster_density_enhanced, sigma=self.refinement_sigma)

            if cluster_density_smooth.max() > 0:
                cluster_density_smooth = cluster_density_smooth / cluster_density_smooth.max()

            cluster_fields.append(cluster_density_smooth)

        self.reclassified_points_ = []
        for idx in noise_indices:
            point = X[idx]

            grid_i = (point[0] - self.grid_bounds_['min_0']) / self.box_size_['size_0']
            grid_j = (point[1] - self.grid_bounds_['min_1']) / self.box_size_['size_1']

            grid_i = max(0, min(grid_i, self.n_boxes - 1))
            grid_j = max(0, min(grid_j, self.n_boxes - 1))

            i0, j0 = int(np.floor(grid_i)), int(np.floor(grid_j))
            i1, j1 = min(i0 + 1, self.n_boxes - 1), min(j0 + 1, self.n_boxes - 1)

            di = grid_i - i0
            dj = grid_j - j0

            max_affinity = 0.0
            best_cluster = -1

            for cid, field in enumerate(cluster_fields):
                affinity = (
                    field[i0, j0] * (1 - di) * (1 - dj) +
                    field[i1, j0] * di * (1 - dj) +
                    field[i0, j1] * (1 - di) * dj +
                    field[i1, j1] * di * dj
                )

                if affinity > max_affinity:
                    max_affinity = affinity
                    best_cluster = cid

            if max_affinity > self.refinement_threshold:
                refined_labels[idx] = best_cluster
                self.reclassified_points_.append({
                    'point_idx': idx,
                    'from_cluster': -1,
                    'to_cluster': best_cluster,
                    'affinity': max_affinity
                })

        if self.verbose and len(self.reclassified_points_) > 0:
            print(f"  Boundary refinement: reclassified {len(self.reclassified_points_)} points")

        return refined_labels

    def predict_labels(self, X, min_cluster_size=1):
        initial_labels = super().predict_labels(X, min_cluster_size)

        if self.boundary_refinement and len(self.final_clusters_boxes_) > 0:
            refined_labels = self._refine_boundaries_with_smoothing(X, initial_labels)

            if min_cluster_size > 1:
                valid_clusters = {}
                for cluster_id in set(refined_labels) - {-1}:
                    cluster_size = int(np.sum(refined_labels == cluster_id))
                    if cluster_size >= min_cluster_size:
                        valid_clusters[cluster_id] = True

                final_labels = np.full(len(refined_labels), -1, dtype=int)
                if valid_clusters:
                    cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(valid_clusters.keys()))}
                    for idx in range(len(refined_labels)):
                        old_label = refined_labels[idx]
                        if old_label in cluster_mapping:
                            final_labels[idx] = cluster_mapping[old_label]

                # FILTER: Remove tiny clusters 
                filtered_labels = final_labels.copy()
                cluster_sizes = {}
                for cluster_id in set(filtered_labels) - {-1}:
                    cluster_sizes[cluster_id] = int(np.sum(filtered_labels == cluster_id))
                
                # Identify clusters to remove
                tiny_clusters = {cid for cid, size in cluster_sizes.items() if size < 7}
                
                # Move tiny cluster points to noise
                for idx in range(len(filtered_labels)):
                    if filtered_labels[idx] in tiny_clusters:
                        filtered_labels[idx] = -1
                
                # Remap remaining clusters to consecutive IDs
                remaining_clusters = sorted(set(filtered_labels) - {-1})
                if remaining_clusters:
                    final_mapping = {old_id: new_id for new_id, old_id in enumerate(remaining_clusters)}
                    for idx in range(len(filtered_labels)):
                        if filtered_labels[idx] != -1:
                            filtered_labels[idx] = final_mapping[filtered_labels[idx]]
                
                return filtered_labels
            else:
                # FILTER: Remove tiny clusters
                filtered_labels = refined_labels.copy()
                cluster_sizes = {}
                for cluster_id in set(filtered_labels) - {-1}:
                    cluster_sizes[cluster_id] = int(np.sum(filtered_labels == cluster_id))
                
                # Identify clusters to remove
                tiny_clusters = {cid for cid, size in cluster_sizes.items() if size < 7}
                
                # Move tiny cluster points to noise
                for idx in range(len(filtered_labels)):
                    if filtered_labels[idx] in tiny_clusters:
                        filtered_labels[idx] = -1
                
                # Remap remaining clusters to consecutive IDs
                remaining_clusters = sorted(set(filtered_labels) - {-1})
                if remaining_clusters:
                    final_mapping = {old_id: new_id for new_id, old_id in enumerate(remaining_clusters)}
                    for idx in range(len(filtered_labels)):
                        if filtered_labels[idx] != -1:
                            filtered_labels[idx] = final_mapping[filtered_labels[idx]]
                
                return filtered_labels
        else:
            if min_cluster_size > 1:
                base_labels = super().predict_labels(X, min_cluster_size)
            else:
                base_labels = initial_labels
            
            # FILTER: Remove tinyclusters
            filtered_labels = base_labels.copy()
            cluster_sizes = {}
            for cluster_id in set(filtered_labels) - {-1}:
                cluster_sizes[cluster_id] = int(np.sum(filtered_labels == cluster_id))
            
            # Identify clusters to remove
            tiny_clusters = {cid for cid, size in cluster_sizes.items() if size < 7}
            
            # Move tiny cluster points to noise
            for idx in range(len(filtered_labels)):
                if filtered_labels[idx] in tiny_clusters:
                    filtered_labels[idx] = -1
            
            # Remap remaining clusters to consecutive IDs
            remaining_clusters = sorted(set(filtered_labels) - {-1})
            if remaining_clusters:
                final_mapping = {old_id: new_id for new_id, old_id in enumerate(remaining_clusters)}
                for idx in range(len(filtered_labels)):
                    if filtered_labels[idx] != -1:
                        filtered_labels[idx] = final_mapping[filtered_labels[idx]]
            
            return filtered_labels

