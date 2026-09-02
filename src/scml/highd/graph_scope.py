"""
Graph-SCOPE: unsupervised structural validation index for clusterings.

Graph-SCOPE judges a clustering from the topology of a k-nearest-neighbour
graph alone -- **no ground truth required**. That makes it usable where
supervised SCOPE cannot go: choosing the number of clusters, comparing
candidate clusterings, and driving hyperparameter search on unlabelled data.
Its natural comparison is with Silhouette, not with ARI.

It decomposes structure into five components:

  C1  Graph Modularity        60%   Reichardt-Bornholdt with resolution
                                    gamma=1.5, which avoids the k=2 resolution
                                    collapse of standard Newman-Girvan
  C2  Boundary Sharpness      10%   cross-cluster edges from boundary points
  C3  Internal Consistency    20%   uniform within-cluster cohesion
  C4  Noise Legitimacy         5%   are noise points genuinely marginal?
  C5  Partition Balance        5%   normalised entropy of cluster sizes

C4 can be measured two ways, which is the only difference between the
implementations in the author's notebooks:

  * **graph cohesion** (default) -- noise points should have few same-cluster
    neighbours. Needs only the graph and the labels, so it works on the output
    of *any* clustering algorithm.
  * **relative density** -- noise points should sit in low-density regions.
    More direct, but ``relative_densities`` is an AdaGraph internal, so this
    mode is only available when clustering with AdaGraph.

Both give identical scores whenever a clustering marks no noise, and differ
only within a 5%-weight component otherwise.

IMPORTANT -- signal, not judge. Graph-SCOPE is a *selection* signal. Using it
to both choose a clustering and then to declare that clustering good is
circular. Judge with supervised SCOPE or ARI against held-out labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(X, k=15):
    """Build the kNN graph used by Graph-SCOPE.

    Returns
    -------
    indices : ndarray (n, k)
        Neighbour indices, self excluded.
    distances : ndarray (n, k)
        Corresponding distances.
    """
    X = np.asarray(X, dtype=float)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean",
                            n_jobs=-1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    # column 0 is the point itself
    return indices[:, 1:], distances[:, 1:]


def compute_graph_scope(knn_indices, labels, gamma=1.5,
                        relative_densities=None):
    """Compute Graph-SCOPE and its five components.

    This is the author's validated v3 implementation, unchanged, with C4
    extended to accept an optional density criterion.

    Parameters
    ----------
    knn_indices : ndarray (n, k)
        Neighbour indices from :func:`build_knn_graph`.
    labels : ndarray (n,)
        Cluster labels; ``-1`` marks noise.
    gamma : float, default=1.5
        Modularity resolution. 1.5 avoids the k=2 resolution collapse of
        standard Newman-Girvan modularity.
    relative_densities : ndarray (n,), optional
        Per-point relative density. When supplied, C4 uses the density
        criterion; otherwise it uses graph cohesion (works for any algorithm).

    Returns
    -------
    (score, components) : (float, dict)
    """
    labels = np.asarray(labels)
    non_noise_mask = labels != -1
    n_non_noise    = int(non_noise_mask.sum())
    if n_non_noise < 4:
        return 0.0, {}

    cluster_ids = sorted(set(labels[non_noise_mask].tolist()))
    n_clusters  = len(cluster_ids)
    if n_clusters < 2:
        return 0.0, {}

    nn_idx          = np.where(non_noise_mask)[0]
    nbr_lbl         = labels[knn_indices[nn_idx]]
    nbr_valid       = nbr_lbl != -1
    intra           = nbr_valid & (nbr_lbl == labels[nn_idx, np.newaxis])
    is_cross        = nbr_valid & (nbr_lbl != labels[nn_idx, np.newaxis])
    m_valid         = int(nbr_valid.sum())
    cross_count     = int(is_cross.sum())
    valid_per_pt    = np.maximum(nbr_valid.sum(axis=1), 1).astype(np.float64)
    per_pt_cohesion = intra.sum(axis=1).astype(np.float64) / valid_per_pt

    # C1 — Reichardt-Bornholdt modularity
    if m_valid == 0:
        c1 = 0.0
    else:
        Q = 0.0
        for cid in cluster_ids:
            c_mask = (labels[nn_idx] == cid)
            e_c    = float(intra[c_mask].sum()) / m_valid
            a_c    = float(c_mask.sum()) / n_non_noise
            Q     += (e_c - gamma * a_c ** 2)
        c1 = float(np.clip(Q, 0.0, 1.0))

    # C2 — Boundary Sharpness
    if cross_count == 0:
        c2 = 1.0
    else:
        median_coh   = float(np.median(per_pt_cohesion))
        is_src_bndry = (per_pt_cohesion < median_coh)[:, np.newaxis]
        c2 = float((is_cross & is_src_bndry).sum()) / cross_count

    # C3 — Internal Consistency
    covs = []
    for cid in cluster_ids:
        c_mask = (labels[nn_idx] == cid)
        fracs  = per_pt_cohesion[c_mask]
        if len(fracs) < 2:
            covs.append(0.0)
            continue
        cov = float(fracs.std()) / (float(fracs.mean()) + 1e-10)
        covs.append(cov)
    c3 = max(0.0, 1.0 - float(np.mean(covs)))

    # C4 — Noise Legitimacy
    noise_mask = labels == -1
    n_noise    = int(noise_mask.sum())
    if n_noise == 0:
        c4 = 1.0
    elif relative_densities is not None:
        rd = np.asarray(relative_densities, dtype=float)
        median_rd = float(np.median(rd[non_noise_mask]))
        c4 = float((rd[noise_mask] < median_rd).mean())
    else:
        noise_nbr_lbl = labels[knn_indices[np.where(noise_mask)[0]]]
        noise_valid   = noise_nbr_lbl != -1
        noise_coh     = noise_valid.sum(axis=1).astype(np.float64) / np.maximum(
            noise_valid.sum(axis=1), 1)
        median_coh_nn = float(np.median(per_pt_cohesion))
        c4 = float((noise_coh < median_coh_nn).mean())

    # C5 — Partition Balance
    sizes    = np.array([(labels == cid).sum() for cid in cluster_ids], dtype=np.float64)
    p        = sizes / sizes.sum()
    entropy  = -float(np.sum(p * np.log(p + 1e-15)))
    max_entr = np.log(n_clusters)
    c5 = entropy / max_entr if max_entr > 1e-10 else 0.0

    score = 0.60 * c1 + 0.10 * c2 + 0.20 * c3 + 0.05 * c4 + 0.05 * c5
    components = {
        'c1_modularity':  round(float(c1), 4),
        'c2_boundary':    round(float(c2), 4),
        'c3_consistency': round(float(c3), 4),
        'c4_noise':       round(float(c4), 4),
        'c5_balance':     round(float(c5), 4),
        'overall':        round(float(score), 4),
    }
    return float(score), components


def graph_scope_score(X_or_knn, labels, k=15, gamma=1.5,
                      relative_densities=None, precomputed_graph=False):
    """Overall Graph-SCOPE in [0, 1] (higher is better).

    Parameters
    ----------
    X_or_knn : ndarray
        Either the data ``X`` (n_samples, n_features), or precomputed
        ``knn_indices`` when ``precomputed_graph=True``.
    labels : array-like (n,)
        Cluster labels; ``-1`` is noise.
    k : int, default=15
        Neighbours used when building the graph from ``X``.
    gamma : float, default=1.5
        Modularity resolution.
    relative_densities : array-like, optional
        Enables the density-based C4 (AdaGraph only).
    precomputed_graph : bool, default=False
        Set True when passing ``knn_indices`` directly -- useful when scoring
        many clusterings of the same data, since the graph is built once.
    """
    knn = (np.asarray(X_or_knn) if precomputed_graph
           else build_knn_graph(X_or_knn, k=k)[0])
    return compute_graph_scope(knn, labels, gamma=gamma,
                               relative_densities=relative_densities)[0]


def graph_scope_report(X_or_knn, labels, k=15, gamma=1.5,
                       relative_densities=None, precomputed_graph=False):
    """Full five-component Graph-SCOPE breakdown as a dict."""
    knn = (np.asarray(X_or_knn) if precomputed_graph
           else build_knn_graph(X_or_knn, k=k)[0])
    return compute_graph_scope(knn, labels, gamma=gamma,
                               relative_densities=relative_densities)[1]
