"""Tests for the high-D track: AdaGraph, Graph-SCOPE, and high-D SLCD."""

import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from scml.highd import (AdaGraph, SLCD, build_knn_graph, compare_k_selection,
                        compute_graph_scope, default_sample_size,
                        graph_scope_report, graph_scope_score)
from scml.lowd import scope_score

warnings.filterwarnings("ignore")


def _hd_blobs(n=800, d=25, centers=5, std=2.5, seed=0):
    return make_blobs(n_samples=n, n_features=d, centers=centers,
                      cluster_std=std, random_state=seed)


def test_adagraph_clusters_in_native_dimensions():
    # No PCA, no projection: AdaGraph takes the raw high-D array.
    X, y = _hd_blobs(d=30)
    labels = AdaGraph().fit_predict(X)
    assert labels.shape == (len(X),)
    assert len(set(labels[labels >= 0])) >= 2


def test_adagraph_tune_beats_untuned():
    X, y = _hd_blobs()
    raw = AdaGraph().fit_predict(X)
    tuned = AdaGraph().tune(X, y, n_trials=40, n_jobs=1, patience=None)
    assert scope_score(X, y, tuned.labels_) >= scope_score(X, y, raw)
    assert tuned.best_params_ is not None


def test_graph_scope_is_unsupervised_and_peaks_at_true_k():
    # Graph-SCOPE never sees y; it should still favour the true k.
    X, y = _hd_blobs(n=1200, d=25, centers=6)
    knn, _ = build_knn_graph(X, k=15)
    scores = {}
    for k in [3, 4, 6, 9, 12]:
        lab = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(X)
        scores[k] = compute_graph_scope(knn, lab)[0]
    assert max(scores, key=scores.get) == 6


def test_graph_scope_report_has_five_components():
    X, y = _hd_blobs()
    labels = KMeans(n_clusters=5, n_init=5, random_state=0).fit_predict(X)
    rep = graph_scope_report(X, labels)
    for key in ["c1_modularity", "c2_boundary", "c3_consistency",
                "c4_noise", "c5_balance", "overall"]:
        assert key in rep
    assert 0.0 <= rep["overall"] <= 1.0


def test_graph_scope_accepts_optional_densities():
    # The density-based C4 path must run and stay in range.
    X, y = _hd_blobs()
    knn, dist = build_knn_graph(X, k=15)
    labels = KMeans(n_clusters=5, n_init=5, random_state=0).fit_predict(X)
    rd = 1.0 / (dist.mean(axis=1) + 1e-12)
    s = compute_graph_scope(knn, labels, relative_densities=rd)[0]
    assert 0.0 <= s <= 1.0


def test_graph_scope_precomputed_graph_matches():
    X, y = _hd_blobs()
    labels = KMeans(n_clusters=5, n_init=5, random_state=0).fit_predict(X)
    knn, _ = build_knn_graph(X, k=15)
    a = graph_scope_score(X, labels)
    b = graph_scope_score(knn, labels, precomputed_graph=True)
    assert abs(a - b) < 1e-12


def test_highd_slcd_never_tunes_full_dataset():
    X, y = _hd_blobs(n=6000, d=30, centers=6)
    s = SLCD(n_trials=40, n_jobs=1, expected_k=6)
    labels = s.fit_predict(X, y)
    # the defining invariant: learning happened on a strict subset
    assert s.sample_size_ < len(X)
    assert len(s.sample_indices_) == s.sample_size_
    assert labels.shape == (len(X),)
    assert scope_score(X, y, labels) > 0.5


def test_default_sample_size_tiers():
    assert default_sample_size(1_500) == 1_500      # cluster directly
    assert default_sample_size(10_000) == 1_000
    assert default_sample_size(50_000) == 2_000
    assert default_sample_size(500_000) == 5_000


def test_compare_k_selection_runs_negative_control():
    X, y = _hd_blobs(n=900, d=20, centers=5)
    res = compare_k_selection(X, y, k_max=8, verbose=False)
    assert "Graph-SCOPE" in set(res.Signal)
    assert "Silhouette" in set(res.Signal)
    nc = res[res.Signal.str.startswith("NEGATIVE")]
    assert len(nc) == 1
    # a shuffled clustering must score ~0 ARI or the harness is broken
    assert abs(float(nc.iloc[0]["ARI"])) < 0.05
