"""Smoke tests for the low-D track, asserting the SLCD-paper spec behavior."""

import warnings

import numpy as np
from sklearn.datasets import make_blobs, make_moons

from scml.lowd import AdaBox, SLCD, scope_score, scope_report

warnings.filterwarnings("ignore")


def _blobs(n=600, centers=4, std=1.0, seed=0):
    return make_blobs(n_samples=n, centers=centers, cluster_std=std,
                      random_state=seed)


def test_small_data_tune_uses_gs_and_absolute_density():
    # Spec: < 1k points -> Grid Search, absolute density (no SLCD).
    X, y = _blobs(n=600)
    m = AdaBox().tune(X, y)
    assert m.best_params_.get("use_relative_density") is False
    assert scope_score(X, y, m.labels_) > 0.85


def test_small_data_tune_beats_raw_engine():
    X, y = _blobs(n=600)
    raw = AdaBox(n_boxes=20, min_density=4).fit(X)
    tuned = AdaBox().tune(X, y)
    assert scope_score(X, y, tuned.labels_) > scope_score(X, y, raw.labels_)


def test_tune_recovers_moons():
    X, y = make_moons(n_samples=600, noise=0.06, random_state=0)
    m = AdaBox().tune(X, y)
    assert scope_score(X, y, m.labels_) > 0.9


def test_slcd_small_big_uses_rs_relative_and_size_defaults():
    # Spec: SLCD uses Random Search + relative density; trials/stages by size.
    X, y = _blobs(n=8000, centers=5)
    s = SLCD(sample_size=400).calibrate(X, y)
    assert s.best_params_.get("use_relative_density") is True
    assert s.n_trials_ == 100      # < 20k
    assert s.cascade_stages_ == 1  # < 20k


def test_slcd_large_size_defaults():
    X, y = _blobs(n=25000, centers=5)
    s = SLCD(sample_size=400).calibrate(X, y)
    assert s.n_trials_ == 200      # > 20k
    assert s.cascade_stages_ == 2  # > 20k


def test_slcd_calibrate_then_deploy_scales():
    X, y = _blobs(n=8000, centers=5)
    s = SLCD(sample_size=400)
    labels = s.fit_predict(X, y)
    assert s.best_params_ is not None
    assert scope_score(X, y, labels) > 0.7


def test_scope_report_keys_and_range():
    X, y = _blobs(n=600)
    labels = AdaBox().tune(X, y).labels_
    assert 0.0 <= scope_score(X, y, labels) <= 1.0
    rep = scope_report(X, y, labels)
    for key in ["Core_Purity", "Boundary_Recall", "Cluster_Precision",
                "Noise_F1", "Cluster_Count_Accuracy", "Overall_Score"]:
        assert key in rep


def test_raw_fit_requires_2d():
    X = np.random.rand(100, 3)
    try:
        AdaBox().fit(X)
    except ValueError:
        return
    raise AssertionError("raw fit should reject non-2D input")


def test_default_sample_size_tiers_and_cluster_floor():
    from scml.lowd import default_sample_size
    assert default_sample_size(10_000) == 200
    assert default_sample_size(100_000) == 800
    assert default_sample_size(1_000_000) == 1_500
    assert default_sample_size(5_000_000) == 5_000
    # cluster-aware floor kicks in above 5 clusters
    assert default_sample_size(12_000, n_clusters=8) == 640
    assert default_sample_size(12_000, n_clusters=25) == 2_000
    # never exceeds dataset size
    assert default_sample_size(300, n_clusters=25) == 300


def test_slcd_records_stage_history_and_nonoverlap():
    X, y = make_blobs(n_samples=30000, centers=5, cluster_std=1.0, random_state=2)
    s = SLCD(); s.calibrate(X, y)
    assert s.cascade_stages_ == 2
    assert len(s.stage_history_) == 2
    # stage 2 sample is larger than stage 1 (cascade grows)
    assert s.stage_history_[1]["sample_size"] >= s.stage_history_[0]["sample_size"]


def test_slcd_sample_size_is_cluster_aware():
    X, y = make_blobs(n_samples=15000, centers=8, cluster_std=1.0, random_state=1)
    s = SLCD(); s.calibrate(X, y)
    assert s.sample_size_ == 640   # 80 * 8


def test_tune_routes_direct_gs_up_to_5000():
    # Datasets <= 5000 points use direct full-data GS + absolute density,
    # matching the published benchmark protocol (validated by exact
    # reproduction of original results).
    X, y = make_blobs(n_samples=2000, centers=4, cluster_std=1.0, random_state=3)
    m = AdaBox().tune(X, y)
    assert m.best_params_.get("use_relative_density") is False


def test_slcd_history_records_ari():
    X, y = make_blobs(n_samples=30000, centers=5, cluster_std=1.0, random_state=2)
    s = SLCD(); s.calibrate(X, y)
    assert "ari" in s.stage_history_[0]


# ---------------------------------------------------------------------------
# Dataset preparation and benchmarking
# ---------------------------------------------------------------------------

def test_prepare_handles_messy_dataframe():
    import pandas as pd
    from scml.lowd import prepare_dataset
    X, y = _blobs(n=300, centers=3)
    df = pd.DataFrame(X, columns=["x", "y"])
    df["extra"] = np.random.rand(len(df))          # third feature -> PCA
    df["sample_id"] = [f"S{i}" for i in range(len(df))]  # junk text column
    df["class"] = [["a", "b", "c"][v] for v in y]  # string labels
    df.loc[0:4, "extra"] = np.nan                  # missing values
    Xp, yp, rep = prepare_dataset(df, verbose=False)
    assert Xp.shape[1] == 2
    assert len(Xp) == len(yp) < len(df)            # NaN rows dropped
    assert rep.label_name == "class"
    assert rep.n_clusters == 3


def test_prepare_does_not_mistake_y_coordinate_for_labels():
    # A column literally named 'y' that holds coordinates must not be
    # accepted as ground truth when a real label column exists.
    import pandas as pd
    from scml.lowd import prepare_dataset
    X, y = _blobs(n=300, centers=3)
    df = pd.DataFrame(X, columns=["x", "y"])
    df["extra"] = np.random.rand(len(df))
    df["label"] = y
    _, _, rep = prepare_dataset(df, verbose=False)
    assert rep.label_name == "label"


def test_prepare_rejects_missing_label_column():
    import pandas as pd
    from scml.lowd import prepare_dataset
    X, _ = _blobs(n=200, centers=3)
    df = pd.DataFrame(X, columns=["f1", "f2"])
    try:
        prepare_dataset(df, verbose=False)
    except ValueError:
        return
    raise AssertionError("should refuse a dataset with no label column")


def test_compare_algorithms_ranks_adabox_first_on_clean_blobs():
    from scml.lowd import compare_algorithms
    X, y = _blobs(n=400, centers=4, std=0.9, seed=1)
    results, preds = compare_algorithms(
        X, y, algorithms=("AdaBox", "DBSCAN"), verbose=False)
    assert set(results["Algorithm"]) == {"AdaBox", "DBSCAN"}
    assert "AdaBox" in preds
    # results are sorted by SCOPE, best first
    assert results.iloc[0]["SCOPE_Overall"] >= results.iloc[1]["SCOPE_Overall"]
    for col in ["ARI", "SCOPE_Overall", "K_found", "Time_s"]:
        assert col in results.columns


def test_prepare_recognizes_float_typed_noise_label():
    # A CSV with a mixed int/-1 label column loads as float64 in pandas
    # (-1 becomes -1.0). Noise must still be recognised as -1, not treated as
    # a third real cluster. Regression test for a real bug found while
    # reproducing the RQ1 CoreRing_Noise30 result.
    import pandas as pd
    from scml.lowd import prepare_dataset
    X, y = _blobs(n=200, centers=2)
    y_with_float_noise = y.astype(float)
    y_with_float_noise[:20] = -1.0
    df = pd.DataFrame(X, columns=["x", "y"])
    df["label"] = y_with_float_noise   # dtype float64, values like -1.0
    _, y_enc, rep = prepare_dataset(df, verbose=False)
    assert rep.n_clusters == 2          # not 3
    assert (y_enc == -1).sum() == 20
