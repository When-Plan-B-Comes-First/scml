"""
One-command comparison of AdaBox against baseline clustering algorithms.

Hand it a dataset and it will:
  1. clean and validate the data (see :mod:`scml.lowd.prepare`),
  2. tune AdaBox with its three-phase calibration,
  3. tune DBSCAN, OPTICS and HDBSCAN by grid search on the *same* objective,
  4. print a results table and draw the comparison plots.

Every algorithm is tuned to maximise the same metric, so no method is
handicapped by being left at default settings.

Quick use:

>>> from scml.lowd import benchmark_dataset
>>> results = benchmark_dataset("my_data.csv")
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from .adabox import AdaBox
from .slcd import SLCD
from .baselines import HAS_HDBSCAN, optimize_dbscan, optimize_hdbscan, optimize_optics
from .metrics import scope_report
from .prepare import prepare_dataset

ALGO_COLORS = {"AdaBox": "blue", "DBSCAN": "green",
               "OPTICS": "red", "HDBSCAN": "purple"}

METRIC_COLUMNS = ["ARI", "SCOPE_Overall", "Core_Purity", "Boundary_Recall",
                  "Cluster_Precision", "Noise_F1", "Count_Accuracy"]


def _row(name, X, y_true, y_pred, params, elapsed):
    """Build one results row for an algorithm."""
    from sklearn.metrics import adjusted_rand_score
    m = scope_report(X, y_true, y_pred)
    return {
        "Algorithm": name,
        "ARI": adjusted_rand_score(y_true, y_pred),
        "SCOPE_Overall": m["Overall_Score"],
        "Core_Purity": m["Core_Purity"],
        "Boundary_Recall": m["Boundary_Recall"],
        "Cluster_Precision": m["Cluster_Precision"],
        "Noise_F1": m["Noise_F1"],
        "Count_Accuracy": m["Cluster_Count_Accuracy"],
        "K_found": int(len(set(y_pred[y_pred >= 0]))),
        "Time_s": round(elapsed, 1),
        "Params": str(params),
    }


def compare_algorithms(X, y_true, algorithms=("AdaBox", "DBSCAN", "OPTICS", "HDBSCAN"),
                       baseline_max_seconds=None, verbose=True):
    """Run and score each algorithm on one prepared dataset.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
    y_true : ndarray of shape (n_samples,)
        Ground-truth labels, ``-1`` for noise.
    algorithms : tuple of str
        Which methods to run.
    verbose : bool
        Print progress as each algorithm finishes.

    Returns
    -------
    results : pandas.DataFrame
        One row per algorithm, sorted by SCOPE.
    predictions : dict
        Algorithm name -> predicted labels.
    """
    import pandas as pd

    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true)
    rows, predictions = [], {}

    for name in algorithms:
        if name == "HDBSCAN" and not HAS_HDBSCAN:
            if verbose:
                print("  HDBSCAN backend not installed - skipping")
            continue
        if verbose:
            print(f"  tuning {name} ...", end="", flush=True)
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if name == "AdaBox":
                    # Up to 5,000 points: tune directly on the full data
                    # (exhaustive GS, the published protocol). Above that,
                    # use SLCD -- sample, calibrate on the sample, deploy the
                    # frozen parameters to the full dataset. That is what SLCD
                    # exists for, and it is dramatically faster than tuning
                    # the full dataset at scale.
                    if len(X) <= 5000:
                        model = AdaBox().tune(X, y_true)
                        y_pred, params = model.labels_, model.best_params_
                    else:
                        slcd = SLCD()
                        y_pred = slcd.fit_predict(X, y_true)
                        params = dict(slcd.best_params_)
                        params["slcd_sample_size"] = slcd.sample_size_
                        params["slcd_n_trials"] = slcd.n_trials_
                        params["slcd_cascade_stages"] = slcd.cascade_stages_
                elif name == "DBSCAN":
                    y_pred, params = optimize_dbscan(X, y_true, max_seconds=baseline_max_seconds)
                elif name == "OPTICS":
                    y_pred, params = optimize_optics(X, y_true, max_seconds=baseline_max_seconds)
                elif name == "HDBSCAN":
                    y_pred, params = optimize_hdbscan(X, y_true, max_seconds=baseline_max_seconds)
                else:
                    raise ValueError(f"Unknown algorithm: {name}")
            elapsed = time.time() - t0
            predictions[name] = y_pred
            rows.append(_row(name, X, y_true, y_pred, params, elapsed))
            if verbose:
                print(f" done in {elapsed:.1f}s "
                      f"(SCOPE {rows[-1]['SCOPE_Overall']:.3f})")
        except Exception as exc:  # keep going if one method fails
            if verbose:
                print(f" FAILED ({exc})")

    if not rows:
        raise RuntimeError("No algorithm produced a result.")

    results = pd.DataFrame(rows).sort_values(
        "SCOPE_Overall", ascending=False).reset_index(drop=True)
    return results, predictions


def print_results_table(results, dataset_name="dataset"):
    """Print the comparison table and a plain-language verdict."""
    display_cols = ["Algorithm"] + METRIC_COLUMNS + ["K_found", "Time_s"]
    table = results[display_cols].copy()
    for c in METRIC_COLUMNS:
        table[c] = table[c].map(lambda v: f"{v:.3f}")

    print("\n" + "=" * 78)
    print(f"CLUSTERING COMPARISON - {dataset_name}")
    print("=" * 78)
    print(table.to_string(index=False))
    print("=" * 78)

    best = results.iloc[0]
    print(f"\nBest by SCOPE: {best['Algorithm']} ({best['SCOPE_Overall']:.3f})")
    if len(results) > 1 and best["Algorithm"] == "AdaBox":
        runner = results.iloc[1]
        gain = best["SCOPE_Overall"] - runner["SCOPE_Overall"]
        pct = (gain / runner["SCOPE_Overall"] * 100) if runner["SCOPE_Overall"] else float("inf")
        print(f"AdaBox leads {runner['Algorithm']} by {gain:.3f} SCOPE "
              f"({pct:+.1f}%)")
    print()


def plot_metric_bars(results, dataset_name="dataset", save_path=None,
                     show=True):
    """Grouped bar chart of every metric, one group of bars per algorithm."""
    import matplotlib.pyplot as plt

    metrics = ["ARI", "SCOPE_Overall", "Core_Purity", "Boundary_Recall",
               "Cluster_Precision", "Noise_F1"]
    algorithms = list(results["Algorithm"])

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metrics))
    width = 0.8 / max(len(algorithms), 1)

    for idx, algo in enumerate(algorithms):
        offset = width * (idx - len(algorithms) / 2 + 0.5)
        scores = results.loc[results["Algorithm"] == algo, metrics].values[0]
        ax.bar(x + offset, scores, width, label=algo,
               color=ALGO_COLORS.get(algo, "gray"), alpha=0.8)

    ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(f"{dataset_name}: Metric Comparison",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_clustering_results(X, y_true, predictions, dataset_name="dataset",
                            save_path=None, show=True):
    """Ground truth beside each algorithm's clustering, with scores in titles."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_rand_score

    n = len(predictions)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
    if n + 1 == 1:
        axes = [axes]

    cluster_mask = y_true >= 0
    noise_mask = y_true == -1
    n_true = len(set(y_true[y_true >= 0]))

    ax = axes[0]
    if cluster_mask.any():
        ax.scatter(X[cluster_mask, 0], X[cluster_mask, 1],
                   c=y_true[cluster_mask], cmap="tab10", s=50, alpha=0.7,
                   edgecolors="k", linewidth=0.5)
    if noise_mask.any():
        ax.scatter(X[noise_mask, 0], X[noise_mask, 1], c="lightgray", s=50,
                   alpha=0.5, marker="x", label="Noise")
        ax.legend()
    ax.set_title(f"Ground Truth\n(K={n_true} clusters)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.3)

    for idx, (algo, y_pred) in enumerate(predictions.items(), start=1):
        ax = axes[idx]
        cm, nm = y_pred >= 0, y_pred == -1
        k = len(set(y_pred[y_pred >= 0]))
        if cm.any():
            ax.scatter(X[cm, 0], X[cm, 1], c=y_pred[cm], cmap="tab10", s=50,
                       alpha=0.7, edgecolors="k", linewidth=0.5)
        if nm.any():
            ax.scatter(X[nm, 0], X[nm, 1], c="lightgray", s=50, alpha=0.5,
                       marker="x", label="Noise")
            ax.legend()
        ari = adjusted_rand_score(y_true, y_pred)
        sc = scope_report(X, y_true, y_pred)["Overall_Score"]
        ax.set_title(f"{algo} (K={k})\nARI: {ari:.3f} | SCOPE: {sc:.3f}",
                     fontsize=12, fontweight="bold",
                     color=ALGO_COLORS.get(algo, "black"))
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Clustering Comparison: {dataset_name}",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def benchmark_dataset(data, y=None, label_column=None, feature_columns=None,
                      dataset_name=None,
                      algorithms=("AdaBox", "DBSCAN", "OPTICS", "HDBSCAN"),
                      standardize=True, reduce_to_2d=True,
                      baseline_max_seconds=None,
                      show_plots=True, save_dir=None, verbose=True):
    """Clean a dataset, run every algorithm, and report the comparison.

    This is the one-call entry point.

    Parameters
    ----------
    data : str | pandas.DataFrame | numpy.ndarray
        CSV/TSV path, DataFrame, or feature array (with ``y`` for labels).
    y : array-like, optional
        Labels, when ``data`` is a feature array.
    label_column, feature_columns :
        Passed through to :func:`scml.lowd.prepare_dataset`.
    dataset_name : str, optional
        Title used in the table and plots.
    algorithms : tuple of str
        Which methods to compare.
    standardize, reduce_to_2d : bool
        Cleaning options, see :func:`scml.lowd.prepare_dataset`.
    baseline_max_seconds : float, optional
        Time cap per baseline algorithm's grid search. Default None means no
        cap -- DBSCAN/OPTICS/HDBSCAN always get the full grid, exactly like
        AdaBox gets its full search, so the comparison stays fair. On very
        large datasets this can take a long time (the published benchmark
        took 30-40+ minutes per dataset on 20-50k points); set this only if
        you explicitly want a faster, capped run instead.
    show_plots : bool
        Display the plots (set False in scripts / CI).
    save_dir : str, optional
        Directory to write ``*_metrics.png``, ``*_clusters.png`` and
        ``*_results.csv``.
    verbose : bool
        Print the preparation report, progress and results table.

    Returns
    -------
    results : pandas.DataFrame
        One row per algorithm, sorted by SCOPE (best first).
    """
    if dataset_name is None:
        dataset_name = data if isinstance(data, str) else "dataset"
        if isinstance(dataset_name, str) and ("/" in dataset_name or "\\" in dataset_name):
            dataset_name = dataset_name.replace("\\", "/").split("/")[-1]

    X, y_true, _ = prepare_dataset(
        data, y=y, label_column=label_column, feature_columns=feature_columns,
        standardize=standardize, reduce_to_2d=reduce_to_2d, verbose=verbose)

    if verbose:
        print(f"\nRunning {len(algorithms)} algorithms on "
              f"{len(X)} points ...")
        if len(X) > 5000:
            print("  note: over 5,000 points - AdaBox uses SLCD (tunes on a "
                  "small sample, then deploys to the full dataset). "
                  "DBSCAN/OPTICS/HDBSCAN still run their FULL grid search on "
                  "the full data by default (baseline_max_seconds=None), for "
                  "a fair comparison -- this can take 20-40+ minutes per "
                  "dataset at 20-50k points. Pass baseline_max_seconds=N to "
                  "cap it if you want a faster, less exhaustive run.")

    results, predictions = compare_algorithms(
        X, y_true, algorithms, baseline_max_seconds=baseline_max_seconds,
        verbose=verbose)

    if verbose:
        print_results_table(results, dataset_name)

    safe = "".join(c if c.isalnum() or c in "-_" else "_"
                   for c in str(dataset_name))
    m_path = c_path = None
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        m_path = os.path.join(save_dir, f"{safe}_metrics.png")
        c_path = os.path.join(save_dir, f"{safe}_clusters.png")
        results.to_csv(os.path.join(save_dir, f"{safe}_results.csv"),
                       index=False)

    plot_metric_bars(results, dataset_name, m_path, show_plots)
    plot_clustering_results(X, y_true, predictions, dataset_name, c_path,
                            show_plots)

    if save_dir and verbose:
        print(f"Saved table and plots to {save_dir}/")

    return results
