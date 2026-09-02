#!/usr/bin/env python3
"""
reproduce/run_all.py -- regenerate the low-D track's headline results with a
single command:

    python reproduce/run_all.py

Two experiments, each run in the regime it is actually meant for. This split
matters: applying the wrong tool at the wrong scale is the fastest way to make
a good method look bad.

  1. SMALL DATA (<= 5,000 points): AdaBox tuned directly, no SLCD.
     Exhaustive Grid Search on the full dataset, compared against
     grid-searched DBSCAN. This is the protocol behind the published benchmark
     results, and it is what the documentation tells users to do at this
     scale. SLCD is deliberately NOT used here: it exists to make tuning
     tractable at scale, not to replace full tuning on data small enough to
     tune directly.

  2. LARGE DATA (30k-50k points): SLCD tuned on a ~500-point sample, then
     deployed to the whole dataset. This is where SLCD earns its keep --
     parameters learned from under 2% of the data and transferred to the rest.
     The transfer gap (sample score minus full-data score) is the number that
     matters.

Outputs land in reproduce/outputs/.
"""

from __future__ import annotations

import os
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from scml.lowd import AdaBox, SLCD, scope_report, scope_score
from scml.lowd.baselines import optimize_dbscan

warnings.filterwarnings("ignore")

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Experiment 1 -- small data, direct tuning (no SLCD)
# ---------------------------------------------------------------------------

def small_datasets():
    """Datasets under the 5,000-point direct-tuning threshold."""
    d = {}
    X, y = make_blobs(n_samples=900, centers=3, cluster_std=1.0,
                      random_state=RANDOM_STATE)
    d["blobs_3"] = (X, y)
    X, y = make_blobs(n_samples=1200, centers=5, cluster_std=1.2,
                      random_state=RANDOM_STATE)
    d["blobs_5"] = (X, y)
    X, y = make_blobs(n_samples=2000, centers=8, cluster_std=1.0,
                      random_state=RANDOM_STATE)
    d["blobs_8"] = (X, y)
    X, y = make_moons(n_samples=800, noise=0.06, random_state=RANDOM_STATE)
    d["moons"] = (X, y)
    return d


def run_small():
    print("=" * 78)
    print("EXPERIMENT 1 - small data (<= 5,000 points): AdaBox tuned directly")
    print("               No SLCD. Exhaustive Grid Search on the full dataset.")
    print("=" * 78)
    rows, components = [], []
    for name, (X, y) in small_datasets().items():
        t0 = time.time()
        labels = AdaBox().tune(X, y).labels_
        t_ada = time.time() - t0

        t0 = time.time()
        db_labels, _ = optimize_dbscan(X, y)
        t_db = time.time() - t0

        rep = scope_report(X, y, labels)
        rows.append({
            "dataset": name, "n_points": len(X),
            "n_true_clusters": rep["N_True_Clusters"],
            "AdaBox_k": rep["N_Pred_Clusters"],
            "AdaBox_SCOPE": round(rep["Overall_Score"], 3),
            "AdaBox_ARI": round(adjusted_rand_score(y, labels), 3),
            "DBSCAN_SCOPE": round(scope_score(X, y, db_labels), 3),
            "DBSCAN_ARI": round(adjusted_rand_score(y, db_labels), 3),
            "AdaBox_s": round(t_ada, 1), "DBSCAN_s": round(t_db, 1),
        })
        components.append({
            "dataset": name,
            "Core_Purity": rep["Core_Purity"],
            "Boundary_Recall": rep["Boundary_Recall"],
            "Cluster_Precision": rep["Cluster_Precision"],
            "Noise_F1": rep["Noise_F1"],
            "Cluster_Count_Accuracy": rep["Cluster_Count_Accuracy"],
        })
        print(f"  [{name:9s}] AdaBox SCOPE={rep['Overall_Score']:.3f} "
              f"(k={rep['N_Pred_Clusters']}/{rep['N_True_Clusters']})   "
              f"DBSCAN SCOPE={rows[-1]['DBSCAN_SCOPE']:.3f}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(components)


# ---------------------------------------------------------------------------
# Experiment 2 -- large data, SLCD transfer (where SLCD belongs)
# ---------------------------------------------------------------------------

def large_datasets():
    """Datasets in the 30k-50k range, where SLCD is the right tool."""
    d = {}
    X, y = make_blobs(n_samples=30000, centers=5, cluster_std=1.0,
                      random_state=RANDOM_STATE)
    d["blobs_5_30k"] = (X, y)
    X, y = make_blobs(n_samples=30000, centers=10, cluster_std=1.0,
                      random_state=RANDOM_STATE + 1)
    d["blobs_10_30k"] = (X, y)
    X, y = make_blobs(n_samples=50000, centers=8, cluster_std=1.2,
                      random_state=RANDOM_STATE + 2)
    d["blobs_8_50k"] = (X, y)
    return d


def run_large(sample_size=500):
    print("\n" + "=" * 78)
    print("EXPERIMENT 2 - large data (30k-50k points): SLCD sample-and-transfer")
    print(f"               Tuned on a {sample_size}-point sample, deployed to all.")
    print("=" * 78)
    rows = []
    for name, (X, y) in large_datasets().items():
        t0 = time.time()
        slcd = SLCD(sample_size=sample_size)
        labels = slcd.fit_predict(X, y)
        t_slcd = time.time() - t0

        full = scope_score(X, y, labels)
        rows.append({
            "dataset": name, "n_points": len(X),
            "n_true_clusters": int(len(set(y[y >= 0]))),
            "sample_size": slcd.sample_size_,
            "sample_pct": round(100 * slcd.sample_size_ / len(X), 2),
            "sample_SCOPE": round(slcd.calibration_score_, 3),
            "full_SCOPE": round(full, 3),
            "transfer_gap": round(slcd.calibration_score_ - full, 3),
            "full_ARI": round(adjusted_rand_score(y, labels), 3),
            "k_found": int(len(set(labels[labels >= 0]))),
            "time_s": round(t_slcd, 1),
        })
        print(f"  [{name:13s}] tuned on {slcd.sample_size_} pts "
              f"({rows[-1]['sample_pct']}% of data)  "
              f"sample={slcd.calibration_score_:.3f} -> full={full:.3f}  "
              f"gap={rows[-1]['transfer_gap']:+.3f}", flush=True)
    return pd.DataFrame(rows)


def main():
    small, components = run_small()
    large = run_large()

    print("\n" + "=" * 78)
    print("EXPERIMENT 1 - AdaBox vs DBSCAN on small data (direct tuning)")
    print("=" * 78)
    print(small.to_string(index=False))

    print("\n" + "=" * 78)
    print("EXPERIMENT 2 - SLCD transfer on large data")
    print("=" * 78)
    print(large.to_string(index=False))

    wins = int((small.AdaBox_SCOPE > small.DBSCAN_SCOPE).sum())
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Small data: AdaBox beats DBSCAN on {wins}/{len(small)} datasets "
          f"(mean SCOPE {small.AdaBox_SCOPE.mean():.3f} vs "
          f"{small.DBSCAN_SCOPE.mean():.3f})")
    print(f"  Large data: mean transfer gap {large.transfer_gap.mean():+.3f} "
          f"SCOPE, tuning on ~{large.sample_pct.mean():.1f}% of the data")
    print("\n  SLCD is a scaling tool, not a replacement for direct tuning:")
    print("  below ~5,000 points tune directly; above it, SLCD earns its keep.")

    # ---- figures ----
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(small))
    ax.bar(x - w / 2, small.AdaBox_SCOPE, w, label="AdaBox (direct tune)")
    ax.bar(x + w / 2, small.DBSCAN_SCOPE, w, label="DBSCAN (grid search)")
    ax.set_xticks(x)
    ax.set_xticklabels(small.dataset)
    ax.set_ylabel("SCOPE", fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_title("Small data (<= 5,000 points): direct tuning, no SLCD",
                 fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "small_data_adabox_vs_dbscan.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(large))
    ax.bar(x - w / 2, large.sample_SCOPE, w, label="SCOPE on the sample")
    ax.bar(x + w / 2, large.full_SCOPE, w, label="SCOPE on the full dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(large.dataset)
    ax.set_ylabel("SCOPE", fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_title("Large data (30k-50k points): SLCD parameter transfer\n"
                 "tuned on ~500 points, deployed to all", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "large_data_slcd_transfer.png"), dpi=150)
    plt.close()

    ax = pd.DataFrame(components).set_index("dataset").plot(
        kind="bar", figsize=(10, 5))
    ax.set_ylabel("Component score")
    ax.set_title("SCOPE five-component breakdown (AdaBox, direct tuning)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "scope_components.png"), dpi=150)
    plt.close()

    small.to_csv(os.path.join(OUT, "small_data_direct_tuning.csv"), index=False)
    large.to_csv(os.path.join(OUT, "large_data_slcd_transfer.csv"), index=False)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write("SC-ML low-D reproduce summary\n=============================\n")
        f.write(f"Small data (direct tuning, no SLCD): AdaBox mean SCOPE "
                f"{small.AdaBox_SCOPE.mean():.3f}, DBSCAN "
                f"{small.DBSCAN_SCOPE.mean():.3f}, "
                f"AdaBox wins {wins}/{len(small)}\n")
        f.write(f"Large data (SLCD transfer): mean gap "
                f"{large.transfer_gap.mean():+.3f} SCOPE on ~"
                f"{large.sample_pct.mean():.1f}% of the data\n")

    print(f"\nWrote tables and figures to {OUT}/")


if __name__ == "__main__":
    main()
