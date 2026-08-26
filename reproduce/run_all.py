#!/usr/bin/env python3
"""
reproduce/run_all.py — regenerate the low-D paper's headline figures and tables
with a single command:

    python reproduce/run_all.py

Outputs are written to reproduce/outputs/:
  * scope_vs_ari_table.csv   — per-dataset SCOPE and ARI for AdaBox
  * scope_components.png     — SCOPE five-component breakdown per dataset
  * summary.txt              — headline aggregate numbers

This script depends only on scml plus matplotlib/pandas (install with
`pip install scml[reproduce]`). It is intentionally self-contained and uses
public scikit-learn datasets so anyone can run it offline.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from scml.lowd import AdaBox, SLCD, scope_report, scope_score

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

RANDOM_STATE = 42


def build_datasets():
    """Return a dict of name -> (X, y) public benchmark datasets."""
    datasets = {}

    X, y = make_blobs(n_samples=900, centers=3, cluster_std=1.0,
                      random_state=RANDOM_STATE)
    datasets["blobs_3"] = (X, y)

    X, y = make_blobs(n_samples=1200, centers=5, cluster_std=1.2,
                      random_state=RANDOM_STATE)
    datasets["blobs_5"] = (X, y)

    X, y = make_blobs(n_samples=2000, centers=8, cluster_std=1.0,
                      random_state=RANDOM_STATE)
    datasets["blobs_8"] = (X, y)

    X, y = make_moons(n_samples=800, noise=0.06, random_state=RANDOM_STATE)
    datasets["moons"] = (X, y)

    return datasets


def cluster_direct(X, y):
    """Cluster via direct three-phase tuning on the full dataset."""
    model = AdaBox().tune(X, y)
    return model.labels_, model.best_params_


def cluster_with_slcd(X, y):
    """Cluster via the SLCD Sample->Calibrate->Deploy workflow."""
    slcd = SLCD(sample_size=min(500, len(X) // 2), random_state=RANDOM_STATE)
    labels = slcd.fit_predict(X, y)
    return labels, slcd.best_params_


def main():
    datasets = build_datasets()
    rows = []
    component_rows = []

    for name, (X, y) in datasets.items():
        # Direct three-phase tuning (upper bound: tunes on all the data)
        d_labels, _ = cluster_direct(X, y)
        d_scope = scope_score(X, y, d_labels)

        # SLCD transfer (calibrate on a sample, deploy to all)
        labels, params = cluster_with_slcd(X, y)
        rep = scope_report(X, y, labels)
        ari = adjusted_rand_score(y, labels)
        rows.append({
            "dataset": name,
            "n_points": len(X),
            "n_true_clusters": rep["N_True_Clusters"],
            "n_pred_clusters": rep["N_Pred_Clusters"],
            "SCOPE_direct_tune": round(d_scope, 3),
            "SCOPE_slcd_transfer": round(rep["Overall_Score"], 3),
            "ARI_slcd_transfer": round(ari, 3),
        })
        component_rows.append({
            "dataset": name,
            "Core_Purity": rep["Core_Purity"],
            "Boundary_Recall": rep["Boundary_Recall"],
            "Cluster_Precision": rep["Cluster_Precision"],
            "Noise_F1": rep["Noise_F1"],
            "Cluster_Count_Accuracy": rep["Cluster_Count_Accuracy"],
        })
        print(f"[{name:9s}] direct_tune={d_scope:.3f}  "
              f"slcd_transfer={rep['Overall_Score']:.3f}  "
              f"ARI={ari:.3f}  pred_k={rep['N_Pred_Clusters']}")

    # ---- Table ----
    df = pd.DataFrame(rows)
    table_path = os.path.join(OUT, "scope_vs_ari_table.csv")
    df.to_csv(table_path, index=False)

    # ---- Figure: SCOPE component breakdown ----
    comp_df = pd.DataFrame(component_rows).set_index("dataset")
    ax = comp_df.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Component score")
    ax.set_title("SCOPE five-component breakdown (AdaBox via SLCD)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(OUT, "scope_components.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # ---- Summary ----
    summary_path = os.path.join(OUT, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("SC-ML low-D reproduce summary\n")
        f.write("=============================\n")
        f.write(f"datasets: {len(df)}\n")
        f.write(f"mean SCOPE (direct tune):   {df['SCOPE_direct_tune'].mean():.3f}\n")
        f.write(f"mean SCOPE (SLCD transfer): {df['SCOPE_slcd_transfer'].mean():.3f}\n")
        f.write(f"mean ARI   (SLCD transfer): {df['ARI_slcd_transfer'].mean():.3f}\n")

    print("\nWrote:")
    print(" ", table_path)
    print(" ", fig_path)
    print(" ", summary_path)


if __name__ == "__main__":
    main()
