#!/usr/bin/env python3
"""
reproduce/highd/run_all.py -- regenerate the high-D track's headline results
with a single command:

    python reproduce/highd/run_all.py

Everything here runs OFFLINE on synthetic data, so anyone can reproduce it
without network access, API keys, or large downloads.

Two experiments:

  1. Irrelevant-dimension sweep (the pre-registered controlled test).
     Signal lives in 5 dimensions; pure-noise dimensions are appended in
     increasing numbers. As noise accumulates, Euclidean distance is
     progressively dominated by noise, so Silhouette should degrade. If
     Graph-SCOPE degrades more slowly, that is direct evidence for
     high-dimensional structural fidelity.

  2. All-informative control. Dimension grows but every dimension carries
     signal. If the mechanism really is noise-domination of distance, the
     advantage should NOT appear here. This is what separates "graph topology
     resists noise" from "graph topology is just better", and it is the reason
     the sweep alone would not be conclusive.

Outputs land in reproduce/highd/outputs/.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from scml.highd import build_knn_graph, compute_graph_scope

warnings.filterwarnings("ignore")

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

RS = 42
N_POINTS = 3000
D_SIGNAL = 5
TRUE_K = 8
K_RANGE = range(2, 21)
KNN_K = 15
SIL_SUB = 2000
CLUSTER_STD = 2.0          # calibrated so neither method gets a free pass
SEEDS = [0, 1, 2, 3, 4]
NOISE_DIMS = [0, 15, 45, 140, 495]
DIMS_B = [5, 20, 50, 145]


def make_data(n, d_signal, d_noise, true_k, cluster_std, seed, noise_scale=1.0):
    """Blobs in d_signal informative dimensions, plus d_noise pure-noise ones."""
    X, y = make_blobs(n_samples=n, n_features=d_signal, centers=true_k,
                      cluster_std=cluster_std, center_box=(-10.0, 10.0),
                      random_state=seed)
    if d_noise > 0:
        rng = np.random.default_rng(seed + 9999)
        X = np.hstack([X, rng.normal(0, noise_scale, size=(n, d_noise))])
    return StandardScaler().fit_transform(X), y


def select_k(X, signal, seed=RS):
    """Choose k by 'gs' (Graph-SCOPE) or 'sil' (Silhouette); return (k, labels)."""
    knn = build_knn_graph(X, k=KNN_K)[0] if signal == "gs" else None
    idx = np.random.default_rng(seed).choice(
        len(X), min(len(X), SIL_SUB), replace=False)
    best = (None, None, -np.inf)
    for k in K_RANGE:
        lab = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(X)
        if signal == "gs":
            val = compute_graph_scope(knn, lab)[0]
        else:
            if len(np.unique(lab[idx])) < 2:
                continue
            val = silhouette_score(X[idx], lab[idx])
        if val > best[2]:
            best = (k, lab, val)
    return best[0], best[1]


def run_axis(varying, informative):
    """Run one axis. `informative=False` appends noise dims; True grows signal."""
    rows = []
    for v in varying:
        for seed in SEEDS:
            if informative:
                X, y = make_data(N_POINTS, v, 0, TRUE_K, CLUSTER_STD, seed)
                total, noise = v, 0
            else:
                X, y = make_data(N_POINTS, D_SIGNAL, v, TRUE_K, CLUSTER_STD, seed)
                total, noise = D_SIGNAL + v, v
            kg, lg = select_k(X, "gs", seed=seed)
            ks, ls = select_k(X, "sil", seed=seed)
            rows.append({
                "total_dim": total, "noise_dim": noise, "seed": seed,
                "GS_k": kg, "GS_ARI": adjusted_rand_score(y, lg),
                "Sil_k": ks, "Sil_ARI": adjusted_rand_score(y, ls),
            })
        print(f"    done {'dim' if informative else 'noise_dim'}={v}")
    df = pd.DataFrame(rows)
    df["delta_ARI"] = df.GS_ARI - df.Sil_ARI
    return df


def main():
    print("Experiment 1 - irrelevant-dimension sweep (primary)")
    axisA = run_axis(NOISE_DIMS, informative=False)
    sumA = axisA.groupby("noise_dim").agg(
        total_dim=("total_dim", "first"),
        GS_k=("GS_k", "mean"), GS_ARI=("GS_ARI", "mean"),
        Sil_k=("Sil_k", "mean"), Sil_ARI=("Sil_ARI", "mean"),
        delta=("delta_ARI", "mean"), delta_sd=("delta_ARI", "std"),
    ).round(4)

    print("\nExperiment 2 - all-informative control")
    axisB = run_axis(DIMS_B, informative=True)
    sumB = axisB.groupby("total_dim").agg(
        GS_ARI=("GS_ARI", "mean"), Sil_ARI=("Sil_ARI", "mean"),
        delta=("delta_ARI", "mean"), delta_sd=("delta_ARI", "std"),
    ).round(4)

    print("\n" + "=" * 78)
    print(f"AXIS A - {D_SIGNAL} informative dims + noise "
          f"(true k={TRUE_K}, {len(SEEDS)} seeds)")
    print("=" * 78)
    print(sumA.to_string())
    print("\n" + "=" * 78)
    print("AXIS B - all dimensions informative (control)")
    print("=" * 78)
    print(sumB.to_string())

    # ---- verdict, stated against the prediction rather than post hoc ----
    d_lo, d_hi = sumA.loc[NOISE_DIMS[0], "delta"], sumA.loc[NOISE_DIMS[-1], "delta"]
    gs_hi, sil_hi = sumA.loc[NOISE_DIMS[-1], "GS_ARI"], sumA.loc[NOISE_DIMS[-1], "Sil_ARI"]
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  delta ARI at {NOISE_DIMS[0]} noise dims:   {d_lo:+.4f}")
    print(f"  delta ARI at {NOISE_DIMS[-1]} noise dims: {d_hi:+.4f}")
    d_peak = float(sumA["delta"].max())
    d_peak_at = int(sumA["delta"].idxmax())
    if max(gs_hi, sil_hi) < 0.20:
        verdict = ("INCONCLUSIVE AT THE ENDPOINT - both methods collapsed at "
                   f"{NOISE_DIMS[-1]} noise dims (ARI {gs_hi:.3f} vs "
                   f"{sil_hi:.3f}), so that regime is unsolvable for either "
                   "and is not a win. NOTE the advantage is non-monotonic: it "
                   f"peaks at +{d_peak:.4f} ARI with {d_peak_at} noise dims. "
                   "Report the peak and the collapse together, not one alone.")
    elif d_hi > 0.10 and d_hi > d_lo:
        verdict = ("SUPPORTED - Graph-SCOPE's advantage grows with irrelevant "
                   "dimensions.")
    elif d_hi < -0.05:
        verdict = "LOSS - Silhouette proved more robust. Reported as such."
    else:
        verdict = "NO EFFECT - no meaningful difference across the sweep."
    print(f"  peak delta:                {d_peak:+.4f} at {d_peak_at} noise dims")
    print(f"  {verdict}")

    db_hi = sumB.loc[DIMS_B[-1], "delta"]
    print(f"\n  Control (all informative, {DIMS_B[-1]} dims): delta = {db_hi:+.4f}")
    print("  " + ("As predicted: little advantage when every dimension carries "
                  "signal, which supports the noise-domination mechanism."
                  if abs(db_hi) < 0.05 else
                  "Advantage also appears with all-informative dimensions, so "
                  "the mechanism is not purely noise-domination."))

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(sumA.index, sumA.GS_ARI, yerr=axisA.groupby("noise_dim").GS_ARI.std(),
                marker="o", label="Graph-SCOPE", capsize=3)
    ax.errorbar(sumA.index, sumA.Sil_ARI, yerr=axisA.groupby("noise_dim").Sil_ARI.std(),
                marker="s", label="Silhouette", capsize=3)
    ax.set_xlabel("Number of pure-noise dimensions", fontweight="bold")
    ax.set_ylabel("ARI of the selected clustering", fontweight="bold")
    ax.set_title("k-selection quality as irrelevant dimensions accumulate",
                 fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(OUT, "noise_dimension_sweep.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    axisA.to_csv(os.path.join(OUT, "axisA_noise_sweep.csv"), index=False)
    axisB.to_csv(os.path.join(OUT, "axisB_informative_dims.csv"), index=False)
    sumA.to_csv(os.path.join(OUT, "axisA_summary.csv"))
    sumB.to_csv(os.path.join(OUT, "axisB_summary.csv"))
    with open(os.path.join(OUT, "verdict.txt"), "w") as f:
        f.write(f"Axis A delta at {NOISE_DIMS[0]} noise dims: {d_lo:+.4f}\n")
        f.write(f"Axis A delta at {NOISE_DIMS[-1]} noise dims: {d_hi:+.4f}\n")
        f.write(f"Peak delta: {d_peak:+.4f} at {d_peak_at} noise dims\n")
        f.write(f"Verdict: {verdict}\n")
        f.write(f"Axis B delta at {DIMS_B[-1]} dims: {db_hi:+.4f}\n")

    print(f"\nWrote results and figure to {OUT}/")


if __name__ == "__main__":
    main()
