"""
my_test.py -- test scml on my own data, using the repo's functions.

Covers the three things you can do, and shows how to change what's changeable.
Replace the data-loading block with your own dataset when ready.
"""

import warnings
import numpy as np
from scml.lowd import AdaBox, SLCD, scope_score, scope_report, default_sample_size

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# LOAD YOUR DATA  (pick one; default is a generated set so this runs as-is)
# ---------------------------------------------------------------------------
# CSV with columns x, y, label:
# import pandas as pd
# df = pd.read_csv("my_dataset.csv")
# X = df[["x", "y"]].to_numpy(); y = df["label"].to_numpy()
#
# Saved numpy arrays:
# X = np.load("my_X.npy"); y = np.load("my_labels.npy")
#
# Generated (no file needed):
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=30000, centers=6, cluster_std=1.0, random_state=1)

n_points = len(X)
print(f"dataset: {n_points} points, {len(set(y)) - (1 if -1 in y else 0)} clusters\n")

# ---------------------------------------------------------------------------
# CASE 1 -- SMALL/MEDIUM DATA (<= 5000 points): AdaBox().tune()
#   Auto-uses Grid Search + absolute density. Just call it.
# ---------------------------------------------------------------------------
if n_points <= 5000:
    m = AdaBox().tune(X, y)
    print("=== AdaBox.tune (small-data path) ===")
    print("clusters:", m.n_clusters_, "| SCOPE:", round(scope_score(X, y, m.labels_), 3))

# ---------------------------------------------------------------------------
# CASE 2 -- LARGE DATA (> 5000 points): SLCD with automatic defaults
#   sample_size, n_trials, cascade_stages all chosen from data size + clusters
# ---------------------------------------------------------------------------
else:
    s = SLCD()                      # all defaults = automatic
    labels = s.fit_predict(X, y)
    print("=== SLCD (automatic defaults) ===")
    print("sample_size used :", s.sample_size_)
    print("n_trials used    :", s.n_trials_)
    print("cascade_stages   :", s.cascade_stages_)
    print("calibration SCOPE:", round(s.calibration_score_, 3))
    print("full-data SCOPE  :", round(scope_score(X, y, labels), 3))
    print("stage history:")
    for h in s.stage_history_:
        print(f"   stage {h['stage']}: size={h['sample_size']} "
              f"score={h['score']:.3f} retuned={h['retuned']}")

    rep = scope_report(X, y, labels)
    print("components:", {k: round(rep[k], 3) for k in
          ["Core_Purity", "Boundary_Recall", "Cluster_Precision",
           "Noise_F1", "Cluster_Count_Accuracy"]})

# ---------------------------------------------------------------------------
# CASE 3 -- OVERRIDE the knobs (works for any large dataset)
#   Pass numbers instead of None to take manual control.
# ---------------------------------------------------------------------------
if n_points > 5000:
    print("\n=== SLCD (manual overrides) ===")
    print("auto sample size would be:", default_sample_size(n_points,
          n_clusters=len(set(y)) - (1 if -1 in y else 0)))
    s2 = SLCD(
        sample_size=1000,     # force a specific calibration sample size
        n_trials=300,         # more Random Search trials (default 100/200)
        cascade_stages=2,     # force 2 cascade stages
        random_state=7,       # change the seed
    )
    labels2 = s2.fit_predict(X, y)
    print(f"used sample={s2.sample_size_} trials={s2.n_trials_} "
          f"stages={s2.cascade_stages_}")
    print("full-data SCOPE:", round(scope_score(X, y, labels2), 3))
