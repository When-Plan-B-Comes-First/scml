"""
examples/highd_unsupervised.py -- tuning AdaGraph with NO ground-truth labels.

    python examples/highd_unsupervised.py

This is the setting clustering is usually deployed in: reinforcement-learning
state abstraction, exploratory analysis, production pipelines -- anywhere
labels do not exist. Graph-SCOPE scores candidates from kNN-graph topology
alone, so no `y` is needed at any point.
"""

import warnings

from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from scml.highd import AdaGraph

warnings.filterwarnings("ignore")

# y is generated only so we can CHECK the result afterwards.
# It is never used for tuning.
X, y = make_blobs(n_samples=1200, n_features=25, centers=6,
                  cluster_std=2.5, random_state=0)

model = AdaGraph().tune_unsupervised(X, n_trials=400)

print(f"Dimensions:       {X.shape[1]}")
print(f"Clusters found:   {model.n_clusters_}  (true: 6 - never shown to the tuner)")
print(f"Graph-SCOPE:      {model.graph_scope_:.3f}  <- the objective optimised")
print(f"Noise discarded:  {100 * model.noise_frac_:.1f}%")
print()
print(f"ARI vs held-out truth: {adjusted_rand_score(y, model.labels_):.3f}")
print("(ARI is the JUDGE. Graph-SCOPE was only the selection SIGNAL --")
print(" reporting Graph-SCOPE as proof of quality would be circular.)")
