"""
examples/highd_quickstart.py -- AdaGraph on high-dimensional data.

    python examples/highd_quickstart.py

AdaGraph clusters in the ORIGINAL dimensionality: no PCA, no UMAP, no
projection to 2-D first.
"""

from sklearn.datasets import make_blobs

from scml.highd import AdaGraph, graph_scope_score
from scml.lowd import scope_score

X, y = make_blobs(n_samples=1500, n_features=30, centers=6,
                  cluster_std=2.5, random_state=0)

model = AdaGraph().tune(X, y, n_trials=100)

print(f"Dimensions:      {X.shape[1]}")
print(f"Clusters found:  {model.n_clusters_}  (true: {len(set(y))})")
print(f"SCOPE:           {scope_score(X, y, model.labels_):.3f}")
print(f"Graph-SCOPE:     {graph_scope_score(X, model.labels_):.3f}  "
      f"(unsupervised - no labels used)")
