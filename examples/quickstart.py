"""
examples/quickstart.py — the 60-second example, runnable as-is:

    python examples/quickstart.py

Uses AdaBox's three-phase tuning (the recommended path when you have labels).
"""

from sklearn.datasets import make_blobs

from scml.lowd import AdaBox, scope_score

X, y = make_blobs(n_samples=800, centers=4, cluster_std=1.0, random_state=0)

model = AdaBox().tune(X, y)        # three-phase calibration
labels = model.labels_

print(f"Clusters found: {model.n_clusters_}")
print(f"SCOPE score:    {scope_score(X, y, labels):.3f}")
print(f"Stage 3 merge:  {model.best_params_.get('stage3_applied')}")
