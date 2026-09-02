"""
examples/highd_slcd_scaling.py -- high-D SLCD at scale.

    python examples/highd_slcd_scaling.py

Sample -> Learn -> Classify -> Deploy: AdaGraph is tuned on a small
density-aware sample, then the remaining points are assigned to the sample's
clusters by a two-pass kNN vote. The full dataset is never tuned.
"""

from sklearn.datasets import make_blobs

from scml.highd import SLCD
from scml.lowd import scope_score

X, y = make_blobs(n_samples=20000, n_features=40, centers=8,
                  cluster_std=2.5, random_state=1)

slcd = SLCD(n_trials=200, expected_k=8)
labels = slcd.fit_predict(X, y)

print(f"Full dataset:    {X.shape[0]} points, {X.shape[1]} dimensions")
print(f"Learned on:      {slcd.sample_size_} points (density-aware sample)")
print(f"Sample score:    {slcd.learn_score_:.3f}")
print(f"Full-data SCOPE: {scope_score(X, y, labels):.3f}")
print(f"Clusters found:  {len(set(labels[labels >= 0]))}  (true: 8)")
