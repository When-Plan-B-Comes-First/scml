"""
examples/slcd_scaling.py — Sample -> Label -> Calibrate -> Deploy on a large dataset.

The three-phase calibration runs on a 500-point stratified sample; the chosen
parameters are deployed to the full dataset. Run as:

    python examples/slcd_scaling.py
"""

from sklearn.datasets import make_blobs

from scml.lowd import SLCD, scope_score

X, y = make_blobs(n_samples=20000, centers=6, cluster_std=1.0, random_state=7)

slcd = SLCD(sample_size=500)
labels = slcd.fit_predict(X, y)

print(f"Calibrated on a {slcd.sample_size}-point sample")
print(f"Best parameters:    {slcd.best_params_}")
print(f"Sample SCOPE:       {slcd.calibration_score_:.3f}")
print(f"Full-dataset SCOPE: {scope_score(X, y, labels):.3f}")
