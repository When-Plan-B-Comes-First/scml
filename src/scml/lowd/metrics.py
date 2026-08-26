"""Public SCOPE metric API for the low-D track.

``scope_score`` returns the single overall SCOPE value; ``scope_report``
returns the full five-component breakdown.
"""

from __future__ import annotations

import numpy as np

from .scope import compute_dice_metrics


def scope_score(X, labels_true, labels_pred, alpha=0.25):
    """Return the overall SCOPE score in [0, 1] (higher is better).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data points.
    labels_true : array-like of shape (n_samples,)
        Ground-truth labels (``-1`` for noise).
    labels_pred : array-like of shape (n_samples,)
        Predicted labels (``-1`` for noise).
    alpha : float, default=0.25
        Core percentile threshold (densest ``alpha`` fraction are cores).
    """
    result = compute_dice_metrics(np.asarray(X), np.asarray(labels_true),
                                  np.asarray(labels_pred), alpha=alpha)
    return float(result["Overall_Score"])


def scope_report(X, labels_true, labels_pred, alpha=0.25):
    """Return the full SCOPE component breakdown as a dict.

    Keys include ``Core_Purity``, ``Boundary_Recall``, ``Cluster_Precision``,
    ``Noise_F1``, ``Cluster_Count_Accuracy``, ``Overall_Score``,
    ``N_True_Clusters`` and ``N_Pred_Clusters``.
    """
    return compute_dice_metrics(np.asarray(X), np.asarray(labels_true),
                                np.asarray(labels_pred), alpha=alpha)
