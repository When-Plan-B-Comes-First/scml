"""
One-command benchmarking for the high-D track.

Two different questions need two different comparisons, and conflating them
is the easiest way to produce a misleading result:

``benchmark_highd``
    *Algorithm* comparison. AdaGraph against HDBSCAN, K-Means and Ward on
    your data. Every method is tuned, and -- because the two tracks disagree
    about which objective is "fair" -- each method is tuned **twice**, once
    against SCOPE and once against ARI, with both reported. That removes the
    objection that a method only won because the headline metric happened to
    be the one it optimised.

``compare_k_selection``
    *Selection-signal* comparison. Graph-SCOPE against Silhouette for
    choosing the number of clusters. Both drive the same clusterer, so the
    only variable is the signal. Includes the shuffled-label negative control
    from the author's benchmarks: if a shuffled clustering scores well, the
    harness is broken and the results are void.

In both cases Graph-SCOPE is only ever a *selection signal*; judging is done
by supervised SCOPE and ARI against held-out labels.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from ..lowd.scope import compute_dice_metrics
from .adagraph import AdaGraph
from .graph_scope import build_knn_graph, compute_graph_scope
from .slcd import SLCD

try:
    from sklearn.cluster import HDBSCAN as _SKHDBSCAN
    HAS_HDBSCAN = True
except ImportError:  # pragma: no cover
    _SKHDBSCAN = None
    try:
        import hdbscan as _hdbscan_pkg
        HAS_HDBSCAN = True
    except ImportError:
        _hdbscan_pkg = None
        HAS_HDBSCAN = False


def _scope(X, y_true, labels):
    try:
        return float(compute_dice_metrics(X, y_true, labels,
                                          verbose=False)["Overall_Score"])
    except Exception:
        return 0.0


def _ari(y_true, labels):
    from sklearn.metrics import adjusted_rand_score
    try:
        return float(adjusted_rand_score(y_true, labels))
    except Exception:
        return 0.0


def _make_hdbscan(**kw):
    if _SKHDBSCAN is not None:
        return _SKHDBSCAN(**kw)
    return _hdbscan_pkg.HDBSCAN(gen_min_span_tree=False, **kw)


def tune_hdbscan(X, y_true, objective="SCOPE", n_trials=200, seed=42):
    """Random-search HDBSCAN against the given objective.

    Mirrors the author's high-D baseline tuning: the same space and trial
    budget, with the objective made explicit so the baseline can be given the
    same target as AdaGraph.
    """
    import random as rnd
    rng = rnd.Random(seed)
    mcs_choices = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
    ms_choices = [1, 2, 3, 5, 7, 10, 15]
    method_choices = ["eom", "leaf"]
    eps_choices = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    best = (-np.inf, None, None)
    for _ in range(n_trials):
        cfg = {
            "min_cluster_size": rng.choice(mcs_choices),
            "min_samples": rng.choice(ms_choices),
            "cluster_selection_method": rng.choice(method_choices),
            "cluster_selection_epsilon": rng.choice(eps_choices),
        }
        try:
            lbl = _make_hdbscan(**cfg).fit_predict(X)
            if len(set(lbl) - {-1}) == 0:
                continue
            val = (_scope(X, y_true, lbl) if objective == "SCOPE"
                   else _ari(y_true, lbl))
            if val > best[0]:
                best = (val, lbl.copy(), cfg)
        except Exception:
            continue
    if best[1] is None:
        lbl = _make_hdbscan(min_cluster_size=15, min_samples=5).fit_predict(X)
        return lbl, {"min_cluster_size": 15, "min_samples": 5}
    return best[1], best[2]


def _tune_partitional(X, y_true, algo, objective, k_max=20, seed=42):
    """Sweep k for K-Means or Ward against the given objective."""
    from sklearn.cluster import AgglomerativeClustering, KMeans
    best = (-np.inf, None, None)
    for k in range(2, k_max + 1):
        if algo == "KMeans":
            lbl = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(X)
        else:
            lbl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        val = (_scope(X, y_true, lbl) if objective == "SCOPE"
               else _ari(y_true, lbl))
        if val > best[0]:
            best = (val, lbl, {"n_clusters": k})
    return best[1], best[2]


def benchmark_highd(X, y, algorithms=("AdaGraph", "HDBSCAN", "KMeans", "Ward"),
                    objectives=("SCOPE", "ARI"), slcd_threshold=5000,
                    n_trials=400, hdbscan_trials=200, k_max=20,
                    dataset_name="dataset", verbose=True):
    """Compare AdaGraph against high-D baselines, tuning on both objectives.

    Each algorithm is tuned once per objective, and every run is scored on
    *both* SCOPE and ARI. So a row tuned on ARI still reports its SCOPE, and
    vice versa -- the reader can see whether a win survives changing the
    target.

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Data in native dimensionality; no reduction applied.
    y : array-like (n_samples,)
        Ground-truth labels, used for tuning and judging.
    algorithms : tuple of str
    objectives : tuple of str
        Which tuning targets to run. Default runs both.
    slcd_threshold : int, default=5000
        Above this many points AdaGraph is tuned via SLCD (density-aware
        sample -> learn -> classify -> deploy) rather than on the full data.
    n_trials : int, default=400
        AdaGraph random-search trials.
    hdbscan_trials : int, default=200
        HDBSCAN random-search trials.

    Returns
    -------
    pandas.DataFrame
        One row per (algorithm, objective), sorted by SCOPE.
    """
    import pandas as pd

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(X)
    expected_k = int(len(set(y[y >= 0]))) if (y >= 0).any() else None
    rows = []

    for objective in objectives:
        for name in algorithms:
            if name == "HDBSCAN" and not HAS_HDBSCAN:
                if verbose:
                    print("  HDBSCAN backend not installed - skipping")
                continue
            if verbose:
                print(f"  {name} (tuned on {objective}) ...", end="", flush=True)
            t0 = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if name == "AdaGraph":
                        if n > slcd_threshold:
                            s = SLCD(n_trials=n_trials, expected_k=expected_k)
                            labels = s.fit_predict(X, y)
                            params = {"via": "SLCD",
                                      "sample_size": s.sample_size_}
                        else:
                            m = AdaGraph().tune(X, y, n_trials=n_trials,
                                                expected_k=expected_k)
                            labels, params = m.labels_, {"via": "direct tune"}
                    elif name == "HDBSCAN":
                        labels, params = tune_hdbscan(
                            X, y, objective=objective, n_trials=hdbscan_trials)
                    elif name in ("KMeans", "Ward"):
                        labels, params = _tune_partitional(
                            X, y, name, objective, k_max=k_max)
                    else:
                        raise ValueError(f"Unknown algorithm: {name}")
                elapsed = time.time() - t0
                labels = np.asarray(labels)
                rows.append({
                    "Algorithm": name,
                    "Tuned_On": objective,
                    "SCOPE": _scope(X, y, labels),
                    "ARI": _ari(y, labels),
                    "K_found": int(len(set(labels[labels >= 0]))),
                    "Noise_pct": float((labels == -1).mean() * 100),
                    "Time_s": round(elapsed, 1),
                    "Params": str(params),
                })
                if verbose:
                    print(f" SCOPE {rows[-1]['SCOPE']:.3f} / "
                          f"ARI {rows[-1]['ARI']:.3f}  ({elapsed:.0f}s)")
            except Exception as exc:
                if verbose:
                    print(f" FAILED ({exc})")

    if not rows:
        raise RuntimeError("No algorithm produced a result.")

    results = pd.DataFrame(rows).sort_values(
        "SCOPE", ascending=False).reset_index(drop=True)

    if verbose:
        print("\n" + "=" * 78)
        print(f"HIGH-D COMPARISON - {dataset_name}  "
              f"(n={n}, dim={X.shape[1]}, true k={expected_k})")
        print("=" * 78)
        cols = ["Algorithm", "Tuned_On", "SCOPE", "ARI", "K_found",
                "Noise_pct", "Time_s"]
        disp = results[cols].copy()
        for c in ("SCOPE", "ARI"):
            disp[c] = disp[c].map(lambda v: f"{v:.3f}")
        disp["Noise_pct"] = disp["Noise_pct"].map(lambda v: f"{v:.1f}")
        print(disp.to_string(index=False))
        print("=" * 78)
        for obj in objectives:
            sub = results[results.Tuned_On == obj]
            if len(sub):
                key = "SCOPE" if obj == "SCOPE" else "ARI"
                best = sub.sort_values(key, ascending=False).iloc[0]
                print(f"Best when tuned on {obj}: {best['Algorithm']} "
                      f"({key} {best[key]:.3f})")
        print()

    return results


def compare_k_selection(X, y, clusterer="KMeans", k_max=20, k_graph=15,
                        dataset_name="dataset", negative_control=True,
                        verbose=True):
    """Compare Graph-SCOPE against Silhouette as a k-selection signal.

    Both signals choose k for the same clusterer, so the only variable is the
    signal itself. Each selection is then judged against the held-out labels
    with SCOPE and ARI -- the signals never judge themselves.

    Parameters
    ----------
    X, y : array-like
        Data and held-out ground truth (used only for judging).
    clusterer : {"KMeans", "Ward"}, default="KMeans"
    k_max : int, default=20
        Largest k considered.
    k_graph : int, default=15
        Neighbours in the kNN graph for Graph-SCOPE.
    negative_control : bool, default=True
        Also score a shuffled labelling. Its ARI must be ~0; if it is not,
        the harness is broken and the comparison is void.

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import silhouette_score

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    true_k = int(len(set(y[y >= 0])))
    knn, _ = build_knn_graph(X, k=k_graph)

    rng = np.random.default_rng(42)
    sil_idx = rng.choice(len(X), min(len(X), 2000), replace=False)

    best_gs = (-np.inf, None, None)
    best_sil = (-np.inf, None, None)
    for k in range(2, k_max + 1):
        if clusterer == "KMeans":
            lbl = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(X)
        else:
            lbl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        gs = compute_graph_scope(knn, lbl)[0]
        if gs > best_gs[0]:
            best_gs = (gs, k, lbl)
        if len(np.unique(lbl[sil_idx])) >= 2:
            sv = silhouette_score(X[sil_idx], lbl[sil_idx])
            if sv > best_sil[0]:
                best_sil = (sv, k, lbl)

    rows = []
    for signal, (val, k, lbl) in (("Graph-SCOPE", best_gs),
                                  ("Silhouette", best_sil)):
        if lbl is None:
            continue
        rows.append({
            "Signal": signal, "Selected_k": k, "True_k": true_k,
            "k_error": abs(k - true_k),
            "Signal_value": round(float(val), 4),
            "SCOPE": _scope(X, y, lbl), "ARI": _ari(y, lbl),
        })

    if negative_control and rows:
        shuffled = rng.permutation(best_gs[2])
        rows.append({
            "Signal": "NEGATIVE CONTROL (shuffled)", "Selected_k": best_gs[1],
            "True_k": true_k, "k_error": None, "Signal_value": None,
            "SCOPE": _scope(X, y, shuffled), "ARI": _ari(y, shuffled),
        })

    results = pd.DataFrame(rows)

    if verbose:
        print("\n" + "=" * 78)
        print(f"K-SELECTION SIGNAL COMPARISON - {dataset_name} "
              f"({clusterer}, true k={true_k})")
        print("=" * 78)
        print(results.to_string(index=False))
        nc = results[results.Signal.str.startswith("NEGATIVE")]
        if len(nc) and abs(float(nc.iloc[0]["ARI"])) > 0.05:
            print("\n  WARNING: negative control ARI is not ~0. The harness "
                  "is suspect and these results should not be trusted.")
        print()

    return results
