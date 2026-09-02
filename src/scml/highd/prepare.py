"""
Turn whatever a user hands you into a dataset AdaGraph can cluster.

This is the high-D counterpart of :mod:`scml.lowd.prepare`, with one decisive
difference: **it does not reduce dimensionality.** AdaGraph clusters in the
original feature space by design, so projecting to 2-D first would defeat the
purpose. Everything else -- label detection, dropping junk columns, removing
missing values, scaling, label encoding -- works the same way.

What it handles:
  * choosing feature and label columns (explicitly or by inference)
  * dropping non-numeric feature columns and identifier-like columns
  * removing rows with missing or infinite values
  * standardising features (recommended: kNN distances assume comparable scales)
  * re-encoding labels to 0..k-1 while preserving -1 as noise
  * optionally capping very large datasets by subsampling
"""

from __future__ import annotations

import numpy as np

from ..lowd.prepare import PreparationReport, _encode_labels, _looks_like_labels


def _load_source_highd(data, label_column, feature_columns, report):
    """Return (X_raw, y_raw) from a CSV path, DataFrame, or array."""
    import pandas as pd

    if isinstance(data, str):
        sep = "\t" if data.lower().endswith((".tsv", ".tab")) else ","
        df = pd.read_csv(data, sep=sep)
        report.add(f"loaded {data}")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(
            f"Unsupported data type {type(data).__name__}. Pass a CSV path, a "
            "pandas DataFrame, or numpy arrays via prepare_highd(X, y=...)."
        )

    # ---- label column ----
    if label_column is None:
        preferred = ["label", "labels", "class", "classes", "target",
                     "cluster", "clusters", "ground_truth", "true_label",
                     "category", "group", "y"]
        lower_map = {str(c).strip().lower(): c for c in df.columns}
        label_column = None
        for name in preferred:
            if name not in lower_map:
                continue
            col = lower_map[name]
            if _looks_like_labels(df[col]):
                label_column = col
                break
            report.add(f"skipped '{col}' as label candidate: looks continuous "
                       "or identifier-like, not cluster labels")
        if label_column is None:
            raise ValueError(
                "Could not identify a ground-truth label column.\n"
                f"Columns available: {list(df.columns)}\n"
                "Pass it explicitly, e.g. prepare_highd(path, "
                "label_column='my_labels')."
            )
        report.add(f"inferred label column: '{label_column}'")
    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found. Available: "
            f"{list(df.columns)}")
    report.label_name = str(label_column)

    # ---- feature columns ----
    if feature_columns is None:
        candidates = [c for c in df.columns if c != label_column]
        numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
        dropped = [str(c) for c in candidates if c not in numeric]
        if dropped:
            shown = dropped if len(dropped) <= 8 else dropped[:8] + ["..."]
            report.add(f"dropped {len(dropped)} non-numeric column(s): {shown}")
        feature_columns = numeric
    if not feature_columns:
        raise ValueError("No numeric feature columns found.")
    report.feature_names = ([str(c) for c in feature_columns]
                            if len(feature_columns) <= 12
                            else [f"{len(feature_columns)} numeric columns"])

    return (df[feature_columns].to_numpy(dtype=float),
            df[label_column].to_numpy())


def prepare_highd(data, y=None, label_column=None, feature_columns=None,
                  noise_values=(-1, "-1", "noise", "Noise", "NOISE", "outlier"),
                  standardize=True, max_samples=None, drop_constant=True,
                  random_state=42, verbose=True):
    """Clean and validate a dataset for AdaGraph, keeping all dimensions.

    Parameters
    ----------
    data : str | pandas.DataFrame | numpy.ndarray
        CSV/TSV path, DataFrame, or feature array (pass labels via ``y``).
    y : array-like, optional
        Ground-truth labels when ``data`` is an array.
    label_column : str, optional
        Ground-truth column name. Inferred from common names if omitted.
    feature_columns : list, optional
        Columns to use as features. Defaults to every numeric non-label column.
    noise_values : tuple
        Label values meaning "noise"; these become ``-1``.
    standardize : bool, default=True
        Scale features to zero mean / unit variance. Strongly recommended:
        the kNN graph uses Euclidean distance, so features on wildly different
        scales would let one dominate the neighbourhood structure.
    max_samples : int, optional
        Randomly subsample to at most this many rows. Useful for very large
        datasets; the kNN graph is the expensive part.
    drop_constant : bool, default=True
        Drop zero-variance columns, which carry no information and can break
        standardisation.
    verbose : bool, default=True
        Print the report.

    Returns
    -------
    X : ndarray (n_samples, n_features)
        **Native dimensionality preserved** -- no PCA, no projection.
    y : ndarray (n_samples,)
        Integer labels, ``-1`` for noise.
    report : PreparationReport
    """
    report = PreparationReport()

    if isinstance(data, np.ndarray) and y is not None:
        X_raw = np.asarray(data, dtype=float)
        y_raw = np.asarray(y)
        report.feature_names = [f"{X_raw.shape[1]} numeric columns"]
    else:
        X_raw, y_raw = _load_source_highd(data, label_column, feature_columns,
                                          report)

    if X_raw.ndim != 2:
        raise ValueError(f"Features must be 2-D, got shape {X_raw.shape}.")
    report.n_rows_in = len(X_raw)
    report.n_features_in = X_raw.shape[1]

    if len(X_raw) != len(y_raw):
        raise ValueError(f"Features and labels differ in length: "
                         f"{len(X_raw)} vs {len(y_raw)}.")

    y_enc = _encode_labels(y_raw, noise_values, report)

    # ---- drop rows with missing / infinite values ----
    finite = np.isfinite(X_raw).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        X_raw, y_enc = X_raw[finite], y_enc[finite]
        report.add(f"removed {n_bad} row(s) with missing or infinite values")

    # ---- drop constant columns ----
    if drop_constant and X_raw.shape[1] > 1:
        keep = X_raw.std(axis=0) > 1e-12
        n_const = int((~keep).sum())
        if n_const and keep.sum() >= 2:
            X_raw = X_raw[:, keep]
            report.add(f"dropped {n_const} zero-variance column(s)")

    if X_raw.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 usable feature columns, found {X_raw.shape[1]}.")
    if len(X_raw) < 20:
        raise ValueError(
            f"Only {len(X_raw)} usable rows remain — too few to build a kNN "
            "graph.")

    # ---- optional subsample ----
    if max_samples is not None and len(X_raw) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_raw), size=max_samples, replace=False)
        X_raw, y_enc = X_raw[idx], y_enc[idx]
        report.add(f"subsampled to {max_samples} rows (max_samples)")

    if standardize:
        from sklearn.preprocessing import StandardScaler
        X_raw = StandardScaler().fit_transform(X_raw)
        report.add("standardised features (zero mean, unit variance)")

    report.n_rows_out = len(X_raw)
    report.n_clusters = int(len(set(y_enc[y_enc >= 0])))
    report.n_noise = int((y_enc == -1).sum())
    report.add(f"kept all {X_raw.shape[1]} dimensions "
               "(AdaGraph clusters natively; no PCA applied)")

    if report.n_clusters < 2:
        raise ValueError(
            f"Found only {report.n_clusters} cluster(s) in the labels. "
            "At least 2 are needed for a meaningful comparison.")
    if report.n_clusters > max(50, 0.5 * report.n_rows_out):
        raise ValueError(
            f"The label column '{report.label_name}' has {report.n_clusters} "
            f"distinct values across {report.n_rows_out} rows, which does not "
            "look like cluster labels. Pass the correct column with "
            "label_column='...'.")

    if verbose:
        print(report)

    return np.ascontiguousarray(X_raw, dtype=float), y_enc, report
