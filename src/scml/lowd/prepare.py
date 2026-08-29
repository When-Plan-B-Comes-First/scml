"""
Turn whatever a user hands you into a dataset AdaBox can cluster.

``prepare_dataset`` accepts a CSV/TSV path, a pandas DataFrame, or plain numpy
arrays, and returns clean 2-D features plus integer labels. It reports every
change it makes so nothing happens silently.

What it handles:
  * choosing feature and label columns (explicitly or by inference)
  * dropping non-numeric feature columns, or encoding them if asked
  * removing rows with missing or infinite values
  * reducing >2 features to 2-D with PCA (AdaBox is a 2-D algorithm)
  * standardising features to zero mean / unit variance
  * re-encoding labels to 0..k-1 while preserving -1 as noise
"""

from __future__ import annotations

import numpy as np


class PreparationReport:
    """Record of every transformation applied to a dataset."""

    def __init__(self):
        self.steps = []
        self.n_rows_in = 0
        self.n_rows_out = 0
        self.n_features_in = 0
        self.n_clusters = 0
        self.n_noise = 0
        self.feature_names = None
        self.label_name = None
        self.pca_variance = None

    def add(self, message):
        self.steps.append(message)

    def __str__(self):
        lines = ["Dataset preparation report", "=" * 60]
        lines.append(f"  rows in:        {self.n_rows_in}")
        lines.append(f"  rows kept:      {self.n_rows_out}")
        lines.append(f"  features in:    {self.n_features_in}")
        if self.feature_names:
            lines.append(f"  feature cols:   {self.feature_names}")
        if self.label_name:
            lines.append(f"  label col:      {self.label_name}")
        lines.append(f"  clusters found in labels: {self.n_clusters}")
        lines.append(f"  points labelled noise:    {self.n_noise}")
        if self.pca_variance is not None:
            lines.append(f"  PCA variance retained:    {self.pca_variance:.1%}")
        if self.steps:
            lines.append("  actions taken:")
            for s in self.steps:
                lines.append(f"    - {s}")
        else:
            lines.append("  actions taken: none (data was already clean)")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()


def _looks_like_labels(series):
    """Heuristic: could this column plausibly be cluster labels?

    Rejects columns that are clearly coordinates (continuous floats with many
    distinct values) or row identifiers (nearly all values unique). Both are
    common trap columns -- a 2-D dataset with an ``x``/``y`` column will
    otherwise have its y-coordinate mistaken for ground truth.
    """
    import pandas as pd

    n = len(series)
    n_unique = int(series.nunique(dropna=True))
    if n_unique < 2:
        return False
    # Identifier-like: almost every row distinct.
    if n_unique > max(50, 0.5 * n):
        return False
    # Continuous float coordinate: non-integral values with many levels.
    if pd.api.types.is_float_dtype(series):
        vals = series.dropna().to_numpy()
        if len(vals) and not np.allclose(vals, np.round(vals)):
            return False
    return True


def _load_source(data, label_column, feature_columns, report):
    """Return (X_raw, y_raw) as numpy arrays from any supported input."""
    import pandas as pd

    # ---- path to a delimited file ----
    if isinstance(data, str):
        sep = "\t" if data.lower().endswith((".tsv", ".tab")) else ","
        df = pd.read_csv(data, sep=sep)
        report.add(f"loaded {data}")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, np.ndarray):
        if label_column is None:
            raise ValueError(
                "When passing a numpy array, also pass labels via the `y` "
                "argument of prepare_dataset(X, y=...)."
            )
        df = pd.DataFrame(data)
    else:
        raise TypeError(
            f"Unsupported data type {type(data).__name__}. Pass a CSV path, "
            "a pandas DataFrame, or numpy arrays."
        )

    # ---- pick the label column ----
    if label_column is None:
        # Checked in priority order: unambiguous names first. 'y' comes last
        # because in 2-D datasets it is far more often a coordinate than a
        # label, and picking it silently produces a nonsense benchmark.
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
            report.add(
                f"skipped column '{col}' as label candidate: it looks like "
                "continuous data or a unique identifier, not cluster labels"
            )

        if label_column is None:
            raise ValueError(
                "Could not identify a ground-truth label column.\n"
                f"Columns available: {list(df.columns)}\n"
                "Pass it explicitly, e.g. "
                "prepare_dataset(path, label_column='my_labels')."
            )
        report.add(f"inferred label column: '{label_column}'")
    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found. Available columns: "
            f"{list(df.columns)}"
        )
    report.label_name = str(label_column)

    # ---- pick the feature columns ----
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != label_column]
        numeric = [c for c in feature_columns
                   if pd.api.types.is_numeric_dtype(df[c])]
        dropped = [str(c) for c in feature_columns if c not in numeric]
        if dropped:
            report.add(f"dropped non-numeric columns: {dropped}")
        feature_columns = numeric
    if not feature_columns:
        raise ValueError("No numeric feature columns found.")
    report.feature_names = [str(c) for c in feature_columns]

    X_raw = df[feature_columns].to_numpy(dtype=float)
    y_raw = df[label_column].to_numpy()
    return X_raw, y_raw


def _is_noise_value(v, noise_set, noise_numeric):
    """True if a single label value should be treated as noise.

    Compares both as text (case-insensitive, e.g. "noise", "outlier") and, when
    the value is numeric, as a number -- so -1, -1.0, and "-1" are all
    recognised as the same noise marker regardless of how pandas typed the
    column (a mixed int/float column loads as float64, turning -1 into -1.0).
    """
    key = str(v).strip()
    if key.lower() in noise_set:
        return True
    try:
        return float(v) in noise_numeric
    except (TypeError, ValueError):
        return False


def _encode_labels(y_raw, noise_values, report):
    """Map labels to 0..k-1 integers, preserving noise as -1."""
    y = np.asarray(y_raw)
    noise_set = {str(v).strip().lower() for v in noise_values}
    noise_numeric = set()
    for v in noise_values:
        try:
            noise_numeric.add(float(v))
        except (TypeError, ValueError):
            pass

    out = np.empty(len(y), dtype=int)
    mapping = {}
    next_id = 0
    for i, v in enumerate(y):
        if _is_noise_value(v, noise_set, noise_numeric):
            out[i] = -1
            continue
        key = str(v).strip()
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        out[i] = mapping[key]

    if next_id > 0 and not np.array_equal(
            np.unique(y[out >= 0]).astype(str),
            np.array(sorted(mapping, key=mapping.get))):
        pass  # ordering detail only; encoding is still valid

    if len(mapping) and set(map(str, np.unique(y))) != set(mapping):
        report.add("labels re-encoded to consecutive integers (noise = -1)")
    return out


def prepare_dataset(data, y=None, label_column=None, feature_columns=None,
                    noise_values=(-1, "-1", "noise", "Noise", "NOISE", "outlier"),
                    standardize=True, reduce_to_2d=True, verbose=True):
    """Clean and validate a dataset for use with AdaBox.

    Parameters
    ----------
    data : str | pandas.DataFrame | numpy.ndarray
        A path to a CSV/TSV file, a DataFrame, or a 2-D feature array. If an
        array, pass labels via ``y``.
    y : array-like, optional
        Ground-truth labels, when ``data`` is a plain feature array.
    label_column : str, optional
        Name of the ground-truth column. Inferred from common names if omitted.
    feature_columns : list of str, optional
        Columns to use as features. Defaults to all numeric non-label columns.
    noise_values : tuple
        Label values that mean "noise"; these become ``-1``.
    standardize : bool, default=True
        Scale features to zero mean and unit variance. Recommended: AdaBox's
        density thresholds and DBSCAN's ``eps`` grid both assume comparable
        feature scales.
    reduce_to_2d : bool, default=True
        If more than 2 features remain, project to 2-D with PCA. AdaBox is a
        2-D algorithm, so this is required for more than two features.
    verbose : bool, default=True
        Print the report.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)
        Integer labels, ``-1`` for noise.
    report : PreparationReport
    """
    report = PreparationReport()

    if isinstance(data, np.ndarray) and y is not None:
        X_raw = np.asarray(data, dtype=float)
        y_raw = np.asarray(y)
        report.feature_names = [f"feature_{i}" for i in range(X_raw.shape[1])]
    else:
        X_raw, y_raw = _load_source(data, label_column, feature_columns, report)

    if X_raw.ndim != 2:
        raise ValueError(f"Features must be 2-D, got shape {X_raw.shape}.")

    report.n_rows_in = len(X_raw)
    report.n_features_in = X_raw.shape[1]

    if len(X_raw) != len(y_raw):
        raise ValueError(
            f"Features and labels have different lengths: "
            f"{len(X_raw)} vs {len(y_raw)}."
        )

    # ---- labels to integers ----
    y_enc = _encode_labels(y_raw, noise_values, report)

    # ---- drop rows with missing / infinite values ----
    finite = np.isfinite(X_raw).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        X_raw, y_enc = X_raw[finite], y_enc[finite]
        report.add(f"removed {n_bad} rows with missing or infinite values")

    if len(X_raw) < 10:
        raise ValueError(
            f"Only {len(X_raw)} usable rows remain — too few to cluster."
        )

    # ---- standardise ----
    if standardize:
        from sklearn.preprocessing import StandardScaler
        X_raw = StandardScaler().fit_transform(X_raw)
        report.add("standardised features (zero mean, unit variance)")

    # ---- reduce to 2-D ----
    if X_raw.shape[1] > 2:
        if not reduce_to_2d:
            raise ValueError(
                f"AdaBox needs 2-D input but the data has {X_raw.shape[1]} "
                "features. Set reduce_to_2d=True or select 2 feature columns."
            )
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        X_raw = pca.fit_transform(X_raw)
        report.pca_variance = float(pca.explained_variance_ratio_.sum())
        report.add(
            f"reduced {report.n_features_in} features to 2-D with PCA "
            f"({report.pca_variance:.1%} of variance retained)"
        )
    elif X_raw.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 feature columns, found {X_raw.shape[1]}."
        )

    report.n_rows_out = len(X_raw)
    report.n_clusters = int(len(set(y_enc[y_enc >= 0])))
    report.n_noise = int((y_enc == -1).sum())

    if report.n_clusters < 2:
        raise ValueError(
            f"Found only {report.n_clusters} cluster(s) in the labels. "
            "At least 2 are needed for a meaningful comparison."
        )
    if report.n_clusters > max(50, 0.5 * report.n_rows_out):
        raise ValueError(
            f"The label column '{report.label_name}' has {report.n_clusters} "
            f"distinct values across {report.n_rows_out} rows, which does not "
            "look like cluster labels (it may be coordinates or an ID). "
            "Pass the correct column with label_column='...'."
        )

    if verbose:
        print(report)

    return np.ascontiguousarray(X_raw, dtype=float), y_enc, report
