# scml — Structure-Centric Machine Learning

[![status: low-D track live](https://img.shields.io/badge/low--D%20track-live-brightgreen)]()
[![status: high-D track coming soon](https://img.shields.io/badge/high--D%20track-coming%20soon-blue)]()

**Structure-Centric Machine Learning (SC-ML)** is a clustering paradigm that
evaluates and produces clustering by recovering *structure* — cores,
boundaries, and noise — rather than chasing exact label matches. This is the
one unified home for its tools: one install, one docs home, one citation
target.

The paradigm ships in two tracks:

| Track | Components | Status |
|-------|-----------|--------|
| **Low-D** | SCOPE (metric) · AdaBox (clustering) · SLCD (parameter transfer) | **Live** |
| **High-D** | Graph-SCOPE · AdaGraph · DA-Sampler | Coming soon |

---

## 60-second quickstart

```bash
pip install git+https://github.com/When-Plan-B-Comes-First/scml.git
```

<sub>Installs straight from GitHub. A shorter `pip install scml` will work once
the package is published to PyPI.</sub>

```python
from sklearn.datasets import make_blobs
from scml.lowd import AdaBox, scope_score

X, y = make_blobs(n_samples=800, centers=4, cluster_std=1.0, random_state=0)

labels = AdaBox().tune(X, y).labels_      # three-phase calibration
print("SCOPE:", round(scope_score(X, y, labels), 3))   # ~0.95
```

```
SCOPE: 0.945
```

`AdaBox().tune(X, y)` runs the three-phase calibration that gives AdaBox its
strength. `fit`, `fit_predict`, `predict`, and `score` follow scikit-learn
conventions, so AdaBox drops into any workflow you already have.

> **Tuning is not optional polish — it is the algorithm.** A single hand-picked
> parameter set (plain `AdaBox(...).fit(X)`) behaves like a generic density
> clusterer and won't match DBSCAN/HDBSCAN. The three-phase calibration
> (parameter search → `min_cluster_size`/merge refinement → anti-fragmentation)
> is what makes AdaBox competitive. Up to ~5,000 points `.tune()` uses an
> exhaustive Grid Search on the full data; beyond that, use `SLCD`, which tunes
> on a sample with Random Search and transfers the parameters at scale.

---

## What's in the low-D track

**SCOPE** — *Structured Clustering Optimization via Performance Evaluation.* A
structure-aware quality metric that decomposes a clustering into five
interpretable components (core purity, boundary recall, cluster precision,
noise F1, cluster-count accuracy) and reports a single `[0, 1]` overall score.
Unlike ARI, it rewards recovering the right *structure*.

```python
from scml.lowd import scope_report
report = scope_report(X, y, labels)   # full five-component breakdown
```

**AdaBox** — adaptive density-based box clustering for 2-D data. Finds
arbitrarily shaped clusters via an adaptive grid of density boxes with Gaussian
boundary refinement. Its power comes from a three-phase calibration (parameter
search → `min_cluster_size`/merge refinement → anti-fragmentation), reached via
`.tune(X, y)` or the `SLCD` workflow — not from any single fixed parameter set.

**SLCD** — *Sample → Label → Calibrate → Deploy.* For datasets larger than
~5,000 points, SLCD tunes AdaBox on a small stratified sample using Random
Search and scale-invariant relative density, then deploys the frozen parameters
to the full dataset. Because the parameters encode structure rather than
distance, they transfer across orders of magnitude of scale.

```python
from scml.lowd import SLCD
labels = SLCD(sample_size=500).fit_predict(X_large, y_large)
```

> **When to use which.** Up to ~5,000 points, use `AdaBox().tune(X, y)` — it
> runs an exhaustive Grid Search with absolute density directly on your full
> data (1–2 minutes at 5k; this is the exact protocol behind the published
> benchmark results). Beyond ~5,000 points, use `SLCD` — it exists to make
> AdaBox's large 6-parameter search tractable at scale, and its value grows
> with size (it shines past ~10,000 points). SLCD is a scaling solution, not a
> replacement for full tuning: on data small enough to tune directly, direct
> tuning wins. SLCD picks its Random-Search trial count (100 below 20k, 200
> above) and cascade depth (1 below 20k, 2 above) automatically.

> SLCD has **separate** low-D and high-D implementations — same idea, different
> code. The low-D one lives in `scml.lowd`; the high-D one will live in
> `scml.highd`.

---

## Benchmark your own dataset

Want to see how AdaBox handles *your* data? One command runs AdaBox and the
baselines on it and reports the comparison:

```bash
pip install "scml[benchmark] @ git+https://github.com/When-Plan-B-Comes-First/scml.git"
```

```python
from scml.lowd import benchmark_dataset

results = benchmark_dataset("my_data.csv")
```

That single call cleans the data, tunes every algorithm, prints a results
table, and draws two plots: a grouped bar chart of all metrics, and a
side-by-side scatter of ground truth against each algorithm's clustering.

```
Algorithm   ARI  SCOPE_Overall  Core_Purity  Boundary_Recall  ...  K_found  Time_s
   AdaBox 0.948          0.988        0.982            0.991  ...        4    18.3
  HDBSCAN 0.940          0.785        0.788            0.700  ...        4     1.0
   DBSCAN 0.936          0.784        0.787            0.699  ...        4     0.4
   OPTICS 0.891          0.636        0.560            0.520  ...        3    19.4

Best by SCOPE: AdaBox (0.988)
```

**The comparison is fair by construction.** Every baseline gets a full grid
search over its own parameters, scored with the same objective used to tune
AdaBox — no method is left at default settings while another is tuned.

**Your data doesn't need to be tidy.** The loader accepts a CSV, a pandas
DataFrame, or numpy arrays, and handles the rest: it finds the label column,
drops non-numeric and identifier columns, removes rows with missing or
infinite values, standardises the features, reduces more than two features to
2-D with PCA, and re-encodes labels to integers with `-1` for noise. It prints
a report of everything it changed, and refuses to run rather than silently
producing a meaningless result if it can't identify real cluster labels.

Requirements: numeric feature columns, plus one ground-truth column named
something recognisable (`label`, `class`, `target`, `cluster`, ...). If yours is
named differently, say so:

```python
results = benchmark_dataset("my_data.csv", label_column="my_labels",
                            save_dir="benchmark_output")
```

Or from the command line:

```bash
python examples/benchmark_my_dataset.py my_data.csv
```

---

## Reproduce the paper results

The `reproduce/` directory regenerates the figures and tables from the paper
with a single command:

```bash
pip install "scml[reproduce] @ git+https://github.com/When-Plan-B-Comes-First/scml.git"
python reproduce/run_all.py
```

Outputs land in `reproduce/outputs/`. This is the validation proof: clone,
install, run one command, see the numbers.

---

## High-D track (coming soon)

The high-D track extends SC-ML to graph-structured and high-dimensional data
with **Graph-SCOPE**, **AdaGraph**, and **DA-Sampler**. The code is in
preparation; each folder under `src/scml/highd/` has a short README pointing to
the relevant arXiv preprint.

---

## Licensing & citation

- **License:** source-available — free for research and non-commercial use,
  paid for commercial use. See [LICENSE](LICENSE). *(Final terms pending IP
  attorney review prior to public release.)*
- **Patents:** see [PATENTS.md](PATENTS.md).
- **Citation:** see [CITATION.cff](CITATION.cff) or the "Cite this repository"
  button on GitHub.

## Links

- Website: <https://structurecentricml.com>
- Preprints: <https://arxiv.org/a/elmahdi_a_1>
