# scml — Structure-Centric Machine Learning

[![status: low-D track live](https://img.shields.io/badge/low--D%20track-live-brightgreen)]()
[![status: high-D track live](https://img.shields.io/badge/high--D%20track-live-brightgreen)]()

**Structure-Centric Machine Learning (SC-ML)** is a clustering paradigm that
evaluates and produces clustering by recovering *structure* — cores,
boundaries, and noise — rather than chasing exact label matches. This is the
one unified home for its tools: one install, one docs home, one citation
target.

The paradigm ships in two tracks:

| Track | Components | Status |
|-------|-----------|--------|
| **Low-D** | SCOPE (metric) · AdaBox (clustering) · SLCD (parameter transfer) | **Live** |
| **High-D** | AdaGraph (clustering) · Graph-SCOPE (unsupervised metric) · SLCD | **Live** |

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
structure-aware *optimization objective*, not just a report-time score. It
decomposes a clustering into five interpretable components (core purity,
boundary recall, cluster precision, noise F1, cluster-count accuracy) and
combines them into a single `[0, 1]` value.

Because it rewards recovering the right *structure* rather than exact label
matching, SCOPE is a better signal than ARI for driving hyperparameter search —
it is the objective every algorithm in this repo is tuned against. Reporting it
is the secondary use; optimizing against it is the point.

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
distances, they transfer across orders of magnitude of scale.

```python
from scml.lowd import SLCD
labels = SLCD().fit_predict(X_large, y_large)
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


### SLCD is a family, not one algorithm

Both tracks avoid the thing that makes large-scale density clustering
impractical: **neither ever tunes the full dataset.** That shared invariant is
what SLCD names. The mechanism differs by track, and the two implementations
are *not* interchangeable:

| | Low-D (shipping here) | High-D (coming soon) |
|---|---|---|
| Expansion | Sample → **Label** → **Calibrate** → Deploy | Sample → **Learn** → **Classify** → Deploy |
| Algorithm | AdaBox | AdaGraph |
| What "Deploy" means | **Parameter transfer** — parameters tuned on the sample cluster the full dataset | **Point assignment** — AdaGraph clusters the sample, then remaining points are assigned to those clusters by voting |

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

## High-D track

AdaGraph clusters **in the original dimensionality** — no PCA, no UMAP, no
projection to 2-D first. In high dimensions absolute distances stop being
informative, but *relative neighbourhoods* still are, so AdaGraph builds a
k-nearest-neighbour graph and runs adaptive density clustering on that.

```python
from scml.highd import AdaGraph, graph_scope_score

labels = AdaGraph().tune(X, y, n_trials=400).labels_
```

**Graph-SCOPE** is an unsupervised structural index — it judges a clustering
from graph topology alone, with **no ground truth**. Its natural comparison is
Silhouette, not ARI, and it works on the output of any clustering algorithm:

```python
from scml.highd import graph_scope_score, graph_scope_report
graph_scope_score(X, labels)      # no y needed
graph_scope_report(X, labels)     # five-component breakdown
```

> **Signal, not judge.** Graph-SCOPE is a *selection* signal. Using it to both
> choose a clustering and then pronounce that clustering good is circular.
> Judge with supervised SCOPE or ARI against held-out labels.

**High-D SLCD** — *Sample → Learn → Classify → Deploy.* AdaGraph is tuned on a
density-aware sample, then the remaining points are assigned to the sample's
clusters by a two-pass kNN vote:

```python
from scml.highd import SLCD
labels = SLCD(n_trials=400).fit_predict(X_large, y_large)
```

### SLCD is a family, not one algorithm

Both tracks share the invariant that gives SLCD its value — **neither ever
tunes the full dataset** — but they reach it differently, and the two
implementations are not interchangeable:

| | Low-D | High-D |
|---|---|---|
| Expansion | Sample → **Label** → **Calibrate** → Deploy | Sample → **Learn** → **Classify** → Deploy |
| Algorithm | AdaBox | AdaGraph |
| Sampling | stratified | density-aware (preserves rare modes) |
| What "Deploy" means | **parameter transfer** — sample-tuned parameters cluster the full data | **point assignment** — the sample is clustered, remaining points join by kNN vote |

### Benchmark your own high-D dataset

Same one-call experience as the low-D track — hand it a CSV:

```python
from scml.highd import benchmark_highd_dataset

results = benchmark_highd_dataset("my_data.csv")
```

That cleans the data, runs AdaGraph against HDBSCAN, K-Means and Ward, prints
the table, and draws two plots. **Every dimension is kept** — the loader drops
junk columns, missing rows and zero-variance features, but never reduces
dimensionality, because clustering natively is the point. PCA appears only in
the plot, for viewing.

Or from the command line:

```bash
python examples/benchmark_my_highd_dataset.py my_data.csv
```

If you already have arrays, or want the k-selection comparison:

```python
from scml.highd import benchmark_highd, compare_k_selection

benchmark_highd(X, y)        # AdaGraph vs HDBSCAN / K-Means / Ward
compare_k_selection(X, y)    # Graph-SCOPE vs Silhouette for choosing k
```

`benchmark_highd` tunes every algorithm **twice — once on SCOPE and once on
ARI — and reports both**, so a win can't be an artifact of the headline metric
happening to be the one a method optimised. `compare_k_selection` includes a
shuffled-label negative control: if it doesn't score ~0, the harness is broken
and the results are void.

---

## Licensing & citation

- **License:** [SC-ML Source-Available License 1.0](LICENSE) — free for
  research, teaching, evaluation, and other non-commercial use; a paid
  commercial license is required for commercial use.
- **Patents:** four U.S. patent applications filed in 2026 cover these methods.
  See [PATENTS.md](PATENTS.md). No patent rights are granted beyond
  non-commercial use.
- **Citation:** see [CITATION.cff](CITATION.cff), or use GitHub's
  "Cite this repository" button.

**Commercial licensing:** ahmed@structurecentricml.com

If you publish results produced with this software, please cite the AdaBox
paper and the repository.

## Links

- **Website:** <https://structurecentricml.com>
- **AdaBox paper (Low-D):** <https://arxiv.org/abs/2603.13339>
- **AdaGraph paper (High-D):** <https://arxiv.org/abs/2605.16320>
