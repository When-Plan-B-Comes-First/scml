# SC-ML implementation specification (authoritative)

This file is the single source of truth for how `scml` must behave. It is
derived from the SLCD paper (the canonical reference) plus explicit rulings from
the author. When the paper, notebook code, and any other source conflict, **the
SLCD paper wins.**

Do not "improve", approximate, or substitute alternative designs. The defaults
below were chosen through hundreds of experiments; reproduce them exactly.

---

## The two paths (decided by dataset size)

AdaBox's power is its tuning. There are two distinct tuning paths, selected by
the size of the dataset being clustered.

### Path A — small-to-medium data (≤ 5,000 points): no SLCD

- **No sampling, no parameter transfer, no cascade.** Tune AdaBox directly on
  the full dataset. This is the exact protocol of the published 62-dataset
  benchmark (RQ1), and this implementation reproduces those results to three
  decimal places (verified: Hospital_PatientRisk 0.729, NYC_TaxiZones 0.778,
  Healthcare_PatientGroups 0.667, CustomerSegmentation_Retail 0.618). GS on
  5,000 points takes ~1–2 minutes.
- **Tuning method: Grid Search** (the full Cartesian product of the search
  space). Random Search is not used here.
- **Density mode: absolute** (`use_relative_density = False`). Relative density
  exists to make parameters transfer across scale; with no transfer, it adds
  nothing, so absolute density is used.
- The three-phase internal optimization (below) still runs: Phase 1 (GS search)
  → Phase 2 (post-hoc) → Phase 3 (anti-fragmentation).

### Path B — large data (> 5,000 points; SLCD shines past ~10,000): SLCD

SLCD = **Sample → Label → Calibrate → Deploy** (four stages).

- **Stage 1 — Sample:** stratified sample preserving cluster proportions
  (≥ 2 points per cluster; noise sampled proportionally).
- **Stage 2 — Label:** tune on the sample with the three-phase optimization,
  scored by SCOPE. **Tuning method: Random Search (default).** RS achieves
  equivalent quality to GS at ~10% of the cost (paper §4.2).
- **Stage 3 — Calibrate:** cascading validation on progressively larger,
  non-overlapping samples; retune via neighbourhood RS only when degradation is
  detected.
- **Stage 4 — Deploy:** freeze parameters, fit the full dataset unmodified.
- **Density mode: relative / scale-invariant** (`use_relative_density = True`).
  This is the property that makes transfer work and is the paper's central
  thesis.

---

## The three-phase internal optimization (the "Label" stage)

Runs in both paths (the only difference is GS vs RS in Phase 1). From paper §4.2:

1. **Phase 1 — Parameter search.** Evaluate configurations from the search space
   (all of them for GS; `n_trials` sampled uniformly for RS), scored by SCOPE.
   Keep the best.
2. **Phase 2 — Post-hoc refinement.** Re-evaluate the Phase 1 winner across all
   6 combinations of `min_cluster_size ∈ {1, 5, 10}` × `merge_adjacent ∈
   {True, False}`. Decouples structural params from the density search.
3. **Phase 3 — Anti-fragmentation.** If predicted cluster count exceeds
   `n_true_clusters × fragmentation_threshold`, iteratively merge the closest
   adjacent cluster pairs (up to 15 rounds). Each merge is scored by SCOPE and
   accepted only if the score improves.

---

## AdaBox search space (six parameters; paper §4.2, notebook-confirmed)

```
n_boxes                  : [15, 20, 25, 30, 35, 40, 45, 50, 60]
regular_threshold_factor : [0.7, 1.1]
merge_adjacent           : [True, False]
refinement_sigma         : [0.5, 1.0, 1.5, 2.0, 2.5]
min_cluster_size         : [1, 5, 10]
# density parameter depends on mode:
#   absolute mode : min_density            ∈ [1, 2, 3, 4, 5]
#   relative mode : relative_density_param ∈ [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
```

---

## Defaults (author rulings; paper-consistent)

| Setting | Small data (< 1k) | Large data (> 1k) |
|---|---|---|
| Use SLCD | No | Yes |
| Tuning method | Grid Search | Random Search |
| Density mode | absolute | relative (scale-invariant) |
| RS trials | n/a | 100 if < 20k; 200 if > 20k (user-overridable) |
| Cascade stages | n/a | 1 stage if < 20k; 2 stages if > 20k |

Other fixed defaults (paper §4.6 / §5.1): `random_state = 42`;
`fragmentation_threshold = 1.1` in the SLCD path (the earlier RQ1 benchmark code used 1.5; the direct-GS path reproduces RQ1 exactly with 1.5); degradation retune trigger: ARI drop > 0.01 OR deployed ARI < 0.4;
neighbourhood RS = 100 trials at neighbourhood scale 0.3; SCOPE is the tuning
objective; core fraction uses k-NN density with the SCOPE v5 settings.

---

## SCOPE (paper §4.2, Paper 1 of the series)

SCOPE = **Structured Clustering Optimization via Performance Evaluation**. The
name is deliberate: SCOPE's primary role is as a tuning *objective*, not merely
a reporting metric. It is a better optimizer signal than ARI, and that is the
claim to preserve in all public wording — do not describe it as "just another
clustering metric".

Weighted five-component composite: Core Purity (25%), Boundary Recall (25%),
Cluster Precision (20%), Noise F1 (20%), Cluster Count Accuracy (10%). Core
identification uses k-NN local density; cluster matching uses the Hungarian
algorithm. This is the only tuning objective, for AdaBox and for every baseline
it is compared against.

---

## Naming (author rulings)

- Public algorithm names: **AdaBox** (low-D) and **AdaGraph** (high-D) only.
- "AdaHD" must never appear in any public-facing place.
- The acronym **SLCD** names a family with a track-specific expansion. Both
  expansions are correct and must be used with their own track; they are not
  interchangeable, and neither is a typo for the other:
    * Low-D  (AdaBox):   **Sample-Label-Calibrate-Deploy** — Deploy = parameter
      transfer. Matches U.S. application 63/969,070.
    * High-D (AdaGraph): **Sample-Learn-Classify-Deploy** — Deploy = assigning
      remaining points to the sample's clusters by voting. Matches U.S.
      application 64/056,834.
  The shared invariant, and the reason both are called SLCD: the full dataset
  is never tuned.

---

## The Calibrate cascade (Stage 3) — exact mechanics

For each cascade stage beyond Stage 1:
1. Draw a fresh stratified sample, **non-overlapping** with all prior samples
   (`stratified_sample_excluding`), of twice the previous stage's size.
2. Deploy the current best parameters to it and compute SCOPE.
3. Degradation is measured in **ARI**. If `ARI drop > degradation_threshold`
   (default **0.01**) OR the deployed ARI falls below the **0.4 quality floor**
   (the paper's two-condition trigger),
   retune via **neighbourhood Random Search** (`slcd_retune_adabox_rs`):
   `retune_trials` (default 100) perturbations around the current params at
   `neighborhood_scale` 0.3, scored by SCOPE, with Phase 3 anti-fragmentation.
   Accept the retune only if it **strictly improves ARI** on this sample.
4. Otherwise accept the deployed parameters unchanged.

Cascade depth defaults: **1 stage if < 20k points, 2 stages if > 20k** (paper
recommends stopping at 2; deeper cascades overfit to intermediate samples).

---

## Sample-size defaults (size tiers x cluster-count floor)

`default_sample_size(n_points, n_clusters)` returns `max(size_tier,
cluster_floor)`, capped at `n_points`.

Size tiers (grow with deployment scale to hold coverage ratio):

| dataset size | base sample |
|---|---|
| < 50,000 | 200 |
| < 500,000 | 800 |
| < 2,000,000 | 1,500 |
| >= 2,000,000 | 5,000 |

Cluster-count floor: if `n_clusters > 5`, raise to `80 * n_clusters`. Small
samples cannot represent many clusters (paper: >15 clusters need larger
samples; experiments show ~80 points/cluster transfers well).

All sample sizes are **user-overridable** via `SLCD(sample_size=...)`.


---

## Phase 1 Random Search sampling (exact mechanics)

RS does NOT sample each parameter independently. It enumerates the full
Cartesian product of valid combinations, then draws `n_trials` of them
uniformly **without replacement** using `np.random.RandomState(random_state)`.
This matches the original notebooks bit-for-bit in structure and avoids
duplicate trials.

---

## Validation status

The direct-GS path (`AdaBox().tune()` for ≤ 5,000 points) reproduces the
original RQ1 benchmark results **exactly (3 decimals)** on every dataset
tested. Any future change that breaks this exact reproduction is a regression.

---

## Benchmark module (single-dataset comparison)

`scml.lowd.benchmark_dataset` reproduces the RQ1 comparison protocol for one
user-supplied dataset:

- **AdaBox** is tuned by its normal routing (direct GS + absolute density at
  ≤5,000 points; SLCD above that).
- **DBSCAN / OPTICS / HDBSCAN** each get the full RQ1 grid, scored with the
  same objective (SCOPE). This fairness rule must not be relaxed — comparing a
  tuned AdaBox against default-parameter baselines would invalidate the claim.
- Reported columns match the RQ1 results table: ARI, SCOPE_Overall,
  Core_Purity, Boundary_Recall, Cluster_Precision, Noise_F1, Count_Accuracy.
- Plots match the notebook: grouped metric bars, and ground-truth-vs-algorithm
  scatter panels with `ARI | SCOPE` in each subplot title.
- HDBSCAN uses scikit-learn's built-in implementation when available, falling
  back to the standalone `hdbscan` package.

`scml.lowd.prepare_dataset` guards the input. It must never silently accept a
coordinate column (e.g. one named `y`) as ground truth; label candidates are
checked in priority order and validated for plausibility, and an implausible
cluster count raises rather than producing a meaningless benchmark.


---

# High-D track specification

## Algorithms

- **AdaGraph** — graph-native adaptive density clustering. Operates in the
  original dimensionality; never reduces to 2-D first. Internal class name
  `AdaBoxGraph` is retained in `_engine.py`; the public estimator is
  `scml.highd.AdaGraph`. **"AdaHD" must never appear in any public-facing
  place.**
- **Graph-SCOPE** — unsupervised structural validation index. Five components:
  modularity 60% (Reichardt-Bornholdt, gamma=1.5), boundary sharpness 10%,
  internal consistency 20%, noise legitimacy 5%, partition balance 5%.
  Component C4 defaults to graph cohesion (works on any algorithm's labels);
  passing `relative_densities` switches it to the density criterion, which is
  only available for AdaGraph output. Both agree exactly when a clustering
  marks no noise.
- **High-D SLCD** — Sample → Learn → Classify → Deploy. Density-aware sample,
  400-trial random search over 12 parameters scored by SCOPE, then two-pass
  kNN vote (`prototype_deploy`) to assign the remaining points.

## Cross-track dependency

The high-D engine imports supervised SCOPE from the low-D track
(`from ..lowd.scope import compute_dice_metrics`). This is intentional and is
one reason the tracks live in a single repository.

## Methodological rules (must not be relaxed)

1. **Graph-SCOPE is a signal, never a judge.** It selects; supervised SCOPE
   and ARI judge against held-out labels. Letting it do both is circular.
2. **Baselines are tuned on both objectives.** `benchmark_highd` tunes every
   algorithm once against SCOPE and once against ARI, and reports both, so no
   method benefits from the headline metric being the one it optimised.
3. **Negative control.** `compare_k_selection` scores a shuffled labelling; if
   its ARI is not ~0 the harness is broken and results are void.
4. **The SuperCon result is withdrawn** and must not appear in any benchmark,
   reproduce target, or documentation.
5. **Report the non-monotonic advantage honestly.** Graph-SCOPE's edge over
   Silhouette peaks in the mid-noise regime and vanishes when the problem is
   unsolvable for all methods. Quote peak and endpoint together.

## Default sample sizes (high-D SLCD)

| dataset size | sample |
|---|---|
| <= 2,000 | all points (cluster directly) |
| < 20,000 | 1,000 |
| < 200,000 | 2,000 |
| >= 200,000 | 5,000 |
