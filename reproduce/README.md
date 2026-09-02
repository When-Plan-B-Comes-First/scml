# reproduce/

```bash
python reproduce/run_all.py          # low-D  (a few minutes)
python reproduce/highd/run_all.py    # high-D (10-15 minutes)
```

Run from a clone of this repository — `reproduce/` ships with the repo, not
with the installed package.

## Low-D: two experiments, two regimes

The split is deliberate. SLCD is a **scaling** tool; using it on small data
makes AdaBox look worse than it is, and using direct tuning on huge data is
impractical. Each experiment therefore runs in the regime the documentation
actually recommends.

**Experiment 1 — small data (≤ 5,000 points).** AdaBox tuned directly with
exhaustive Grid Search, no SLCD, against grid-searched DBSCAN. This is the
protocol behind the published results.

**Experiment 2 — large data (30k–50k points).** SLCD tuned on a 500-point
sample, then deployed to the full dataset. The number that matters is the
**transfer gap**: sample score minus full-data score. A gap near zero means the
parameters learned from ~1% of the data work on all of it.

Outputs land in `outputs/`:

- `small_data_direct_tuning.csv` / `small_data_adabox_vs_dbscan.png`
- `large_data_slcd_transfer.csv` / `large_data_slcd_transfer.png`
- `scope_components.png`, `summary.txt`
