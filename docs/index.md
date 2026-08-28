# SC-ML documentation

This is the documentation home for the `scml` package. The low-D track is live;
the high-D track is coming soon.

## Install

```bash
pip install scml
```

## Low-D API

- `scml.lowd.AdaBox` — adaptive density-based box clustering (sklearn-style).
- `scml.lowd.SLCD` — Sample → Label → Calibrate → Deploy parameter transfer
  (the low-D instantiation of SLCD; the high-D track uses
  Sample → Learn → Classify → Deploy).
- `scml.lowd.scope_score(X, y_true, y_pred)` — overall SCOPE score (the
  tuning objective, not only a reporting metric).
- `scml.lowd.scope_report(X, y_true, y_pred)` — five-component breakdown.
- `scml.lowd.benchmark_dataset(data)` — one-call comparison against
  DBSCAN / OPTICS / HDBSCAN on your own dataset.

See the project [README](../README.md) for the 60-second quickstart and
`examples/` for runnable scripts.

## Reproduce

```bash
pip install scml[reproduce]
python reproduce/run_all.py
```
