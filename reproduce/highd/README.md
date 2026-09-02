# reproduce/highd

Regenerates the high-D track's headline result offline:

```bash
python reproduce/highd/run_all.py
```

Runs entirely on synthetic data — no network, no downloads, no API keys.

**Experiment 1 (primary).** Signal lives in 5 dimensions; pure-noise
dimensions are appended in increasing numbers. As noise accumulates, Euclidean
distance is progressively dominated by it, so Silhouette should degrade. If
Graph-SCOPE degrades more slowly, that is evidence for high-dimensional
structural fidelity.

**Experiment 2 (control).** Dimension grows but every dimension carries
signal. If the mechanism is really noise-domination of distance, the advantage
should *not* appear here. Without this control the sweep alone proves nothing.

Outputs land in `outputs/`: per-seed CSVs, summaries, a figure, and a verdict
file.

## Reading the result honestly

The advantage is **non-monotonic**. It is large in the mid-noise regime and
disappears once the problem becomes unsolvable for every method. The verdict
reports both the peak and the endpoint; quoting either alone would misrepresent
the finding.
