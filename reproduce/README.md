# reproduce/

One command regenerates the low-D paper's headline figures and tables:

```bash
pip install scml[reproduce]
python reproduce/run_all.py
```

Outputs are written to `reproduce/outputs/`:

- `scope_vs_ari_table.csv` — per-dataset SCOPE vs ARI for AdaBox (via SLCD)
- `scope_components.png` — SCOPE five-component breakdown per dataset
- `summary.txt` — aggregate headline numbers

The script uses public scikit-learn datasets so it runs offline and anywhere.
