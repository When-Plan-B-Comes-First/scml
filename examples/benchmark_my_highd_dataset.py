"""
benchmark_my_highd_dataset.py -- see how AdaGraph handles YOUR high-dimensional
data, against HDBSCAN, K-Means and Ward.

Run as-is for a demo:

    python examples/benchmark_my_highd_dataset.py

Or point it at your own CSV:

    python examples/benchmark_my_highd_dataset.py my_data.csv

    python examples/benchmark_my_highd_dataset.py my_data.csv my_label_column

Your CSV needs numeric feature columns (any number of them -- they are all
kept) plus one ground-truth column named something recognisable ('label',
'class', 'target', 'cluster', ...).

Unlike the low-D benchmark, nothing is reduced to 2-D: AdaGraph clusters in
your data's native dimensionality. PCA appears only in the plot, for viewing.
"""

import sys

from scml.highd import benchmark_highd_dataset


def main():
    args = sys.argv[1:]
    if args:
        results = benchmark_highd_dataset(
            args[0],
            label_column=args[1] if len(args) > 1 else None,
            save_dir="highd_benchmark_output",
        )
    else:
        from sklearn.datasets import make_blobs
        print("No dataset given - running a demo on 25-dimensional blobs.\n"
              "Pass a CSV path to benchmark your own data.\n")
        X, y = make_blobs(n_samples=1200, n_features=25, centers=6,
                          cluster_std=2.5, random_state=0)
        results = benchmark_highd_dataset(
            X, y=y, dataset_name="demo_25D",
            save_dir="highd_benchmark_output")

    print("\nTable and plots written to highd_benchmark_output/")
    return results


if __name__ == "__main__":
    main()
