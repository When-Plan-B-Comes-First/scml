"""
benchmark_my_dataset.py -- point this at your own data and see how AdaBox
compares against DBSCAN, OPTICS and HDBSCAN.

Run it as-is for a demo on generated data:

    python examples/benchmark_my_dataset.py

Or give it your own CSV:

    python examples/benchmark_my_dataset.py my_data.csv

Your CSV needs numeric feature columns plus one column of ground-truth labels
named something recognisable ('label', 'class', 'target', 'cluster', ...). If
your label column has a different name, pass it as the second argument:

    python examples/benchmark_my_dataset.py my_data.csv my_label_column

Everything else -- dropping junk columns, removing missing values, scaling,
reducing to 2-D, encoding labels -- is handled for you.
"""

import sys

from scml.lowd import benchmark_dataset


def main():
    args = sys.argv[1:]

    if args:
        path = args[0]
        label_column = args[1] if len(args) > 1 else None
        results = benchmark_dataset(
            path,
            label_column=label_column,
            save_dir="benchmark_output",
        )
    else:
        # No file given: demonstrate on generated data.
        from sklearn.datasets import make_blobs
        print("No dataset given - running a demo on generated blobs.\n"
              "Pass a CSV path to benchmark your own data.\n")
        X, y = make_blobs(n_samples=600, centers=4, cluster_std=0.9,
                          random_state=1)
        results = benchmark_dataset(X, y=y, dataset_name="demo_blobs",
                                    save_dir="benchmark_output")

    print("\nTable and plots written to benchmark_output/")
    return results


if __name__ == "__main__":
    main()
