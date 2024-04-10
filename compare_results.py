import argparse
import numpy as np
import os

from scipy.stats import ttest_ind


def load_data(data_path):
    data = np.load(data_path)
    data = np.mean(data, axis=-1)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1")
    parser.add_argument("--path2")

    args = parser.parse_args()

    data_path1 = os.path.join(args.path1, "linearResults.npy")
    data_path2 = os.path.join(args.path2, "linearResults.npy")

    # First axis is the subject, second axis is the cv fold.
    results1 = load_data(data_path1)
    results2 = load_data(data_path2)

    nsub, ncv = results1.shape

    for i in range(nsub):
        res = ttest_ind(results1[i], results2[i])

        print(f"[Sub{i:2d}] - mean of results1: {np.mean(results1[i]):.3f}, mean of results2: {np.mean(results2[i]):.3f}, pvalue of T-test: {res.pvalue:.3f}")        # noqa: E501
