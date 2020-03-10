#!/usr/bin/env python
# wujian@2018

import os
import glob
import argparse

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from libs.data_handler import ArchiveReader, DirReader
from libs.utils import filekey


class NumpyReader(DirReader):
    """
    Numpy matrix reader
    """
    def __init__(self, obj_dir):
        super(NumpyReader, self).__init__(obj_dir, "npy")

    def _load(self, key):
        return np.load(self.index_dict[key])


def run(args):
    pca = PCA(n_components=args.dim)

    is_dir = os.path.isdir(args.rspec_or_dir)
    samples = []
    feats_reader = ArchiveReader(
        args.rspec_or_dir) if not is_dir else NumpyReader(args.rspec_or_dir)
    for _, feats in feats_reader:
        if feats.ndim != 1:
            feats = np.average(feats, 0)
        samples.append(feats)
    org_mat = np.stack(samples)
    pca_mat = pca.fit_transform(org_mat)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = np.split(pca_mat, 3, axis=1)
    ax.scatter(x, y, z, s=2)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to visualize embeddings (egs: ivector/xvector/dvector) "
        "using PCA transform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("rspec_or_dir",
                        type=str,
                        help="Read specifier of "
                        "archives/Directory of ndarrays")
    parser.add_argument("--dim",
                        type=int,
                        default=3,
                        help="Number of components in PCA transform")
    args = parser.parse_args()
    run(args)
