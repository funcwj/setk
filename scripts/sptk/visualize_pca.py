#!/usr/bin/env python
# wujian@2018

import os
import glob
import argparse

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from libs.data_handler import ArchiveReader
from libs.utils import filekey


class NumpyReader(object):
    def __init__(self, src_dir):
        if not os.path.isdir(src_dir):
            raise RuntimeError("NumpyReader expect dir as input")
        flist = glob.glob(os.path.join(src_dir, "*.npy"))
        self.index_dict = {filekey(f): f for f in flist}

    def __iter__(self):
        for key, path in self.index_dict.items():
            yield key, np.load(path)


def run(args):
    pca = PCA(n_components=args.dim)

    is_dir = os.path.isdir(args.rspec_or_dir)
    samples = []
    feats_reader = ArchiveReader(
        args.rspec_or_dir,
        matrix=(args.input == "matrix")) if not is_dir else NumpyReader(
            args.rspec_or_dir)
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
    parser.add_argument("--input",
                        type=str,
                        default="vector",
                        choices=["matrix", "vector"],
                        help="Input data type")
    parser.add_argument("--dim",
                        type=int,
                        default=3,
                        help="Number of components in PCA transform")
    args = parser.parse_args()
    run(args)
