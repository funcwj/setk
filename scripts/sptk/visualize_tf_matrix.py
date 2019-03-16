#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

from libs.data_handler import ArchiveReader
from libs.utils import get_logger, filekey

logger = get_logger(__name__)


class NumpyReader(object):
    def __init__(self, src_dir):
        if not os.path.isdir(src_dir):
            raise RuntimeError("NumpyReader expect dir as input")
        flist = glob.glob(os.path.join(src_dir, "*.npy"))
        self.index_dict = {filekey(f): f for f in flist}

    def __iter__(self):
        for key, path in self.index_dict.items():
            yield key, np.load(path)


def save_figure(key, mat, dest, cmap="jet", shift=10, frequency=16000):
    num_frames, num_bins = mat.shape
    plt.figure()
    plt.imshow(
        mat.T, origin="lower", cmap=cmap, aspect="auto", interpolation="none")
    plt.title(key)
    xp = np.linspace(0, num_frames - 1, 5)
    yp = np.linspace(0, num_bins - 1, 6)
    plt.xticks(xp, ["{:.2f}".format(t) for t in (xp * shift)])
    plt.yticks(
        yp,
        ["{:.1f}".format(t) for t in np.linspace(0, frequency / 2, 6) / 1000])
    plt.xlabel('Time(s)')
    plt.ylabel('Frequency(kHz)')
    plt.savefig(dest)
    plt.close()
    logger.info('Save utterance {} to {}.png'.format(key, dest))


def run(args):
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    is_dir = os.path.isdir(args.rspec_or_dir)
    # ndarrays or archives
    mat_reader = ArchiveReader(
        args.rspec_or_dir) if not is_dir else NumpyReader(args.rspec_or_dir)
    for key, mat in mat_reader:
        if args.apply_log:
            mat = np.log10(mat)
        if args.trans:
            mat = np.transpose(mat)
        if args.norm:
            mat = mat / np.max(np.abs(mat))
        save_figure(
            key,
            mat,
            os.path.join(args.cache_dir, key.replace('.', '-')),
            cmap=args.cmap,
            shift=args.frame_shift * 1e-3,
            frequency=args.frequency)


# now support input from stdin
# shuf mask.scp | head | copy-feats scp:- ark:- | ./scripts/sptk/visualize_tf_matrix.py -
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to visualize kaldi's features/numpy's ndarray on T-F domain. "
        "egs: spectral/spatial features or T-F mask. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "rspec_or_dir",
        type=str,
        help="Read specifier of archives or directory of ndarrays")
    parser.add_argument(
        "--frame-hop", type=int, default=16, help="Frame shift in ms")
    parser.add_argument(
        "--frequency", type=int, default=16000, help="Sample frequency(Hz)")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="figure",
        help="Directory to cache pictures")
    parser.add_argument(
        "--apply-log", action="store_true", help="Apply log on input features")
    parser.add_argument(
        "--trans",
        action="store_true",
        help="Apply matrix transpose on input features")
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Normalize values in [-1, 1] before visualization")
    parser.add_argument(
        "--cmap",
        choices=["binary", "jet", "hot"],
        default="jet",
        help="Colormap used when save figures")
    args = parser.parse_args()
    run(args)
