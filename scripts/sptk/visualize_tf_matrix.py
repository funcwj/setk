#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from libs.data_handler import ArchiveReader
from libs.utils import get_logger, filekey
from libs.opts import StrToBoolAction

logger = get_logger(__name__)


class NumpyReader(object):
    def __init__(self, src_dir):
        src_dir = Path(src_dir)
        if not src_dir.is_dir():
            raise RuntimeError("NumpyReader expect dir as input")
        flist = glob.glob((src_dir / "*.npy").as_posix())
        self.index_dict = {filekey(f): f for f in flist}

    def __iter__(self):
        for key, path in self.index_dict.items():
            yield key, np.load(path)


def save_figure(key, mat, dest, cmap="jet", hop=10, sr=16000, size=3):
    """
    Save figure to disk
    """
    def plot(mat, num_frames, num_bins, xticks=True):
        plt.imshow(mat.T,
                   origin="lower",
                   cmap=cmap,
                   aspect="auto",
                   interpolation="none")
        if xticks:
            xp = np.linspace(0, num_frames - 1, 5)
            plt.xticks(xp, ["{:.2f}".format(t) for t in (xp * hop)])
            plt.xlabel("Time(s)")
        else:
            # disble xticks
            plt.xticks([])
        yp = np.linspace(0, num_bins - 1, 6)
        fs = np.linspace(0, sr / 2, 6) / 1000
        plt.yticks(yp, ["{:.1f}".format(t) for t in fs])
        plt.ylabel("Frequency(kHz)")

    if mat.ndim == 3:
        N, T, F = mat.shape
    else:
        T, F = mat.shape
        N = 1
    plt.figure(figsize=(max(size * T / F, size) + 2, size + 2))
    if N != 1:
        for i in range(N):
            plt.subplot(int(f"{N}1{i + 1}"))
            plot(mat[i], T, F, xticks=i == N - 1)
    else:
        plot(mat, T, F)
    plt.savefig(dest)
    plt.close()
    logger.info(f"Save utterance {key} to {dest}.png")


def run(args):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    is_dir = Path(args.rspec_or_dir).is_dir()
    # ndarrays or archives
    mat_reader = ArchiveReader(
        args.rspec_or_dir) if not is_dir else NumpyReader(args.rspec_or_dir)
    for key, mat in mat_reader:
        if args.apply_log:
            print("hehe")
            mat = np.log10(mat)
        if args.trans:
            mat = np.swapaxes(mat, -1, -2)
        if args.norm:
            mat = mat / np.max(np.abs(mat))
        save_figure(key,
                    mat,
                    cache_dir / key.replace('.', '-'),
                    cmap=args.cmap,
                    hop=args.frame_hop * 1e-3,
                    sr=args.sr,
                    size=args.size)


# now support input from stdin
# shuf mask.scp | head | copy-feats scp:- ark:- | ./scripts/sptk/visualize_tf_matrix.py -
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to visualize kaldi's features/numpy's ndarray on T-F domain. "
        "egs: spectral/spatial features or T-F mask. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("rspec_or_dir",
                        type=str,
                        help="Read specifier of archives "
                        "or directory of ndarrays")
    parser.add_argument("--frame-hop",
                        type=int,
                        default=16,
                        help="Frame shift in ms")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample frequency (Hz)")
    parser.add_argument("--cache-dir",
                        type=str,
                        default="figure",
                        help="Directory to cache pictures")
    parser.add_argument("--apply-log",
                        action=StrToBoolAction,
                        default=False,
                        help="Apply log on input features")
    parser.add_argument("--trans",
                        action=StrToBoolAction,
                        default=False,
                        help="Apply matrix transpose on input features")
    parser.add_argument("--norm",
                        action=StrToBoolAction,
                        default=False,
                        help="Normalize values in [-1, 1] "
                        "before visualization")
    parser.add_argument("--cmap",
                        choices=["binary", "jet", "hot"],
                        default="jet",
                        help="Colormap used when save figures")
    parser.add_argument("--size",
                        type=int,
                        default=3,
                        help="Minimum height of images (in inches)")
    args = parser.parse_args()
    run(args)
