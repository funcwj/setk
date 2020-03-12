#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from libs.data_handler import ScriptReader, ArchiveReader, DirReader
from libs.utils import get_logger, filekey
from libs.opts import StrToBoolAction

default_font = "Times New Roman"
default_dpi = 200
default_fmt = "jpg"

logger = get_logger(__name__)


class NumpyReader(DirReader):
    """
    Numpy matrix reader
    """
    def __init__(self, obj_dir):
        super(NumpyReader, self).__init__(obj_dir, "npy")

    def _load(self, key):
        return np.load(self.index_dict[key])


def save_figure(key,
                mat,
                dest,
                cmap="jet",
                hop=256,
                sr=16000,
                title="",
                size=3):
    """
    Save figure to disk
    """
    def sub_plot(ax, mat, num_frames, num_bins, xticks=True, title=""):
        ax.imshow(np.transpose(mat),
                  origin="lower",
                  cmap=cmap,
                  aspect="auto",
                  interpolation="none")
        if xticks:
            xp = np.linspace(0, num_frames - 1, 5)
            ax.set_xticks(xp)
            ax.set_xticklabels([f"{t:.2f}" for t in (xp * hop / sr)],
                               fontproperties=default_font)
            ax.set_xlabel("Time(s)", fontdict={"family": default_font})
        else:
            ax.set_xticks([])
        yp = np.linspace(0, num_bins - 1, 6)
        fs = np.linspace(0, sr / 2, 6) / 1000
        ax.set_yticks(yp)
        ax.set_yticklabels([f"{t:.1f}" for t in fs],
                           fontproperties=default_font)
        ax.set_ylabel("Frequency(kHz)", fontdict={"family": default_font})
        if title:
            ax.set_title(title, fontdict={"family": default_font})

    logger.info(f"Plot TF-mask of utterance {key} to {dest}.{default_fmt}...")
    if mat.ndim == 3:
        N, T, F = mat.shape
    else:
        T, F = mat.shape
        N = 1
    fig, ax = plt.subplots(nrows=N)
    if N != 1:
        ts = title.split(";")
        for i in range(N):
            if len(ts) == N:
                sub_plot(ax[i], mat[i], T, F, xticks=i == N - 1, title=ts[i])
            else:
                sub_plot(ax[i], mat[i], T, F, xticks=i == N - 1)
    else:
        sub_plot(ax, mat, T, F, title=title)
    fig.savefig(f"{dest}.{default_fmt}", dpi=default_dpi, format=default_fmt)


def run(args):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    reader_templ = {
        "dir": NumpyReader,
        "scp": ScriptReader,
        "ark": ArchiveReader
    }
    # ndarrays or archives
    mat_reader = reader_templ[args.input](args.rspec)
    for key, mat in mat_reader:
        if mat.ndim == 3 and args.index >= 0:
            mat = mat[args.index]
        if args.apply_log:
            mat = np.log10(mat)
        if args.trans:
            mat = np.swapaxes(mat, -1, -2)
        if args.norm:
            mat = mat / np.max(np.abs(mat))
        save_figure(key,
                    mat,
                    cache_dir / key.replace(".", "-"),
                    cmap=args.cmap,
                    hop=args.frame_hop,
                    sr=args.sr,
                    size=args.size,
                    title=args.title)


# now support input from stdin
# shuf mask.scp | head | copy-feats scp:- ark:- | ./scripts/sptk/visualize_tf_matrix.py -
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to visualize kaldi's features/numpy's ndarray on T-F domain. "
        "egs: spectral/spatial features or T-F mask. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("rspec",
                        type=str,
                        help="Read specifier of archives "
                        "or directory of ndarrays")
    parser.add_argument("--input",
                        type=str,
                        choices=["ark", "scp", "dir"],
                        default="dir",
                        help="Type of the input read specifier")
    parser.add_argument("--frame-hop",
                        type=int,
                        default=256,
                        help="Frame shift in samples")
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
                        default=5,
                        help="Minimum height of images (in inches)")
    parser.add_argument("--index",
                        type=int,
                        default=-1,
                        help="Channel index to plot, -1 means all")
    parser.add_argument("--title",
                        type=str,
                        default="",
                        help="Title of the pictures")
    args = parser.parse_args()
    run(args)
