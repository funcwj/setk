#!/usr/bin/env python
# coding=utf-8
# wujian@2020

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from libs.data_handler import SpectrogramReader
from libs.opts import StftParser
from libs.utils import get_logger

default_font = "Times New Roman"
default_font_size = 10
default_dpi = 200
default_fmt = "jpg"

logger = get_logger(__name__)


def save_figure(key, mat, dest, cmap="jet", hop=256, sr=16000, title=""):
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
            ax.set_xlabel("Time (s)", fontdict={"family": default_font})
        else:
            ax.set_xticks([])
        yp = np.linspace(0, num_bins - 1, 6)
        fs = np.linspace(0, sr / 2, 6) / 1000
        ax.set_yticks(yp)
        ax.set_yticklabels([f"{t:.1f}" for t in fs],
                           fontproperties=default_font)
        ax.set_ylabel("Frequency (kHz)", fontdict={"family": default_font})
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
    plt.close(fig)


def run(args):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "round_power_of_two": args.round_power_of_two,
        "window": args.window,
        "center":
            args.center  # false to comparable with kaldi
    }
    reader = SpectrogramReader(args.wav_scp,
                               **stft_kwargs,
                               apply_abs=True,
                               apply_log=True,
                               transpose=True)

    for key, mat in reader:
        if mat.ndim == 3 and args.index >= 0:
            mat = mat[args.index]
        save_figure(key,
                    mat,
                    cache_dir / key.replace(".", "-"),
                    cmap=args.cmap,
                    hop=args.frame_hop,
                    sr=args.sr,
                    title=args.title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to visualize audio spectrogram.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp", type=str, help="Read specifier of audio")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample frequency (Hz)")
    parser.add_argument("--cache-dir",
                        type=str,
                        default="spectrogram",
                        help="Directory to dump spectrograms")
    parser.add_argument("--cmap",
                        choices=["binary", "jet", "hot"],
                        default="jet",
                        help="Colormap used when save figures")
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
