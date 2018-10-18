#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import argparse
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from libs.utils import get_logger, filekey

logger = get_logger(__name__)


def save_figure(key, mat, dest, binary=False, shift=10, frequency=16000):
    num_frames, num_bins = mat.shape
    plt.figure()
    plt.imshow(
        mat.T,
        origin="lower",
        cmap="jet" if not binary else "binary",
        aspect="auto",
        interpolation="none")
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
    logger.info('Save utterance {} to {}'.format(key, dest))


def run(args):
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    for f in glob.glob("{}/*.npy".format(args.npy_dir)):
        mat = np.load(f)
        key = filekey(f)
        if args.apply_log:
            mat = np.log10(mat)
        if args.transpose:
            mat = np.transpose(mat)
        save_figure(
            key,
            mat,
            os.path.join(args.cache_dir, key.replace('.', '-')),
            binary=args.binary,
            shift=args.frame_shift * 1e-3,
            frequency=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Command to visualize Numpy\'s .npy matrix on T-F domain. egs: log-spectrum or T-F mask\n"
        "egs: ./visualize_npy_matrix.py  masks_dir --cache-dir figure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'npy_dir', type=str, help="Directory of Numpy\'s 2D-arrays")
    parser.add_argument(
        '--frame-shift',
        dest='frame_shift',
        type=int,
        default=16,
        help="Frame shift in ms")
    parser.add_argument(
        '--frequency',
        dest='frequency',
        type=int,
        default=16000,
        help="Sample frequency(Hz)")
    parser.add_argument(
        '--cache-dir',
        type=str,
        default="figure",
        dest="cache_dir",
        help="Directory to cache pictures")
    parser.add_argument(
        '--apply-log',
        action="store_true",
        dest="apply_log",
        help="Apply log on input features")
    parser.add_argument(
        '--transpose',
        action="store_true",
        dest="transpose",
        help="Apply matrix transpose on input features")
    parser.add_argument(
        '--binary',
        action="store_true",
        dest="binary",
        help="Using binary(black->bigger) colormap instead of jet")
    args = parser.parse_args()
    run(args)
