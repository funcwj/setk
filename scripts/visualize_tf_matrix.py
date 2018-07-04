#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import argparse
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from iobase import read_ark


def save_figure(key, mat, dest, frame_shift=10, frequency=16000):
    num_frames, num_bins = mat.shape
    plt.imshow(
        mat.T, origin="lower", cmap="jet", aspect="auto", interpolation="none")
    plt.title(key)
    xp = np.linspace(0, num_frames - 1, 5)
    yp = np.linspace(0, num_bins - 1, 6)
    plt.xticks(xp, ["%.02f" % t for t in (xp * frame_shift)])
    plt.yticks(yp,
               ["%.01f" % t for t in np.linspace(0, frequency / 2, 6) / 1000])
    plt.xlabel('Time(s)')
    plt.ylabel('Frequency(kHz)')
    logging.info('Dump utterance {} to {}'.format(key, dest))
    plt.savefig(dest)


def run(args):
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    with open(args.feature_ark, 'rb') as ark:
        for key, mat in read_ark(ark):
            if args.apply_log:
                mat = np.log10(mat)
            save_figure(
                key,
                mat,
                os.path.join(args.cache_dir, key.replace('.', '-')),
                frame_shift=args.frame_shift * 1e-3,
                frequency=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Command to visualize kaldi\'s features on T-F domain. egs: log-spectrum or T-F mask\n' \
                                    'egs: ./visualize_tf_matrix.py  predict_mask.10.ark --cache-dir predict_mask\n')
    parser.add_argument(
        'feature_ark', type=str, help="path of kaldi\'s feature archives")
    parser.add_argument(
        '--frame-shift',
        dest='frame_shift',
        type=int,
        default=10,
        help="frame shift in ms")
    parser.add_argument(
        '--frequency',
        dest='frequency',
        type=int,
        default=16000,
        help="sample frequency(Hz)")
    parser.add_argument(
        '--cache-dir',
        type=str,
        default="figure",
        dest="cache_dir",
        help="directory to cache pictures")
    parser.add_argument(
        '--apply-log',
        action="store_true",
        dest="apply_log",
        help="apply log on reading features")
    args = parser.parse_args()
    run(args)
