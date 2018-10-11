#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import argparse
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from libs.data_handler import ArchiveReader
from libs.utils import get_logger

logger = get_logger(__name__)



def save_figure(key, mat, dest, frame_shift):
    num_frames, num_doas = mat.shape
    plt.imshow(
        mat.T,
        origin="lower",
        cmap="binary",
        aspect="auto",
        interpolation="none")
    plt.title(key)
    xp = np.linspace(0, num_frames - 1, 5)
    yp = np.linspace(0, num_doas, 7)
    plt.xticks(xp, ["{:.02f}".format(t) for t in (xp * frame_shift)])
    plt.yticks(yp, ["%d" % d for d in yp])
    plt.xlabel('Time(s)')
    plt.ylabel('DoA')
    plt.savefig(dest)


def run(args):
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    ark_reader = ArchiveReader(args.srp_ark)
    for key, mat in ark_reader:
        dst = os.path.join(args.cache_dir, key.replace('.', '-'))
        save_figure(key, mat, dst, args.frame_shift * 1e-3)
        logger.info('Save utterance {} to {}.png'.format(key, dst))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Command to visualize augular spectrum\n' \
                                    'egs: ./visualize_angular_spectrum.py a.ark --cache-dir demo\n')
    parser.add_argument(
        'srp_ark',
        type=str,
        help="Path of augular spectrum in kaldi\'s archive format")
    parser.add_argument(
        '--frame-shift',
        dest='frame_shift',
        type=int,
        default=16,
        help="Frame shift in ms")
    parser.add_argument(
        '--cache-dir',
        type=str,
        default="figure.png",
        dest="cache_dir",
        help="Location to dump pictures")
    args = parser.parse_args()
    run(args)
