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


def save_figure(key, mat, dest, hop=16, samp_tdoa=False, size=3):
    num_frames, num_doas = mat.shape
    plt.figure(figsize=(max(size * num_frames / num_doas, size + 2), size + 2))
    # binary: black -> higher
    plt.imshow(mat.T,
               origin="lower",
               cmap="binary",
               aspect="auto",
               interpolation="none")
    plt.title(key)
    # plt.colorbar()
    xp = np.linspace(0, num_frames - 1, 5)
    yp = np.linspace(0, num_doas - 1, 7)
    plt.xticks(xp, ["{:.02f}".format(t) for t in (xp * hop)])
    plt.yticks(yp, ["%d" % d for d in yp])
    plt.xlabel("Time(s)")
    plt.ylabel("DoA" if not samp_tdoa else "TDoA Index")
    plt.savefig(dest)
    plt.close()
    logger.info('Save utterance {} to {}.png'.format(key, dest))


def run(args):
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    ark_reader = ArchiveReader(args.srp_ark)
    for key, mat in ark_reader:
        dst = os.path.join(args.cache_dir, key.replace('.', '-'))
        save_figure(key,
                    mat,
                    dst,
                    hop=args.frame_hop * 1e-3,
                    samp_tdoa=args.tdoa,
                    size=args.size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Command to visualize augular spectrum.\n"
        "egs: ./visualize_angular_spectrum.py a.ark --cache-dir demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "srp_ark",
        type=str,
        help="Path of augular spectrum in kaldi\'s archive format")
    parser.add_argument("--frame-hop",
                        type=int,
                        default=16,
                        help="Frame shift in ms")
    parser.add_argument("--cache-dir",
                        type=str,
                        default="figure",
                        help="Location to dump pictures")
    parser.add_argument("--sample-tdoa",
                        dest="tdoa",
                        action="store_true",
                        help="Sample TDoA instead of DoA when "
                        "computing spectrum")
    parser.add_argument("--size",
                        type=int,
                        default=3,
                        help="Minimum height of images (in inches)")
    args = parser.parse_args()
    run(args)
