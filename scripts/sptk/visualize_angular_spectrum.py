#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import argparse
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from iobase import read_general_mat


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
    plt.colorbar()
    plt.savefig(dest)


def run(args):
    if not os.path.exists(os.path.dirname(args.save_to)):
        os.makedirs(os.path.dirname(args.save_to))
    with open(args.feature_mat, 'rb') as f:
        mat = read_general_mat(f, direct_access=True)
        save_figure(args.title, mat, args.save_to, args.frame_shift * 1e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Command to visualize augular spectrum\n' \
                                    'egs: ./visualize_angular_spectrum.py a.mat --save-to demo\n')
    parser.add_argument(
        'feature_mat',
        type=str,
        help="path of augular spectrum in kaldi\'s matrix format")
    parser.add_argument(
        '--frame-shift',
        dest='frame_shift',
        type=int,
        default=10,
        help="frame shift in ms")
    parser.add_argument(
        '--save-to',
        type=str,
        default="figure.png",
        dest="save_to",
        help="location to save pictures")
    parser.add_argument(
        '--title',
        type=str,
        default="",
        dest="title",
        help="title of figures to plot in")
    args = parser.parse_args()
    run(args)
