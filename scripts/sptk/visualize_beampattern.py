#!/usr/bin/env python

# wujian@2019

import argparse

import numpy as np
import matplotlib.pyplot as plt

from libs.beamformer import beam_pattern


def run(args):
    # (B) x F x M
    weight = np.load(args.weight)
    multi_beam = weight.ndim == 3
    # A x M x F
    steer_vector = np.load(args.steer_vector)
    A, _, F = steer_vector.shape
    steer_vector = np.einsum("amf->fam", steer_vector)

    if multi_beam:
        if args.beam >= weight.shape[0]:
            raise RuntimeError("Beam index out of range: " +
                               f"{args.beam} vs {weight.shape[0]}")
        pattern = beam_pattern(weight[args.beam], steer_vector)
    else:
        pattern = beam_pattern(weight, steer_vector)
    xp = np.linspace(0, F - 1, 6)
    xt = np.linspace(0, args.sr // 2, 6) / 1000
    yp = np.linspace(0, A, 5)
    plt.imshow(pattern.T, cmap="jet", origin="lower")
    plt.xticks(xp, [f"{t:.1f}" for t in xt])
    plt.yticks(yp, [f"{int(t)}" for t in (yp * args.doa_range / A)])
    plt.ylabel("DoA (degree)")
    plt.xlabel("Frequency (kHz)")
    if multi_beam:
        plt.title(f"BeamPattern of Beam-{args.beam+1}")
    else:
        plt.title(f"BeamPattern")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to plot beam pattern of the fixed beamformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("weight",
                        type=str,
                        help="Weight of the fixed beamformer "
                        "(in shape B x F x N or F x N)")
    parser.add_argument("steer_vector",
                        type=str,
                        help="Pre-computed steer vector in each "
                        "directions (in shape A x M x F, A: number "
                        "of DoAs, M: microphone number, F: FFT bins)")
    parser.add_argument("--beam",
                        type=int,
                        default=0,
                        help="Beam index to plot "
                        "(if contains multi-beam)")
    parser.add_argument("--doa-range",
                        type=int,
                        default=360,
                        help="Maximum DoA value")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the data")
    args = parser.parse_args()
    run(args)