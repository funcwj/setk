#!/usr/bin/env python

# wujian@2020
"""
Compute steer vector (based on array geometry) for linear/circular arrays
"""

import argparse

import numpy as np

from libs.beamformer import linear_steer_vector, circular_steer_vector
from libs.opts import StrToBoolAction, str2tuple


def run(args):
    if args.type == "linear":
        topo = np.array(str2tuple(args.linear_topo))
        candidate_doa = np.linspace(0, 180, args.num_doas) * np.pi / 180
    else:
        topo = None
        step = 360 / args.num_doas
        candidate_doa = np.arange(0, 360, step) * np.pi / 180

    sv = []
    for doa in candidate_doa:
        if topo is None:
            sv.append(
                circular_steer_vector(args.circular_radius,
                                      args.circular_number,
                                      doa,
                                      args.num_bins,
                                      c=args.speed,
                                      sr=args.sr,
                                      center=args.circular_center))
        else:
            sv.append(
                linear_steer_vector(topo,
                                    doa,
                                    args.num_bins,
                                    c=args.speed,
                                    sr=args.sr))
    # A x F x M
    sv = np.stack(sv)
    # norm or not
    if args.normalize:
        sv = sv / sv.shape[-1]**0.5
    # A x M x F
    sv = sv.transpose(0, 2, 1)
    np.save(args.steer_vector, sv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to compute steer vectors, using for SSL & BF & AF computation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("steer_vector",
                        type=str,
                        help="Output location of the steer vector")
    parser.add_argument("--num-doas",
                        type=int,
                        default=181,
                        help="Step size when sampling the DoA")
    parser.add_argument("--num-bins",
                        type=int,
                        default=257,
                        help="Number of the FFT points used")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of input wave")
    parser.add_argument("--speed",
                        type=float,
                        default=340,
                        help="Speed of sound")
    parser.add_argument("--linear-topo",
                        type=str,
                        default="",
                        help="Topology of linear microphone arrays")
    parser.add_argument("--circular-number",
                        type=int,
                        default=6,
                        help="Number of the micriphones in circular arrays")
    parser.add_argument("--circular-radius",
                        type=float,
                        default=0.05,
                        help="Radius of circular array")
    parser.add_argument("--circular-center",
                        action=StrToBoolAction,
                        default=False,
                        help="Is there a microphone put in the "
                        "center of the circular array?")
    parser.add_argument("--type",
                        type=str,
                        choices=["linear", "circular"],
                        default="linear",
                        help="Geometry of the microphone array")
    parser.add_argument("--normalize",
                        action=StrToBoolAction,
                        default=False,
                        help="Normalzed steer vector or not")
    args = parser.parse_args()
    run(args)