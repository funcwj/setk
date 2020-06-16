#!/usr/bin/env python

# wujian@2020

import json
import random
import argparse

import numpy as np

from .libs.data_handler import WaveReader


def run(args):

    random.seed(args.seed)
    np.random.seed(args.seed)

    src_sampler = WaveReader(args.src_spk, sample_rate=args.sr)
    with open(args.simu_cfg, "w") as simu_cfg:

        for i in range(args.num_utts):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do create data simulation configurations, "
        "for wav_simulate.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    parser.add_argument("src_spk", type=str, help="Source speaker scripts")
    parser.add_argument("rir_cfg",
                        type=str,
                        help="RIR configurations (in json format). "
                        "See rir_generate_{1,2}d")
    parser.add_argument("simu_cfg",
                        type=str,
                        help="Output configuration of the data simulation")
    parser.add_argument("--num-utts",
                        type=int,
                        default=1000,
                        help="Number of the simulated utterances")
    parser.add_argument("--simu-dir",
                        type=str,
                        default="simu",
                        help="Simulation directory to dump data")
    parser.add_argument("--point-noise",
                        type=str,
                        required=True,
                        help="Add pointsource noises")
    parser.add_argument("--point-noise-snr",
                        type=str,
                        default="5,15",
                        help="SNR of the pointsource noises")
    parser.add_argument("--isotropic-noise",
                        type=str,
                        default="",
                        help="Add isotropic noises")
    parser.add_argument("--isotropic-noise-snr",
                        type=str,
                        default="10,20",
                        help="SNR of the isotropic noises")
    parser.add_argument("--dump-channel",
                        type=int,
                        default=-1,
                        help="Index of the channel to dump out (-1 means all)")
    parser.add_argument('--norm-factor',
                        type=float,
                        default="0.5,0.9",
                        help="Normalization factor of the final output")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Value of the sample rate")
    parser.add_argument("--seed",
                        type=int,
                        default=666,
                        help="Random seed to set")
    run(args)