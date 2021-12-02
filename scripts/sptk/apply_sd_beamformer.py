#!/usr/bin/env python
# coding=utf-8
# wujian@2020

import argparse

from libs.utils import get_logger
from libs.opts import StftParser, str2tuple, StrToBoolAction
from apply_classic_beamformer import run as run_classic_beamformer

logger = get_logger(__name__)


def run(args):
    args.beamformer = "sd"
    run_classic_beamformer(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to apply supperdirective beamformer (linear & circular array).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Rspecifier for multi-channel wave file")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Directory to dump enhanced results")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the input wave")
    parser.add_argument("--speed",
                        type=float,
                        default=343,
                        help="Speed of sound")
    parser.add_argument("--geometry",
                        type=str,
                        choices=["linear", "circular"],
                        default="linear",
                        help="Geometry of the microphone array")
    parser.add_argument("--linear-topo",
                        type=str2tuple,
                        default=(),
                        help="Topology of linear microphone arrays")
    parser.add_argument("--circular-around",
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
    parser.add_argument("--utt2doa",
                        type=str,
                        default="",
                        help="Given DoA for each utterances, in degrees")
    parser.add_argument("--doa",
                        type=float,
                        default=0,
                        help="DoA for all utterances if "
                             "--utt2doa is not assigned")
    args = parser.parse_args()
    run(args)
