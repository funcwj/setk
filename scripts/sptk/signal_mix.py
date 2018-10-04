#!/usr/bin/env python

# wujian@2018

import argparse
import os

import numpy as np 
from libs.utils import get_logger, read_wav, write_wav

logger = get_logger(__name__)


def run(args):
    num_srcs = len(args.src_wav)
    if num_srcs == 1:
        logger.warn("Detect only one source, copy source signal")
        samps = read_wav(args.src_wav[0])
    else:
        # N x S
        src_samps = np.array([read_wav(f) for f in args.src_wav])
        samps = np.sum(src_samps, axis=0) / num_srcs
    write_wav(args.mix_wav, samps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to mix signal from different sources")
    parser.add_argument(
        "src_wav", type=str, nargs="+", help="List of multiple source signal")
    parser.add_argument(
        "mix_wav", type=str, help="Location to dump mixed signal")
    args = parser.parse_args()
    run(args)
