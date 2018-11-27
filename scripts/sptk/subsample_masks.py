#!/usr/bin/env python

# wujian@2018

import argparse

import numpy as np

from libs.data_handler import ScriptReader, ArchiveWriter
from libs.utils import get_logger

logger = get_logger(__name__)


def subsample_mask(matrix):
    F, T = matrix.shape
    if (F - 1) % 2 != 0:
        raise RuntimeError(
            "Seems not a mask matrix, shape in {:d} x {:d}".format(T, F))
    subsample_len = (F - 1) // 2 + 1
    subsampled = np.zeros((subsample_len, T))
    subsampled[0] = matrix[0]
    # 1 <- 1+2, 2 <- 3+4 ...
    for f in range(1, subsample_len):
        subsampled[f] = (matrix[2 * f - 1] + matrix[2 * f]) / 2
    return np.transpose(subsampled)


def run(args):
    mask_reader = ScriptReader(args.src_scp)
    with ArchiveWriter(args.dst_ark, args.scp) as writer:
        for key, mask in mask_reader:
            if not args.trans:
                mask = np.transpose(mask)
            # mask: shape as FxT
            subsampled = subsample_mask(mask)
            writer.write(key, subsampled)
    logger.info("Subsampled {:d} mask matrix".format(len(mask_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to subsample masks from 32ms to 16ms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src_scp", type=str, help="Input mask script")
    parser.add_argument("dst_ark", type=str, help="Subsampled mask archive")
    parser.add_argument(
        "--scp",
        type=str,
        default="",
        help="If assigned, dump corresponding script")
    parser.add_argument(
        "--trans",
        action="store_true",
        help="Transpose mask or not, possible in shape FxT")
    args = parser.parse_args()
    run(args)
