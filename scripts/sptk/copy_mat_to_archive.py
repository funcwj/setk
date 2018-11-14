#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import argparse
import glob
import numpy as np
import scipy.io as sio

from libs.utils import filekey, get_logger, EPSILON
from libs.data_handler import ArchiveWriter, NumpyReader, MatReader

logger = get_logger(__name__)


def run(args):
    src_reader = NumpyReader(args.src_scp) if not args.matlab else MatReader(
        args.src_scp, args.key)
    num_mat = 0
    mat_list = []
    with ArchiveWriter(args.dst_ark, args.scp) as writer:
        for key, mat in src_reader:
            if args.transpose:
                mat = np.transpose(mat)
            if args.apply_log:
                mat = np.log(np.maximum(mat, EPSILON))
            if args.minus_by_one:
                mat = 1 - mat
            if not args.merge:
                writer.write(key, mat)
            else:
                mat_list.append(mat)
            num_mat += 1
        if args.merge:
            mat = np.vstack(mat_list)
            writer.write(filekey(args.dst_ark), mat)
            logger.info(
                "Merge {0} matrix into archive {1}, shape as {2[0]}x{2[1]}".
                format(num_mat, args.dst_ark, mat.shape))
    if not args.merge:
        logger.info("Copy {0} matrix into archive {1}".format(
            num_mat, args.dst_ark))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to copy MATLAB's .mat or Python's .npy (real)matrix "
        "to kaldi's archives",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "src_scp", type=str, help="Source scripts for .mat/.npy files")
    parser.add_argument(
        "dst_ark", type=str, help="Location to dump kaldi's archives")
    parser.add_argument(
        "--scp",
        type=str,
        default=None,
        help="If assigned, generate corresponding .scp for archives")
    parser.add_argument(
        "--matlab.key",
        type=str,
        dest="key",
        default="matrix",
        help="String key to index matrix in MATLAB's .mat file")
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="If true, transpose matrix before write to archives")
    parser.add_argument(
        "--apply-log",
        action="store_true",
        help="If true, apply log before write to archives")
    parser.add_argument(
        "--matlab",
        action="store_true",
        help="If true, treat src_scp as object of .mat files")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="If true, write src_scp in one matrix in final archives")
    parser.add_argument(
        "--minus-by-one",
        action="store_true",
        help="If true,  write (1 - matrix) to archives")
    args = parser.parse_args()
    run(args)