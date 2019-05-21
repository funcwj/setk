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

supported_op = ["trans", "log", "minus", "stack"]


def run(args):
    src_reader = NumpyReader(args.src_scp) if args.src == "npy" else MatReader(
        args.src_scp, args.key)
    num_mat = 0
    mat_list = []
    mat = (args.output == "matrix")
    ops = args.op.split(",")
    for op in ops:
        if op and op not in supported_op:
            raise RuntimeError("Unknown operation: {}".format(op))
    stack = "stack" in ops
    with ArchiveWriter(args.dst_ark, args.scp, matrix=mat) as writer:
        for key, mat in src_reader:
            for op in ops:
                if op == "trans":
                    mat = np.transpose(mat)
                elif op == "log":
                    mat = np.log(np.maximum(mat, EPSILON))
                elif op == "minus":
                    mat = 1 - mat
                else:
                    pass
            if stack:
                mat_list.append(mat)
            else:
                writer.write(key, mat)
            num_mat += 1
        if stack:
            mat = np.vstack(mat_list)
            writer.write(filekey(args.dst_ark), mat)
            logger.info("Merge {0} matrix into archive {1}, shape as "
                        "{2[0]}x{2[1]}".format(num_mat, args.dst_ark,
                                               mat.shape))
    if not stack:
        logger.info("Copy {0} matrix into archive {1}".format(
            num_mat, args.dst_ark))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to copy MATLAB's .mat or Python's .npy (real)matrix "
        "to kaldi's archives",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src_scp",
                        type=str,
                        help="Source scripts for .mat/.npy files")
    parser.add_argument("dst_ark",
                        type=str,
                        help="Location to dump kaldi's archives")
    parser.add_argument("--scp",
                        type=str,
                        default=None,
                        help="If assigned, generate corresponding "
                        ".scp for archives")
    parser.add_argument("--mat-index",
                        type=str,
                        dest="key",
                        default="data",
                        help="A string to index data in MATLAB's .mat file")
    parser.add_argument("--op",
                        type=str,
                        default="",
                        help="Operations to applied on source "
                        "matrix/vector, separated by \",\", now support "
                        "trans/log/minus/stack")
    parser.add_argument("--src-format",
                        type=str,
                        dest="src",
                        choices=["npy", "mat"],
                        default="npy",
                        help="Data format in the input rspecifier")
    parser.add_argument("--output",
                        type=str,
                        choices=["matrix", "vector"],
                        default="matrix",
                        help="Type of the data to dump in archives")
    args = parser.parse_args()
    run(args)