#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import os
import argparse
import glob
import numpy as np
import scipy.io as sio

from libs.utils import filekey, get_logger, EPSILON
from libs.data_handler import ScriptReader, NumpyWriter

logger = get_logger(__name__)


def run(args):
    src_reader = ScriptReader(args.src_scp, matrix=(not args.vector))
    num_mat = 0
    with NumpyWriter(args.dst_dir, args.scp) as writer:
        for key, mat in src_reader:
            if args.trans:
                mat = np.transpose(mat)
            writer.write(key, mat)
            num_mat += 1
    logger.info("Copy {0} matrix/vector into directory {1}".format(
        num_mat, args.dst_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to copy Kaldi's archives to Numpy's ndarrays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "src_scp", type=str, help="Kaldi's matrix/vector script, to be copyed")
    parser.add_argument(
        "dst_dir", type=str, help="Location to dump numpy's ndarray")
    parser.add_argument(
        "--transpose",
        action="store_true",
        dest="trans",
        help="If true, transpose matrix before write to ndarray")
    parser.add_argument(
        "--input-vector",
        action="store_true",
        dest="vector",
        help="If true, input is vector instead of matrix")
    parser.add_argument(
        "--scp",
        type=str,
        default="",
        help="If assigned, dump corresponding script")
    args = parser.parse_args()
    run(args)