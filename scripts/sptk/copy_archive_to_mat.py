#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import os
import argparse
import glob
import numpy as np
import scipy.io as sio

from libs.utils import filekey, get_logger, EPSILON
from libs.data_handler import ScriptReader, ArchiveReader, NumpyWriter, MatWriter

logger = get_logger(__name__)


def run(args):
    src_format = args.input == "matrix"
    src_reader = ScriptReader(
        args.src_dec,
        matrix=src_format) if args.src == "scp" else ArchiveReader(
            args.src_dec, matrix=src_format)
    num_done = 0
    WriterImpl = {"npy": NumpyWriter, "mat": MatWriter}[args.dst]
    with WriterImpl(args.dst_dir, args.scp) as writer:
        for key, mat in src_reader:
            if args.trans:
                mat = np.transpose(mat)
            writer.write(key, mat)
            num_done += 1
    logger.info("Copy {0} {1} into directory {2}".format(
        num_done, "matrices" if src_format else "vectors", args.dst_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to copy Kaldi's archives to Numpy's ndarrays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src_dec",
                        type=str,
                        help="Rspecifier for input features(.ark/.scp)")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Location to dump numpy's ndarray")
    parser.add_argument("--src-format",
                        type=str,
                        dest="src",
                        choices=["ark", "scp"],
                        default="scp",
                        help="Format of input rspecifier")
    parser.add_argument("--dst-format",
                        type=str,
                        dest="dst",
                        choices=["npy", "mat"],
                        default="npy",
                        help="Format of the data to transform to")
    parser.add_argument("--transpose",
                        action="store_true",
                        dest="trans",
                        help="If true, transpose matrix "
                        "before write to ndarray")
    parser.add_argument("--input",
                        type=str,
                        choices=["matrix", "vector"],
                        default="matrix",
                        help="Type of the data in source rspecifier")
    parser.add_argument("--scp",
                        type=str,
                        default="",
                        help="If assigned, dump corresponding scripts")
    args = parser.parse_args()
    run(args)