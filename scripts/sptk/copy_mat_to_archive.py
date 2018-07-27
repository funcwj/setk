#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import argparse
import glob
import numpy as np 
import scipy.io as sio

from utils import filekey, get_logger
from data_handler import ArchiveWriter

logger = get_logger(__name__)


def run(args):
    num_mat = 0
    with ArchiveWriter(args.archive, args.scp) as writer:
        for f in glob.glob("{}/*.mat".format(args.mat_dir)):
            with open(f, "rb") as fd:
                mat_dict = sio.loadmat(fd)
            num_mat += 1
            if args.key not in mat_dict:
                raise KeyError("Could not find \'{}\' in matrix dictionary".format(
                    args.key))
            mat = mat_dict[args.key]
            key = filekey(f)
            if args.transpose:
                mat = np.transpose(mat)
            if args.minus_by_one:
                mat = 1 - mat
            writer.write(key, mat)
    logger.info("Copy {:d} matrix into archive {}".format(num_mat, args.archive))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to copy a set of MATLAB's .mat (real)matrix to kaldi's .scp & .ark files"
    )
    parser.add_argument(
        "mat_dir",
        type=str,
        help="Source directory which contains a list of .mat files")
    parser.add_argument(
        "archive", type=str, help="Location to dump float matrix archive")
    parser.add_argument(
        "--scp",
        type=str,
        default=None,
        help="If assigned, generate corresponding .scp for archive")
    parser.add_argument(
        "--key",
        type=str,
        default="matrix",
        help="String key to index matrix in MATLAB's .mat file")
    parser.add_argument(
        "--transpose",
        action="store_true",
        default=False,
        help="If true, transpose matrix before write to archives")
    parser.add_argument(
        "--minus-by-one",
        action="store_true",
        default=False,
        dest="minus_by_one",
        help="If true,  write (1 - matrix) to archives")
    args = parser.parse_args()
    run(args)