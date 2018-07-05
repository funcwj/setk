#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import argparse
import glob
import scipy.io as sio

from utils import filekey, get_logger
from data_handler import ArchieveWriter

logger = get_logger(__name__)


def run(args):
    num_mat = 0
    with ArchieveWriter(args.archieve, args.scp) as writer:
        for f in glob.glob("{}/*.mat".format(args.mat_dir)):
            with open(f, "rb") as fd:
                mat_dict = sio.loadmat(fd)
            num_mat += 1
            if args.key not in mat_dict:
                raise KeyError("Could not find \'{}\' in matrix dictionary".format(
                    args.key))
            key = filekey(f)
            writer.write(key, mat_dict[args.key])
    logger.info("Copy {:d} matrix into archieve".format(num_mat))


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
        "archieve", type=str, help="Location to dump float matrix archieve")
    parser.add_argument(
        "--scp",
        type=str,
        default=None,
        help="If assigned, generate corresponding .scp for archieve")
    parser.add_argument(
        "--key",
        type=str,
        default="matrix",
        help="String key to index matrix in MATLAB's .mat file")
    args = parser.parse_args()
    run(args)