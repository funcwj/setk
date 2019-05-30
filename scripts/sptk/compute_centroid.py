#!/usr/bin/env python
# wujian@2018

import argparse

import numpy as np

from libs.data_handler import NumpyReader, NumpyWriter, parse_scps
from libs.utils import get_logger

logger = get_logger(__name__)


def run(args):
    numpy_reader = NumpyReader(args.npy_scp)

    spk2utt = parse_scps(args.spk2utt, num_tokens=-1) if args.spk2utt else None

    with NumpyWriter(args.dump_dir, args.scp) as writer:
        if spk2utt is None:
            for key, mat in numpy_reader:
                if mat.ndim != 2:
                    raise RuntimeError(
                        "--spk2utt is None, so input ndarray must be 2D, got {:d}"
                        .format(mat.ndim))
                if args.normalize:
                    mat = mat / np.linalg.norm(
                        mat, ord=2, axis=1, keepdims=True)
                writer.write(key, np.mean(mat, axis=0))
            logger.info("Processed {:d} speakers".format(len(numpy_reader)))
        else:
            for spkid, uttlist in spk2utt.items():
                spkset = []
                for uttid in uttlist:
                    vec = numpy_reader[uttid]
                    if vec.ndim != 1:
                        raise RuntimeError(
                            "--spk2utt is not None, expect input as vector, got {:d}"
                            .format(vec.ndim))
                    if args.normalize:
                        vec = vec / np.linalg.norm(vec)
                    spkset.append(vec)
                spk_mat = np.stack(spkset)
                writer.write(spkid, np.mean(spk_mat, axis=0))
            logger.info("Processed {:d} speakers".format(len(spk2utt)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute means of numpy vectors/matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("npy_scp", type=str, help="Input numpy rspecifier")
    parser.add_argument("--dump-dir",
                        type=str,
                        default="mean",
                        help="Directory to dump computed results")
    parser.add_argument("--spk2utt",
                        type=str,
                        default="",
                        help="Rspecifier for speaker to utterance-list map")
    parser.add_argument("--scp",
                        type=str,
                        default="",
                        help="If assigned, generate corresponding scripts")
    parser.add_argument("--normalize",
                        action="store_true",
                        help="If true, normalize vectors before compute means")
    args = parser.parse_args()
    run(args)