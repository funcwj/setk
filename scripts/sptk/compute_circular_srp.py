#!/usr/bin/env python

# wujian@2019

import argparse

import numpy as np

from libs.data_handler import SpectrogramReader, ArchiveWriter
from libs.utils import get_logger, nextpow2, EPSILON
from libs.opts import StftParser
from libs.spatial import gcc_phat_diag

logger = get_logger(__name__)


def run(args):
    srp_pair = [
        tuple(map(int, p.split(","))) for p in args.diag_pair.split(";")
    ]
    if not len(srp_pair):
        raise RuntimeError(f"Bad configurations with --pair {args.pair}")
    logger.info(f"Compute gcc with {srp_pair}")

    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "round_power_of_two": args.round_power_of_two,
        "window": args.window,
        "center": args.center,  # false to comparable with kaldi
        "transpose": True  # T x F
    }
    num_done = 0
    num_ffts = nextpow2(
        args.frame_len) if args.round_power_of_two else args.frame_len
    reader = SpectrogramReader(args.wav_scp, **stft_kwargs)
    with ArchiveWriter(args.srp_ark, args.scp) as writer:
        for key, stft_mat in reader:
            num_done += 1
            srp = []
            # N x T x F
            for (i, j) in srp_pair:
                srp.append(
                    gcc_phat_diag(stft_mat[i],
                                  stft_mat[j],
                                  min(i, j) * np.pi * 2 / args.n,
                                  args.d,
                                  num_bins=num_ffts // 2 + 1,
                                  sr=args.sr,
                                  num_doa=args.num_doa))
            srp = np.average(np.stack(srp), axis=0)
            nan = np.sum(np.isnan(srp))
            if nan:
                raise RuntimeError(f"Matrix {key} has nan ({nan:d}) items)")
            writer.write(key, srp)
            if not num_done % 1000:
                logger.info(f"Processed {num_done:d} utterances...")
    logger.info(f"Processd {len(reader):d} utterances done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to compute SRP augular spectrum for circular arrays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Rspecifier for multi-channel wave")
    parser.add_argument("srp_ark", type=str, help="Location to dump features")
    parser.add_argument("--scp",
                        type=str,
                        default="",
                        help="If assigned, generate corresponding scripts")
    parser.add_argument("--n", type=int, default=6, help="Number of arrays")
    parser.add_argument("--d",
                        type=float,
                        default=0.07,
                        help="Diameter of circular array")
    parser.add_argument("--diag-pair",
                        type=str,
                        default="0,3;1,4;2,5",
                        help="Compute gcc between those diagonal arrays")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of input wave")
    parser.add_argument("--num-doa",
                        type=int,
                        default=121,
                        help="Number of DoA to sample between 0 and 2pi")
    args = parser.parse_args()
    run(args)
