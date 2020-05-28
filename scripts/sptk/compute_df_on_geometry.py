#!/usr/bin/env python

# wujian@2020
"""
Compute directional/angle feature using steer vector (based on array geometry)
"""

import argparse

import numpy as np 

from libs.opts import StftParser
from libs.data_handler import SpectrogramReader, ArchiveWriter, ScpReader
from libs.spatial import directional_feats
from libs.utils import get_logger

logger = get_logger(__name__)


def run(args):
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "round_power_of_two": args.round_power_of_two,
        "window": args.window,
        "center": args.center,  # false to comparable with kaldi
        "transpose": False  # F x T
    }
    stft_reader = SpectrogramReader(args.wav_scp, **stft_kwargs)
    if args.utt2idx:
        utt2idx = ScpReader(args.utt2idx, value_processor=int)
        logger.info(f"Using --utt2idx={args.utt2idx}")
    else:
        utt2idx = None
        logger.info(f"Using --doa-idx={args.doa_idx}")

    df_pair = [
        tuple(map(int, p.split(","))) for p in args.diag_pair.split(";")
    ]
    if not len(df_pair):
        raise RuntimeError(f"Bad configurations with --pair {args.pair}")
    logger.info(f"Compute directional feature with {df_pair}")

    # A x M x F
    steer_vector = np.load(args.steer_vector)

    num_done = 0
    with ArchiveWriter(args.dup_ark, args.scp) as writer:
        for key, stft in stft_reader:
            # sv: M x F
            if utt2idx is None:
                sv = steer_vector[args.doa_idx]
            elif key in utt2idx:
                sv = steer_vector[utt2idx[key]]
            else:
                logger.warn(f"Missing utt2idx for utterance {key}")
                continue
            # stft: M x F x T
            df = directional_feats(stft, sv, df_pair=df_pair)
            writer.write(key, df)
            num_done += 1
            if not num_done % 1000:
                logger.info(f"Processed {num_done:d} utterance...")
    logger.info(f"Processed {num_done:d} utterances over {len(stft_reader):d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute directional features for linear arrays, "
        "based on given steer vector. Also see scripts/sptk/compute_steer_vector.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Multi-Channel wave scripts in kaldi format")
    parser.add_argument("steer_vector",
                        type=str,
                        help="Pre-computed steer vector in each "
                        "directions (in shape A x M x F, A: number "
                        "of DoAs, M: microphone number, F: FFT bins)")
    parser.add_argument("dup_ark",
                        type=str,
                        help="Location to dump features (in ark format)")
    parser.add_argument("--utt2idx",
                        type=str,
                        default="",
                        help="utt2idx for index (between "
                        "[0, A - 1]) of the DoA.")
    parser.add_argument("--doa-idx",
                        type=int,
                        default=0,
                        help="DoA index for all utterances if --utt2idx=\"\"")
    parser.add_argument("--scp",
                        type=str,
                        default="",
                        help="If assigned, generate corresponding "
                        "feature scripts")
    parser.add_argument("--df-pair",
                        type=str,
                        default="0,1",
                        help="Microphone pairs for directional "
                        "feature computation")
    args = parser.parse_args()
    run(args)