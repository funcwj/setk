#!/usr/bin/env python

# wujian@2018
"""
Compute directional features using steer vector, based on TF-mask
"""

import argparse

import numpy as np

from libs.beamformer import solve_pevd, compute_covar
from libs.data_handler import SpectrogramReader, ScriptReader, NumpyReader, ArchiveWriter
from libs.opts import StftParser
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
    feat_reader = SpectrogramReader(args.wav_scp, **stft_kwargs)
    MaskReader = {"numpy": NumpyReader, "kaldi": ScriptReader}
    mask_reader = MaskReader[args.fmt](args.mask_scp)

    df_pair = [tuple(map(int, p.split(","))) for p in args.df_pair.split(";")]
    if not len(df_pair):
        raise RuntimeError(f"Bad configurations with --pair {args.pair}")
    logger.info(f"Compute directional feature with {df_pair}")

    num_done = 0
    with ArchiveWriter(args.dup_ark, args.scp) as writer:
        for key, obs in feat_reader:
            if key in mask_reader:
                speech_masks = mask_reader[key]
                # make sure speech_masks in T x F
                _, F, _ = obs.shape
                if speech_masks.shape[0] == F:
                    speech_masks = np.transpose(speech_masks)
                speech_masks = np.minimum(speech_masks, 1)
                # obs: N x F x T
                speech_covar = compute_covar(obs, speech_masks)
                sv = solve_pevd(speech_covar)
                df = directional_feats(obs, sv.T, df_pair=df_pair)
                writer.write(key, df)
                num_done += 1
                if not num_done % 1000:
                    logger.info(f"Processed {num_done:d} utterance...")
            else:
                logger.warn(f"Missing TF-mask for utterance {key}")
    logger.info(f"Processed {num_done:d} utterances over {len(feat_reader):d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to compute directional features for arbitrary arrays, "
        "based on estimated TF-masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Multi-Channel wave scripts in kaldi format")
    parser.add_argument("mask_scp",
                        type=str,
                        help="Scripts of masks in kaldi's "
                             "archive or numpy's ndarray")
    parser.add_argument("dup_ark",
                        type=str,
                        help="Location to dump features in kaldi's archives")
    parser.add_argument("--scp",
                        type=str,
                        default="",
                        help="If assigned, generate corresponding "
                             "feature scripts")
    parser.add_argument("--mask-format",
                        dest="fmt",
                        choices=["kaldi", "numpy"],
                        default="kaldi",
                        help="Define format of masks, in kaldi's "
                             "archives or numpy's ndarray")
    parser.add_argument("--df-pair",
                        type=str,
                        default="0,1",
                        help="Microphone pairs for directional "
                             "feature computation")
    args = parser.parse_args()
    run(args)
