#!/usr/bin/env python

# wujian@2018
"""
Compute directional features using steer vector, based on TF-mask
"""

import argparse

import numpy as np

from libs.utils import nextpow2, get_logger
from libs.opts import StftParser
from libs.spatial import directional_feats
from libs.data_handler import SpectrogramReader, ScriptReader, NumpyReader, ArchiveWriter
from libs.beamformer import MvdrBeamformer

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

    num_bins = nextpow2(args.frame_len) // 2 + 1
    beamformer = MvdrBeamformer(num_bins)

    num_done = 0
    with ArchiveWriter(args.dup_ark, args.scp) as writer:
        for key, spect in feat_reader:
            if key in mask_reader:
                speech_masks = mask_reader[key]
                # make sure speech_masks in T x F
                _, F, _ = spect.shape
                if speech_masks.shape[0] == F:
                    speech_masks = np.transpose(speech_masks)
                speech_masks = np.minimum(speech_masks, 1)
                # spectrogram: N x F x T
                speech_covar = beamformer.compute_covar_mat(
                    speech_masks, spect)
                sv = beamformer.compute_steer_vector(speech_covar)
                df = directional_feats(spect, sv.T, df_pair=None)
                writer.write(key, df)
                num_done += 1
                if not num_done % 1000:
                    logger.info(f"Processed {num_done:d} utterance...")
            else:
                logger.warn(f"Missing TF-mask for utterance {key}")
    logger.info(f"Processed {num_done:d} utterances over {len(feat_reader):d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute directional features for linear arrays, "
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
    args = parser.parse_args()
    run(args)