#!/usr/bin/env python

# wujian@2019

import argparse

import numpy as np

from libs.data_handler import SpectrogramReader, NumpyReader
from libs.opts import StftParser, str2tuple
from libs.ssl import ml_ssl, srp_ssl, music_ssl
from libs.utils import get_logger, EPSILON

logger = get_logger(__name__)


def add_wta(masks_list, eps=1e-4):
    """
    Produce winner-take-all masks
    """
    masks = np.stack(masks_list, axis=-1)
    max_mask = np.max(masks, -1)
    wta_masks = []
    for spk_mask in masks_list:
        m = np.where(spk_mask == max_mask, spk_mask, eps)
        wta_masks.append(m)
    return wta_masks


def get_doa(stft, steer_vector, mask, srp_pair, angles, output, backend):
    if srp_pair:
        idx = srp_ssl(stft, steer_vector, srp_pair=srp_pair, mask=mask)
    elif backend == "ml":
        idx = ml_ssl(stft, steer_vector, mask=mask, compression=-1, eps=EPSILON)
    else:
        idx = music_ssl(stft, steer_vector, mask=mask)
    return idx if output == "index" else angles[idx]


def run(args):
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "round_power_of_two": args.round_power_of_two,
        "window": args.window,
        "center": args.center,
        "transpose": True
    }
    steer_vector = np.load(args.steer_vector)
    logger.info(f"Shape of the steer vector: {steer_vector.shape}")
    num_doa, _, _ = steer_vector.shape
    min_doa, max_doa = str2tuple(args.doa_range)
    if args.output == "radian":
        angles = np.linspace(min_doa * np.pi / 180, max_doa * np.pi / 180,
                             num_doa + 1)
    else:
        angles = np.linspace(min_doa, max_doa, num_doa + 1)

    spectrogram_reader = SpectrogramReader(args.wav_scp, **stft_kwargs)
    mask_reader = None
    if args.mask_scp:
        mask_reader = [NumpyReader(scp) for scp in args.mask_scp.split(",")]
    online = (args.chunk_len > 0 and args.look_back > 0)
    if online:
        logger.info("Set up in online mode: chunk_len " +
                    f"= {args.chunk_len}, look_back = {args.look_back}")

    if args.backend == "srp":

        def split_index(sstr):
            return [tuple(map(int, p.split(","))) for p in sstr.split(";")]

        srp_pair = split_index(args.srp_pair)
        srp_pair = ([t[0] for t in srp_pair], [t[1] for t in srp_pair])
        logger.info(f"Choose srp-based algorithm, srp pair is {srp_pair}")
    else:
        srp_pair = None

    with open(args.doa_scp, "w") as doa_out:
        for key, stft in spectrogram_reader:
            # stft: M x T x F
            _, _, F = stft.shape
            if mask_reader:
                # T x F => F x T
                mask = [r[key] for r in mask_reader] if mask_reader else None
                if args.mask_eps >= 0 and len(mask_reader) > 1:
                    mask = add_wta(mask, eps=args.mask_eps)
                mask = mask[0]
                # F x T => T x F
                if mask.shape[-1] != F:
                    mask = mask.transpose()
            else:
                mask = None
            if not online:
                doa = get_doa(stft, steer_vector, mask, srp_pair, angles,
                              args.output, args.backend)
                logger.info(f"Processing utterance {key}: {doa:.4f}")
                doa_out.write(f"{key}\t{doa:.4f}\n")
            else:
                logger.info(f"Processing utterance {key}...")
                _, T, _ = stft.shape
                online_doa = []
                for t in range(0, T, args.chunk_len):
                    s = max(t - args.look_back, 0)
                    if mask is not None:
                        chunk_mask = mask[..., s:t + args.chunk_len]
                    else:
                        chunk_mask = None
                    stft_chunk = stft[:, s:t + args.chunk_len, :]
                    doa = get_doa(stft_chunk, steer_vector, chunk_mask,
                                  srp_pair, angles, args.output, args.backend)
                    online_doa.append(doa)
                doa_str = " ".join([f"{d:.4f}" for d in online_doa])
                doa_out.write(f"{key}\t{doa_str}\n")
    logger.info(f"Processing {len(spectrogram_reader)} utterance done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to ML/SRP based sound souce localization (SSL)."
        "Also see scripts/sptk/compute_steer_vector.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Multi-channel wave rspecifier")
    parser.add_argument("steer_vector",
                        type=str,
                        help="Pre-computed steer vector in each "
                        "directions (in shape A x M x F, A: number "
                        "of DoAs, M: microphone number, F: FFT bins)")
    parser.add_argument("doa_scp",
                        type=str,
                        help="Wspecifier for estimated DoA")
    parser.add_argument("--backend",
                        type=str,
                        default="ml",
                        choices=["ml", "srp", "music"],
                        help="Which algorithm to choose for SSL")
    parser.add_argument("--srp-pair",
                        type=str,
                        default="",
                        help="Microphone index pair to compute srp response")
    parser.add_argument("--doa-range",
                        type=str,
                        default="0,360",
                        help="DoA range")
    parser.add_argument("--mask-scp",
                        type=str,
                        default="",
                        help="Rspecifier for TF-masks in numpy format")
    parser.add_argument("--output",
                        type=str,
                        default="degree",
                        choices=["radian", "degree", "index"],
                        help="Output type of the DoA")
    parser.add_argument("--mask-eps",
                        type=float,
                        default=-1,
                        help="Value of eps used in masking winner-take-all")
    parser.add_argument("--chunk-len",
                        type=int,
                        default=-1,
                        help="Number frames per chunk "
                        "(for online setups)")
    parser.add_argument("--look-back",
                        type=int,
                        default=125,
                        help="Number of frames to look back "
                        "(for online setups)")
    args = parser.parse_args()
    run(args)
