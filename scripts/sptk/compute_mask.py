#!/usr/bin/env python

# wujian@2018
"""
Compute IAM(FFT-mask,SMM)/IBM/IRM/PSM masks, using as training targets
"""

import argparse
import numpy as np

from libs.data_handler import SpectrogramReader, ArchiveWriter
from libs.utils import get_logger, cmat_abs
from libs.opts import get_stft_parser

logger = get_logger(__name__)


def compute_mask(speech, noise_or_mixture, mask):
    """
    for signal model:
        y = x1 + x2
    def f = STFT(x):
        f(y) = f(x1) + f(x2) => |f(y)| = |f(x1) + f(x2)| < |f(x1)| + |f(x2)|
    for irm:
        1) M(x1) = |f(x1)| / (|f(x1)| + |f(x2)|)            DongYu
        2) M(x1) = |f(x1)| / sqrt(|f(x1)|^2 + |f(x2)|^2)    DeliangWang
        s.t. 1 >= 2) >= 1) >= 0
    for iam(FFT-mask, smm):
        M(x1) = |f(x1)| / |f(y)| = |f(x1)| / |f(x1) + f(x2)| in [0, oo]
    for psm:
        M(x1) = |f(x1) / f(y)| = |f(x1)| * cos(delta_phase) / |f(y)|
    """
    if mask == "ibm":
        binary_mask = cmat_abs(speech) > cmat_abs(noise_or_mixture)
        return binary_mask.astype(np.float)
    # irm/iam/psm
    if mask == "irm":
        denominator = cmat_abs(speech) + cmat_abs(noise_or_mixture)
        # or denominator = np.sqrt(np.abs(speech)**2 + np.abs(noise_or_mixture)**2)
    else:
        denominator = cmat_abs(noise_or_mixture)
    if mask == "psm":
        return cmat_abs(speech) * np.cos(
            np.angle(noise_or_mixture) - np.angle(speech)) / denominator
    else:
        # irm/iam
        return cmat_abs(speech) / denominator


def run(args):
    # shape: T x F, complex
    stft_kwargs = {
        "frame_length": args.frame_length,
        "frame_shift": args.frame_shift,
        "window": args.window,
        "center": args.center,  # false to comparable with kaldi
    }

    speech_reader = SpectrogramReader(args.speech_scp, **stft_kwargs)
    bnoise_reader = SpectrogramReader(args.noise_scp, **stft_kwargs)

    num_utts = 0
    cutoff = args.cutoff
    with ArchiveWriter(args.mask_ark, args.scp) as writer:
        for key, speech in speech_reader:
            if key in bnoise_reader:
                num_utts += 1
                noise = bnoise_reader[key]
                mask = compute_mask(speech, noise, args.mask)
                if cutoff > 0:
                    num_items = np.sum(mask > cutoff)
                    mask = np.minimum(mask, cutoff)
                    if num_items:
                        logger.info("Clip {:d} items for utterance {}".format(
                            num_items, key))
                    mask = np.maximum(mask, 0)
                writer.write(key, mask)
    logger.info("Processed {} utterances".format(num_utts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute Tf-mask(as targets for Kaldi's nnet3, "
        "only for 2 component case, egs: speech & noise)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[get_stft_parser()])
    parser.add_argument(
        "speech_scp", type=str, help="Target speech scripts in Kaldi format")
    parser.add_argument(
        "noise_scp", type=str, help="Background noise scripts in Kaldi format")
    parser.add_argument(
        "mask_ark", type=str, help="Location to dump mask archives")
    parser.add_argument(
        "--scp",
        type=str,
        default="",
        help="If assigned, generate corresponding mask scripts")
    parser.add_argument(
        "--mask",
        type=str,
        default="irm",
        choices=["irm", "ibm", "iam", "psm"],
        help=
        "Type of masks(irm/ibm/iam(FFT-mask,smm)/psm) to compute. Note that "
        "if iam/psm assigned, second .scp is expected to be noisy component")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=-1,
        help="Cutoff values(<=0, not cutoff) for some non-bounded masks, "
        "egs: iam/psm")
    args = parser.parse_args()
    run(args)