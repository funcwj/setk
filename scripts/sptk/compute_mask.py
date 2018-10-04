#!/usr/bin/env python

# wujian@2018

import argparse
import numpy as np 

from libs.data_handler import SpectrogramReader, ArchiveWriter
from libs.utils import get_logger

logger = get_logger(__name__)

def compute_mask(speech, noise, mtype):
    if mtype == "ibm":
        binary_mask = np.abs(speech) > np.abs(noise)
        return binary_mask.astype(np.float)
    else:
        denominator = np.abs(speech) + np.abs(noise)
        return np.abs(speech) / denominator

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
    with ArchiveWriter(args.mask_ark, args.scp) as writer:
        for key, speech in speech_reader:
            if key in bnoise_reader:
                num_utts += 1
                noise = bnoise_reader[key]
                mask = compute_mask(speech, noise, args.mask)
                writer.write(key, mask)
    logger.info("Processed {} utterances".format(num_utts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to compute Tf-mask(as targets for Kaldi's nnet3, only for 2 component case, egs: speech & noise)"
    )
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
        choices=['irm', 'ibm'],
        help="Type of masks(irm/ibm) to compute")
    parser.add_argument(
        "--frame-length",
        type=int,
        default=1024,
        dest="frame_length",
        help="Frame length in number of samples")
    parser.add_argument(
        "--frame-shift",
        type=int,
        default=256,
        dest="frame_shift",
        help="Frame shift in number of samples")
    parser.add_argument(
        "--center",
        action="store_true",
        default=False,
        dest="center",
        help="Parameter \'center\' in librosa.stft functions")
    parser.add_argument(
        "--window",
        default="hann",
        dest="window",
        help="Type of window function, see scipy.signal.get_window")

    args = parser.parse_args()
    run(args)