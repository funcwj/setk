#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import argparse
import os

import numpy as np
from tqdm import tqdm
from libs.data_handler import SpectrogramReader
from libs.utils import istft, get_logger, cmat_abs
from libs.opts import get_stft_parser

logger = get_logger(__name__)


def compute_mask(mixture, targets_list, mask_type):
    """
    Arguments:
        mixture: STFT of mixture signal(complex result) 
        targets_list: python list of target signal's STFT results(complex result)
        mask_type: ["irm", "ibm", "iam", "psm"]
    Return:
        masks_list
    """
    if mask_type == "ibm":
        max_index = np.argmax(
            np.stack([cmat_abs(mat) for mat in targets_list]), 0)
        return [max_index == s for s in range(len(targets_list))]

    if mask_type == "irm":
        denominator = sum([cmat_abs(mat) for mat in targets_list])
    else:
        denominator = cmat_abs(mixture)
    if mask_type != "psm":
        masks = [cmat_abs(mat) / denominator for mat in targets_list]
    else:
        mixture_phase = np.angle(mixture)
        masks = [
            cmat_abs(mat) * np.cos(mixture_phase - np.angle(mat)) / denominator
            for mat in targets_list
        ]
    return masks


def run(args):
    # return complex result
    stft_kwargs = {
        "frame_length": args.frame_length,
        "frame_shift": args.frame_shift,
        "window": args.window,
        "center": args.center
    }
    logger.info("Using mask: {}".format(args.mask.upper()))
    mixture_reader = SpectrogramReader(args.mix_scp, **stft_kwargs)
    ref_scp_list = args.ref_scp.split(",")
    logger.info("Number of speakers: {:d}".format(len(ref_scp_list)))
    targets_reader = [
        SpectrogramReader(scp, **stft_kwargs) for scp in ref_scp_list
    ]
    num_utts = 0
    for key, mixture in tqdm(mixture_reader):
        nsamps = mixture_reader.nsamps(key) if args.keep_length else None
        skip = False
        for reader in targets_reader:
            if key not in reader:
                logger.info("Skip utterance {}, missing targets".format(key))
                skip = True
                break
        if skip:
            continue
        num_utts += 1
        targets_list = [reader[key] for reader in targets_reader]
        spk_masks = compute_mask(mixture, targets_list, args.mask)
        for index, mask in enumerate(spk_masks):
            istft(
                os.path.join(args.dump_dir, '{}.s{}.wav'.format(
                    key, index + 1)),
                mixture * mask,
                **stft_kwargs,
                fs=args.fs,
                nsamps=nsamps)
    logger.info("Processed {} utterance!".format(num_utts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do oracle speech separation, "
        "using specified mask(IAM|IBM|IRM|PSM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[get_stft_parser()])
    parser.add_argument(
        "mix_scp",
        type=str,
        help="Location of mixture wave scripts in kaldi format")
    parser.add_argument(
        "--ref-scp",
        type=str,
        required=True,
        help="Reference speaker wave scripts in kaldi format, "
        "separated using \',\'")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="cache",
        help="Location to dump seperated speakers")
    parser.add_argument(
        "--mask",
        type=str,
        default="irm",
        choices=["iam", "irm", "ibm", "psm"],
        help="Type of mask to use for speech separation")
    parser.add_argument(
        "--sample-frequency",
        type=int,
        default=16000,
        dest="fs",
        help="Waveform data sample frequency")
    parser.add_argument(
        "--keep-length",
        action="store_true",
        help="If ture, keep result the same length as orginal")
    args = parser.parse_args()
    run(args)