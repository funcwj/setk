#!/usr/bin/env python

# wujian@2018

import argparse
import os

import numpy as np

from libs.utils import stft, istft, get_logger
from libs.opts import StftParser
from libs.data_handler import SpectrogramReader, NumpyReader, ScriptReader, WaveWriter

logger = get_logger(__name__)


def run(args):
    # shape: T x F, complex
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "window": args.window,
        "center": args.center,
    }
    spectrogram_reader = SpectrogramReader(
        args.wav_scp,
        **stft_kwargs,
        round_power_of_two=args.round_power_of_two)
    phase_reader = None
    if args.phase_ref:
        phase_reader = SpectrogramReader(
            args.phase_ref,
            **stft_kwargs,
            round_power_of_two=args.round_power_of_two)
        logger.info("Using phase reference from {}".format(args.phase_ref))
    MaskReader = {"numpy": NumpyReader, "kaldi": ScriptReader}
    mask_reader = MaskReader[args.fmt](args.mask_scp)

    num_done = 0
    with WaveWriter(args.dst_dir, fs=args.sf) as writer:
        for key, specs in spectrogram_reader:
            # if multi-channel, choose ch0
            if specs.ndim == 3:
                specs = specs[0]
            # specs: T x F
            if key in mask_reader:
                num_done += 1
                mask = mask_reader[key]
                # mask sure mask in T x F
                _, F = specs.shape
                if mask.shape[0] == F:
                    mask = np.transpose(mask)
                logger.info("Processing utterance {}...".format(key))
                if mask.shape != specs.shape:
                    raise ValueError(
                        "Dimention mismatch between mask and spectrogram"
                        "({0[0]} x {0[1]} vs {1[0]} x {1[1]}), need check configures"
                        .format(mask.shape, specs.shape))
                nsamps = spectrogram_reader.nsamps(
                    key) if args.keep_length else None
                norm = spectrogram_reader.samp_norm(
                    key) if args.mixed_norm else None
                # use phase from ref
                if phase_reader is not None:
                    angle = np.angle(phase_reader[key])
                    phase = np.exp(angle * 1j)
                    samps = istft(np.abs(specs) * mask * phase,
                                  **stft_kwargs,
                                  norm=norm,
                                  nsamps=nsamps)
                else:
                    samps = istft(specs * mask,
                                  **stft_kwargs,
                                  norm=norm,
                                  nsamps=nsamps)
                writer.write(key, samps)
    logger.info("Processed {:d} utterances over {:d}".format(
        num_done, len(spectrogram_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to separate target component from mixtures given Tf-masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Mixture wave scripts in kaldi format")
    parser.add_argument("mask_scp",
                        type=str,
                        help="Scripts of masks in kaldi's "
                        "archive or numpy's ndarray")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Location to dump separated wave files")
    parser.add_argument("--sample-frequency",
                        type=int,
                        default=16000,
                        dest="sf",
                        help="Waveform data sample frequency")
    parser.add_argument("--phase-ref",
                        type=str,
                        default="",
                        help="If assigned, use phase of it "
                        "instead of mixture")
    parser.add_argument("--mask-format",
                        dest="fmt",
                        choices=["kaldi", "numpy"],
                        default="kaldi",
                        help="Define format of masks, kaldi's "
                        "archives or numpy's ndarray")
    parser.add_argument("--keep-length",
                        action="store_true",
                        help="If ture, keep result the same length as orginal")
    parser.add_argument("--use-mixed-norm",
                        action="store_true",
                        dest="mixed_norm",
                        help="If true, keep norm of separated "
                        "same as mixed one")
    args = parser.parse_args()
    run(args)
