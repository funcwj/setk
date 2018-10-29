#!/usr/bin/env python

# wujian@2018
"""
Esimate signal from fbank or (log)-magnitude/power spectrum using Griffin Lim algorithm
"""
import os
import argparse
import librosa as audio_lib
import numpy as np

from libs.utils import get_logger, get_stft_parser, nfft, griffin_lim, write_wav, EPSILON
from libs.data_handler import ScriptReader

logger = get_logger(__name__)


def run(args):
    griffin_lim_kwargs = {
        "frame_length": args.frame_length,
        "frame_shift": args.frame_shift,
        "window": args.window,
        "center": args.center,
        "transpose": True,
        "epochs": args.epochs
    }

    feature_reader = ScriptReader(args.feat_scp)

    if args.fbank:
        mel_kwargs = {
            "n_mels": args.num_bins,
            "fmin": args.min_freq,
            "fmax": args.max_freq,
            "htk": True
        }
        # N x F
        mel_weights = audio_lib.filters.mel(args.samp_freq,
                                            nfft(args.frame_length),
                                            **mel_kwargs)
        # F x N
        mel_inv_weights = np.linalg.pinv(mel_weights)

    for key, spec in feature_reader:
        # if log, tranform to linear
        if args.apply_log:
            spec = np.exp(spec)
        # convert fbank to spectrum
        # feat: T x N
        if args.fbank:
            spec = np.maximum(spec @ np.transpose(mel_inv_weights), EPSILON)
        # if power spectrum, tranform to magnitude spectrum
        if args.apply_pow:
            spec = np.sqrt(spec)
        if spec.shape[1] - 1 != nfft(args.frame_length) // 2:
            raise RuntimeError("Seems missing --fbank options?")
        # griffin lim
        samps = griffin_lim(spec, **griffin_lim_kwargs)
        write_wav(
            os.path.join(args.dump_dir, "{}.wav".format(key)),
            samps,
            fs=args.samp_freq,
            normalize=args.normalize)
    logger.info("Process {:d} utterance done".format(len(feature_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to restore signal from fbank/spectrogram using Griffin Lim algorithm."
        "(NOTE: fbank/mel-spectrogram performs not very well)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[get_stft_parser()])
    parser.add_argument(
        "feat_scp", type=str, help="Source fbank or magnitude script")
    parser.add_argument(
        "dump_dir", type=str, help="Location to dump estimated wave")
    parser.add_argument(
        "--sample-frequency",
        type=int,
        default=16000,
        dest="samp_freq",
        help="Waveform data sample frequency")
    parser.add_argument(
        "--apply-log",
        action="store_true",
        default=False,
        dest="apply_log",
        help="Corresponding option in feature computation")
    parser.add_argument(
        "--apply-pow",
        action="store_true",
        default=False,
        dest="apply_pow",
        help="Corresponding option in feature computation")
    parser.add_argument(
        "--normalize-samples",
        action="store_true",
        default=False,
        dest="normalize",
        help="If true, normalize sample values between [-1, 1]")
    parser.add_argument(
        "--fbank",
        action="store_true",
        default=False,
        dest="fbank",
        help="Using fbank as input features")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to iterate griffin lim algorithm")
    parser.add_argument(
        "--fbank.num-bins",
        default=40,
        type=int,
        dest="num_bins",
        help="Number of mel-bins defined in mel-filters")
    parser.add_argument(
        "--fbank.min-freq",
        default=0,
        type=int,
        dest="min_freq",
        help="Low cutoff frequency for mel bins")
    parser.add_argument(
        "--fbank.max-freq",
        default=8000,
        type=int,
        dest="max_freq",
        help="High cutoff frequency for mel bins")
    args = parser.parse_args()
    run(args)
