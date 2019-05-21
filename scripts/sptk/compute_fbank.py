#!/usr/bin/env python

# wujian@2018
"""
Compute melspectrogram/fbank features(using librosa kernels) and write in kaldi format
"""

import argparse
import librosa as audio_lib
import numpy as np

from libs.utils import stft, get_logger, nfft, EPSILON
from libs.opts import StftParser
from libs.data_handler import SpectrogramReader, ArchiveWriter
from libs.exraw import BinaryWriter

logger = get_logger(__name__)


def run(args):
    mel_kwargs = {
        "n_mels": args.num_bins,
        "fmin": args.min_freq,
        "fmax": args.max_freq,
        "htk": True
    }
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "round_power_of_two": args.round_power_of_two,
        "window": args.window,
        "center": args.center,  # false to comparable with kaldi
        "apply_log": False,
        "apply_pow": False,
        "normalize": args.norm,
        "apply_abs": True,
        "transpose": False  # F x T
    }

    if args.max_freq > args.samp_freq // 2:
        raise RuntimeError("Max frequency for mel exceeds sample frequency")
    spectrogram_reader = SpectrogramReader(args.wav_scp, **stft_kwargs)
    # N x F
    mel_weights = audio_lib.filters.mel(
        args.samp_freq,
        nfft(args.frame_len) if args.round_power_of_two else args.frame_len,
        **mel_kwargs)
    WriterImpl = {"kaldi": ArchiveWriter, "exraw": BinaryWriter}[args.format]

    with WriterImpl(args.dup_ark, args.scp) as writer:
        for key, spectrum in spectrogram_reader:
            # N x F * F x T = N * T => T x N
            fbank = np.transpose(
                np.dot(mel_weights,
                       spectrum[0] if spectrum.ndim == 3 else spectrum))
            if args.log:
                fbank = np.log(np.maximum(fbank, EPSILON))
            writer.write(key, fbank)
    logger.info("Process {:d} utterances".format(len(spectrogram_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to extract mel-spectrogram/fbank features(using sptk's librosa kernels) "
        "and write as kaldi's archives",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Source location of wave scripts in kaldi format")
    parser.add_argument("dup_ark",
                        type=str,
                        help="Location to dump spectrogram's features")
    parser.add_argument("--format",
                        type=str,
                        default="kaldi",
                        choices=["kaldi", "exraw"],
                        help="Output archive format, see "
                        "format in sptk/libs/exraw.py")
    parser.add_argument("--scp",
                        type=str,
                        default="",
                        help="If assigned, generate corresponding "
                        "scripts for archives")
    parser.add_argument("--sample-frequency",
                        type=int,
                        default=16000,
                        dest="samp_freq",
                        help="Waveform data sample frequency")
    parser.add_argument("--apply-log",
                        action="store_true",
                        dest="log",
                        help="If true, using log mel-spectrogram "
                        "instead of linear")
    parser.add_argument("--normalize-samples",
                        action="store_true",
                        dest="norm",
                        help="If true, normalize sample "
                        "values between [-1, 1]")
    parser.add_argument("--num-bins",
                        default=40,
                        type=int,
                        help="Number of mel-bins defined in mel-filters")
    parser.add_argument("--min-freq",
                        default=0,
                        type=int,
                        help="Low cutoff frequency for mel bins")
    parser.add_argument("--max-freq",
                        default=8000,
                        type=int,
                        help="High cutoff frequency for mel bins")
    args = parser.parse_args()
    run(args)
