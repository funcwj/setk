#!/usr/bin/env python

# wujian@2018
"""
Compute melspectrogram/fbank features(using librosa kernels) and write in kaldi format
"""

import argparse
import librosa as audio_lib
import numpy as np

from libs.utils import stft, get_logger, nfft, EPSILON
from libs.data_handler import SpectrogramReader, ArchiveWriter

logger = get_logger(__name__)


def run(args):
    mel_kwargs = {
        "n_mels": args.num_bins,
        "fmin": args.min_freq,
        "fmax": args.max_freq,
        "htk": True
    }
    stft_kwargs = {
        "frame_length": args.frame_length,
        "frame_shift": args.frame_shift,
        "window": args.window,
        "center": args.center,  # false to comparable with kaldi
        "apply_log": False,
        "apply_pow": args.apply_pow,
        "normalize": args.normalize,
        "apply_abs": True,
        "transpose": False  # F x T
    }

    spectrogram_reader = SpectrogramReader(args.wav_scp, **stft_kwargs)
    # N x F
    mel_weights = audio_lib.filters.mel(args.samp_freq,
                                        nfft(args.frame_length), **mel_kwargs)
    num_utts = 0

    with ArchiveWriter(args.dup_ark, args.scp) as writer:
        for key, spect in spectrogram_reader:
            # N x F * F x T = N * T => T x N
            fbank = np.transpose(np.dot(mel_weights, spect))
            if args.apply_log:
                fbank = np.log(np.maximum(fbank, EPSILON))
            writer.write(key, fbank)
            num_utts += 1
    logger.info("Process {:d} utterances".format(num_utts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to extract melspectrogram/fbank features(using sptk's librosa kernels) "
        "and write as kaldi's archives")
    parser.add_argument(
        "wav_scp",
        type=str,
        help="Source location of wave scripts in kaldi format")
    parser.add_argument(
        "dup_ark", type=str, help="Location to dump spectrogram's features")
    parser.add_argument(
        "--scp",
        type=str,
        default="",
        help="If assigned, generate corresponding scripts for archives")
    parser.add_argument(
        "--sample-frequency",
        type=int,
        default=16000,
        dest="samp_freq",
        help="Waveform data sample frequency, default = 16000")
    parser.add_argument(
        "--frame-length",
        type=int,
        default=1024,
        dest="frame_length",
        help="Frame length in number of samples, default = 1024(16kHz)")
    parser.add_argument(
        "--frame-shift",
        type=int,
        default=256,
        dest="frame_shift",
        help="Frame shift in number of samples, default = 256(16kHz)")
    parser.add_argument(
        "--center",
        action="store_true",
        default=False,
        dest="center",
        help="Parameter \'center\' in librosa.stft functions")
    parser.add_argument(
        "--apply-log",
        action="store_true",
        default=False,
        dest="apply_log",
        help="If true, using log mel-spectrogram instead of linear")
    parser.add_argument(
        "--apply-pow",
        action="store_true",
        default=False,
        dest="apply_pow",
        help="If true, extract power spectrogram instead of energy spectrogram"
    )
    parser.add_argument(
        "--normalize-samples",
        action="store_true",
        default=False,
        dest="normalize",
        help="If true, normalize sample values between [-1, 1]")
    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        dest="window",
        help=
        "Type of window function, see scipy.signal.get_window, default = hann")
    parser.add_argument(
        "--num-bins",
        default=40,
        type=int,
        dest="num_bins",
        help="Number of mel-bins defined in mel-filters, default = 40")
    parser.add_argument(
        "--min-freq",
        default=0,
        type=int,
        dest="min_freq",
        help="Low cutoff frequency for mel bins, default = 0")
    parser.add_argument(
        "--max-freq",
        default=8000,
        type=int,
        dest="max_freq",
        help="High cutoff frequency for mel bins, default = 8000(16kHz)")
    args = parser.parse_args()
    run(args)
