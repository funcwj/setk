#!/usr/bin/env python

# wujian@2018
"""
Esimate signal from fbank or (log)-magnitude/power spectrum using Griffin Lim algorithm
"""
import argparse
import numpy as np

from libs.utils import get_logger, griffin_lim, inverse_stft
from libs.opts import StftParser, StrToBoolAction
from libs.data_handler import ScriptReader, WaveWriter, SpectrogramReader, NumpyReader

logger = get_logger(__name__)


def run(args):
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "window": args.window,
        "center": args.center,
    }

    FeatureReader = {"numpy": NumpyReader, "kaldi": ScriptReader}
    feature_reader = FeatureReader[args.fmt](args.feat_scp)

    phase_reader = None
    if args.phase_ref:
        phase_reader = SpectrogramReader(
            args.phase_ref,
            **stft_kwargs,
            round_power_of_two=args.round_power_of_two)
        logger.info(f"Using phase reference from {args.phase_ref}")

    with WaveWriter(args.dump_dir, sr=args.sr,
                    normalize=args.normalize) as writer:
        for key, spec in feature_reader:
            logger.info(f"Processing utterance {key}...")
            # if log, tranform to linear
            if args.apply_log:
                spec = np.exp(spec)
            # if power spectrum, tranform to magnitude spectrum
            if args.apply_pow:
                spec = np.sqrt(spec)
            if phase_reader is None:
                # griffin lim
                samps = griffin_lim(spec,
                                    epoches=args.epoches,
                                    transpose=True,
                                    norm=0.8,
                                    **stft_kwargs)
            else:
                if key not in phase_reader:
                    raise KeyError(f"Missing key {key} in phase reader")
                ref = phase_reader[key]
                angle = np.angle(ref[0] if ref.ndim == 3 else ref)
                phase = np.exp(angle * 1j)
                samps = inverse_stft(spec * phase, **stft_kwargs, norm=0.8)
            writer.write(key, samps)
    logger.info(f"Processed {len(feature_reader)} utterance done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to restore signal from spectrogram.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("feat_scp",
                        type=str,
                        help="Source fbank or magnitude script")
    parser.add_argument("dump_dir",
                        type=str,
                        help="Location to dump estimated wave")
    parser.add_argument("--sample-frequency",
                        type=int,
                        default=16000,
                        dest="sr",
                        help="Waveform data sample frequency")
    parser.add_argument("--feat-format",
                        dest="fmt",
                        choices=["kaldi", "numpy"],
                        default="kaldi",
                        help="Define format of features, kaldi's "
                        "archives or numpy's ndarray")
    parser.add_argument("--apply-log",
                        action=StrToBoolAction,
                        default=False,
                        help="Corresponding option in feature computation")
    parser.add_argument("--apply-pow",
                        action=StrToBoolAction,
                        default=False,
                        help="Corresponding option in feature computation")
    parser.add_argument("--normalize-samples",
                        action=StrToBoolAction,
                        default=False,
                        dest="normalize",
                        help="If true, normalize sample "
                        "values between [-1, 1]")
    parser.add_argument("--epoches",
                        type=int,
                        default=30,
                        help="Number of epoches to iterate "
                        "griffin lim algorithm")
    parser.add_argument("--phase-ref",
                        type=str,
                        default="",
                        help="If assigned, use phase of it "
                        "instead of griffin lim algorithm")
    args = parser.parse_args()
    run(args)
