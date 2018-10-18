#!/usr/bin/env python

# wujian@2018
"""
Compute some typical spatial features(SRP/IPD/MSC)
"""

import argparse

from libs.utils import get_stft_parser, get_logger, nfft
from libs.data_handler import SpectrogramReader, ArchiveWriter
from libs.spatial import srp_phat, ipd, msc

logger = get_logger(__name__)


def compute_spatial_feats(args, S):
    if args.type == "srp":
        srp_kwargs = {
            "sample_frequency": args.samp_frequency,
            "num_doa": args.num_doa,
            "num_bins": nfft(args.frame_length) // 2 + 1,
            "samp_doa": not args.samp_tdoa
        }
        linear_topo = list(map(float, args.linear_topo.split(",")))
        return srp_phat(S, linear_topo, **srp_kwargs)
    elif args.type == "ipd":
        indexes = list(map(int, args.ipd_index.split(",")))
        if len(indexes) != 2:
            raise ValueError(
                "Invalid --ipd.index configuration detected: {}".format(
                    args.ipd_index))
        if S.ndim < 3:
            raise ValueError("Only one-channel STFT available")
        L, R = indexes
        return ipd(S[L], S[R], sin=args.ipd_sin)
    else:
        return msc(S, context=args.context)


def run(args):
    stft_kwargs = {
        "frame_length": args.frame_length,
        "frame_shift": args.frame_shift,
        "window": args.window,
        "center": args.center,  # false to comparable with kaldi
        "transpose": True  # T x F
    }
    spectrogram_reader = SpectrogramReader(args.wav_scp, **stft_kwargs)

    num_utts = 0
    with ArchiveWriter(args.dup_ark, args.scp) as writer:
        for key, spectrogram in spectrogram_reader:
            # spectrogram: shape NxTxF
            feats = compute_spatial_feats(args, spectrogram)
            # feats: T x F
            writer.write(key, feats)
            num_utts += 1
    logger.info("Processed {} for {:d} utterances".format(
        args.type.upper(), num_utts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to compute some typical spatial features, egs: SRP/MSC/IPD. ("
        "SRP: SRP-PHAT Anguler Spectrum, MSC: Magnitude Squared Coherence, IPD: Interchannel Phase Difference)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[get_stft_parser()])

    parser.add_argument(
        'wav_scp', type=str, help="Multi-channel wave scripts in kaldi format")
    parser.add_argument(
        "dup_ark",
        type=str,
        help="Location to dump features in kaldi's archives")
    parser.add_argument(
        "--scp",
        type=str,
        default="",
        help="If assigned, generate corresponding feature scripts")
    parser.add_argument(
        "--type",
        type=str,
        default="srp",
        choices=["srp", "msc", "ipd"],
        help="Type of spatial features to compute")
    parser.add_argument(
        "--srp.sample-rate",
        type=int,
        dest="samp_frequency",
        default=16000,
        help="Sample frequency of input wave")
    parser.add_argument(
        "--srp.sample-tdoa",
        dest="samp_tdoa",
        action="store_true",
        help="Sample TDoA instead of DoA when computing spectrum")
    parser.add_argument(
        "--srp.num_doa",
        type=int,
        dest="num_doa",
        default=181,
        help="Number of DoA to sampled from 0 to 180 degress")
    parser.add_argument(
        "--srp.topo",
        type=str,
        dest="linear_topo",
        default="0,0.2,0.4,0.8",
        help="Topology description of microphone arrays")
    parser.add_argument(
        "--ipd.sin",
        dest="ipd_sin",
        action="store_true",
        help="Append sinIPD to cosIPD spatial features")
    parser.add_argument(
        "--ipd.index",
        type=str,
        dest="ipd_index",
        default="0,1",
        help="Given two channels to compute IPD spatial features")
    parser.add_argument(
        "--msc.context",
        type=int,
        dest="context",
        default=1,
        help="Value of context in MSC computation")
    args = parser.parse_args()
    run(args)
