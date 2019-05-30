#!/usr/bin/env python

# wujian@2019

import argparse

import numpy as np

from libs.data_handler import SpectrogramReader, ArchiveWriter
from libs.utils import get_logger, nfft, EPSILON
from libs.opts import StftParser

logger = get_logger(__name__)


def gcc_phat_diag(si,
                  sj,
                  angle_delta,
                  d,
                  speed=340,
                  num_doa=121,
                  sr=16000,
                  normalize=True,
                  num_bins=513,
                  apply_floor=True):
    """
    Compute gcc-phat between diagonal microphones
    """
    doa_samp = np.linspace(0, np.pi * 2, num_doa)
    tau = np.cos(angle_delta - doa_samp) * d / speed
    # omega = 2 * pi * fk
    omega = np.linspace(0, sr / 2, num_bins) * 2 * np.pi
    # F x D
    trans = np.exp(1j * np.outer(omega, tau))
    # coherence matrix, T x F
    coherence = np.exp(1j * (np.angle(si) - np.angle(sj)))
    # T x D
    spectrum = np.real(coherence @ trans)
    if normalize:
        spectrum = spectrum / np.max(np.maximum(np.abs(spectrum), EPSILON))
    if apply_floor:
        spectrum = np.maximum(spectrum, 0)
    return spectrum


def run(args):
    srp_pair = [
        tuple(map(int, p.split(","))) for p in args.diag_pair.split(";")
    ]
    if not len(srp_pair):
        raise RuntimeError("Bad configurations with --pair {}".format(
            args.pair))
    logger.info("Compute gcc with {}".format(srp_pair))

    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "round_power_of_two": args.round_power_of_two,
        "window": args.window,
        "center": args.center,  # false to comparable with kaldi
        "transpose": True  # T x F
    }
    num_done = 0
    num_ffts = nfft(
        args.frame_len) if args.round_power_of_two else args.frame_len
    reader = SpectrogramReader(args.wav_scp, **stft_kwargs)
    with ArchiveWriter(args.srp_ark, args.scp) as writer:
        for key, stft_mat in reader:
            num_done += 1
            srp = []
            # N x T x F
            for (i, j) in srp_pair:
                srp.append(
                    gcc_phat_diag(stft_mat[i],
                                  stft_mat[j],
                                  min(i, j) * np.pi * 2 / args.n,
                                  args.d,
                                  num_bins=num_ffts // 2 + 1,
                                  sr=args.sr,
                                  num_doa=args.num_doa))
            srp = sum(srp) / len(srp_pair)
            nan = np.sum(np.isnan(srp))
            if nan:
                raise RuntimeError("Matrix {} has nan ({:d}} items)".format(
                    key, nan))
            writer.write(key, srp)
            if not num_done % 1000:
                logger.info("Processed {:d} utterances...".format(num_done))
    logger.info("Processd {:d} utterances done".format(len(reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to compute SRP augular spectrum for circular arrays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Rspecifier for multi-channel wave")
    parser.add_argument("srp_ark", type=str, help="Location to dump features")
    parser.add_argument("--scp",
                        type=str,
                        default="",
                        help="If assigned, generate corresponding scripts")
    parser.add_argument("--n", type=int, default=6, help="Number of arrays")
    parser.add_argument("--d",
                        type=float,
                        default=0.07,
                        help="Diameter of circular array")
    parser.add_argument("--diag-pair",
                        type=str,
                        default="0,3;1,4;2,5",
                        help="Compute gcc between those diagonal arrays")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of input wave")
    parser.add_argument("--num-doa",
                        type=int,
                        default=121,
                        help="Number of DoA to sample between 0 and 2pi")
    args = parser.parse_args()
    run(args)
