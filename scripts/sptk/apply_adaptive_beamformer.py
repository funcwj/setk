#!/usr/bin/env python
# coding=utf-8
# wujian@2018
#
"""
Do mvdr/gevd adaptive beamformer
"""

import argparse
import os

import numpy as np
from scipy.io import loadmat

from libs.utils import istft, get_logger, nfft, cmat_abs
from libs.opts import StftParser
from libs.data_handler import SpectrogramReader, ScriptReader, NumpyReader, WaveWriter
from libs.beamformer import MvdrBeamformer, GevdBeamformer, PmwfBeamformer
from libs.beamformer import OnlineGevdBeamformer, OnlineMvdrBeamformer

logger = get_logger(__name__)


def do_online_beamform(beamformer, speech_mask, interf_mask, stft_mat, args):
    """
    Do online beamformer(gevd, mvdr):
    Arguments:
        speech_mask: shape as T x F
        stft_mat: shape as N x F x T
    Return:
        stft_enh: shape as F x T
    """
    chunk_size = args.chunk_size
    beamformer.reset_stats(args.alpha)
    num_chunks = stft_mat.shape[-1] // chunk_size
    enh_chunks = []
    for c in range(num_chunks + 1):
        base = chunk_size * c
        if c == num_chunks:
            noise_mask = None if interf_mask is None else interf_mask[base:]
            chunk = beamformer.run(speech_mask[base:],
                                   stft_mat[:, :, base:],
                                   noise_mask=noise_mask,
                                   normalize=args.ban)
        else:
            noise_mask = None if interf_mask is None else interf_mask[
                base:base + chunk_size]
            chunk = beamformer.run(speech_mask[base:base + chunk_size],
                                   stft_mat[:, :, base:base + chunk_size],
                                   noise_mask=noise_mask,
                                   normalize=args.ban)
        enh_chunks.append(chunk)
    return np.hstack(enh_chunks)


def compute_vad_masks(spectrogram, proportion):
    """
    We ignore several minimum values and keep proportion*100% energy
    Arguments:
        spectrogram: F x T
    Return:
        vad_mask: T x F
    """
    energy_mat = cmat_abs(spectrogram)
    energy_vec = np.sort(energy_mat.flatten())
    filter_energy = np.sum(energy_vec) * (1 - proportion)
    threshold = 0
    cumsum, index = 0, 0
    while index < energy_vec.shape[0]:
        threshold = energy_vec[index]
        cumsum += threshold
        if cumsum > filter_energy:
            break
        index += 1
    # silence if 1
    vad_mask = (energy_mat < threshold)
    return vad_mask.transpose(), index


def run(args):
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "window": args.window,
        "center": args.center,  # false to comparable with kaldi
        "transpose": False  # F x T
    }
    spectrogram_reader = SpectrogramReader(
        args.wav_scp,
        round_power_of_two=args.round_power_of_two,
        **stft_kwargs)
    MaskReader = {"numpy": NumpyReader, "kaldi": ScriptReader}
    tgt_mask_reader = MaskReader[args.fmt](args.tgt_mask)
    itf_mask_reader = MaskReader[args.fmt](
        args.tgt_mask) if args.itf_mask else None
    if itf_mask_reader is not None:
        logger.info(f"Using interfering masks from {args.itf_mask}")
    online = False
    num_bins = nfft(args.frame_len) // 2 + 1
    ref_channel = args.pmwf_ref if args.pmwf_ref >= 0 else None
    supported_beamformer = {
        "mvdr": MvdrBeamformer(num_bins),
        "gevd": GevdBeamformer(num_bins),
        "pmwf-0": PmwfBeamformer(num_bins, beta=0, ref_channel=ref_channel),
        "pmwf-1": PmwfBeamformer(num_bins, beta=1, ref_channel=ref_channel)
    }
    supported_online_beamformer = {
        "mvdr": OnlineMvdrBeamformer(num_bins, args.channels, args.alpha),
        "gevd": OnlineGevdBeamformer(num_bins, args.channels, args.alpha),
    }
    if args.chunk_size <= 0:
        logger.info(f"Using offline {args.beamformer} beamformer")
        beamformer = supported_beamformer[args.beamformer]
    else:
        if args.chunk_size < 32:
            raise RuntimeError(f"Seems chunk size({args.chunk_size:.2f}) " +
                               "too small for online beamformer")
        beamformer = supported_online_beamformer[args.beamformer]
        online = True
        logger.info(f"Using online {args.beamformer} beamformer, " +
                    f"chunk size = {args.chunk_size:d}")

    num_done = 0
    with WaveWriter(args.dst_dir, fs=args.samp_freq) as writer:
        for key, stft_mat in spectrogram_reader:
            if key in tgt_mask_reader:
                power = spectrogram_reader.power(key)
                logger.info(
                    f"Processing utterance {key}, " +
                    f"signal power {10 * np.log10(power + 1e-5):.2f}...")
                # prefer T x F
                speech_mask = tgt_mask_reader[key]
                # constraint [0, 1]
                if itf_mask_reader is None:
                    speech_mask = np.minimum(speech_mask, 1)
                    interf_mask = None
                else:
                    interf_mask = itf_mask_reader[key]
                # make sure speech_mask at shape T x F
                _, F, _ = stft_mat.shape
                # if in F x T
                if speech_mask.shape[0] == F:
                    speech_mask = np.transpose(speech_mask)
                    if interf_mask is not None:
                        interf_mask = np.transpose(interf_mask)
                if 0.5 < args.vad_proportion < 1:
                    vad_mask, N = compute_vad_masks(stft_mat[0],
                                                    args.vad_proportion)
                    logger.info(f"Filtering {N} TF-masks...")
                    speech_mask = np.where(vad_mask, 1.0e-4, speech_mask)
                    if interf_mask is not None:
                        interf_mask = np.where(vad_mask, 1.0e-4, interf_mask)
                # stft_enh, stft_mat: (N) x F x T
                try:
                    if not online:
                        stft_enh = beamformer.run(speech_mask,
                                                  stft_mat,
                                                  noise_mask=interf_mask,
                                                  normalize=args.ban)
                    else:
                        stft_enh = do_online_beamform(beamformer, speech_mask,
                                                      interf_mask, stft_mat,
                                                      args)
                except np.linalg.LinAlgError:
                    logger.error(f"Raise linalg error: {key}")
                    continue
                # masking beamformer output if necessary
                if args.mask:
                    stft_enh = stft_enh * np.transpose(speech_mask)
                samps = istft(stft_enh, power=power, **stft_kwargs)
                writer.write(key, samps)
                num_done += 1
    logger.info(f"Processed {num_done:d} utterances " +
                f"out of {len(spectrogram_reader):d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to run adaptive(mvdr/gevd/pmwf) beamformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("wav_scp",
                        type=str,
                        help="Multi-channel wave scripts in kaldi format")
    parser.add_argument("tgt_mask",
                        type=str,
                        help="Scripts of target masks in kaldi's "
                        "archive or numpy's ndarray")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Location to dump enhanced wave files")
    parser.add_argument("--itf-mask",
                        type=str,
                        default="",
                        help="Scripts of interfering masks in kaldi's "
                        "archive or numpy's ndarray")
    parser.add_argument("--mask-format",
                        dest="fmt",
                        choices=["kaldi", "numpy"],
                        default="kaldi",
                        help="Define format of masks, kaldi's "
                        "archives or numpy's ndarray")
    parser.add_argument("--beamformer",
                        type=str,
                        default="mvdr",
                        choices=["mvdr", "gevd", "pmwf-0", "pmwf-1"],
                        help="Type of adaptive beamformer to apply")
    parser.add_argument("--pmwf-ref",
                        type=int,
                        default=-1,
                        help="Reference channel for PMWF beamformer")
    parser.add_argument("--sample-frequency",
                        type=int,
                        default=16000,
                        dest="samp_freq",
                        help="Waveform data sample frequency")
    parser.add_argument("--post-filter",
                        dest="ban",
                        action="store_true",
                        help="Do Blind Analytical Normalization(BAN) or not")
    parser.add_argument("--post-mask",
                        dest="mask",
                        action="store_true",
                        help="Masking enhanced spectrogram "
                        "after beamforming or not")
    parser.add_argument("--vad-proportion",
                        type=float,
                        default=1,
                        help="Energy proportion to filter "
                        "silence masks [0.5, 1]")
    parser.add_argument("--online.alpha",
                        default=0.8,
                        dest="alpha",
                        type=float,
                        help="Remember coefficient when "
                        "updating covariance matrix")
    parser.add_argument("--online.chunk-size",
                        default=-1,
                        type=int,
                        dest="chunk_size",
                        help="If >= 64, using online beamformer instead")
    parser.add_argument("--online.channels",
                        default=4,
                        type=int,
                        dest="channels",
                        help="Number of channels available")
    args = parser.parse_args()
    run(args)
