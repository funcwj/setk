#!/usr/bin/env python

# wujian@2020

import os
import argparse
import pathlib

import numpy as np
import scipy.signal as ss

from libs.utils import EPSILON, read_wav, write_wav
from libs.opts import str2tuple


def coeff_snr(sig_pow, ref_pow, snr):
    """
    For
        mix = Sa + alpha*Sb
    Given
        SNR = 10*log10[Pa/(Pb * alpha^2)]
    we got
        alpha = Pa/[Pb*10^(SNR/10)]^0.5
    """
    return (ref_pow / (sig_pow * 10**(snr / 10) + EPSILON))**0.5


def add_room_response(spk, rir, early_energy=False, sr=16000):
    """
    Convolute source signal with selected rirs
    Args
        spk: S
        rir: N x R
    Return
        revb: N x S
    """
    if spk.ndim != 1:
        raise RuntimeError(f"Can not convolve rir with {spk.ndim}D signals")
    S = spk.shape[-1]
    revb = [ss.fftconvolve(spk, r)[:S] for r in rir]
    # revb = ss.oaconvolve(spk[None, ...], rir)[..., :S]
    revb = np.asarray(revb)

    if early_energy:
        rir_ch0 = rir[0]
        rir_peak = np.argmax(rir_ch0)
        rir_beg_idx = max(0, int(rir_peak - 0.001 * sr))
        rir_end_idx = min(rir_ch0.size, int(rir_peak + 0.05 * sr))
        early_rir = rir_ch0[rir_beg_idx:rir_end_idx]
        early_rev = ss.oaconvolve(spk, early_rir)[:S]
        return revb, np.mean(early_rev**2)
    else:
        return revb, np.mean(revb[0]**2)


def add_speaker(mix_nsamps,
                src_spk,
                src_begin,
                sdr,
                src_rir=None,
                channel=-1,
                sr=16000):
    """
    Mix source speakers
    """
    spk_image, spk_power = [], []
    for i, spk in enumerate(src_spk):
        if src_rir is None:
            src = spk[None, ...] if spk.ndim == 1 else spk
            spk_image.append(src)
            spk_power.append(np.mean(src[0]**2))
        else:
            rir = src_rir[i]
            if rir.ndim == 1:
                rir = rir[None, ...]
            if channel >= 0:
                if rir.ndim == 2:
                    rir = rir[channel:channel + 1]
            revb, p = add_room_response(spk, rir, sr=sr)
            spk_image.append(revb)
            spk_power.append(p)
    # make mix
    N, _ = spk_image[0].shape
    mix = [np.zeros([N, mix_nsamps]) for _ in src_spk]
    # start mixing
    ref_power = spk_power[0]
    for i, image in enumerate(spk_image):
        dur = image.shape[-1]
        beg = src_begin[i]
        coeff = 1 if i == 0 else coeff_snr(spk_power[i], ref_power, sdr[i])
        mix[i][..., beg:beg + dur] += coeff * image
    return mix


def add_point_noise(mix_nsamps,
                    ref_power,
                    noise,
                    noise_begin,
                    snr,
                    noise_rir=None,
                    channel=-1,
                    sr=16000):
    """
    Add pointsource noises
    """
    image = []
    image_power = []
    for i, noise in enumerate(noise):
        beg = noise_begin[i]
        dur = min(noise.shape[-1], mix_nsamps - beg)

        if noise_rir is None:
            src = noise[None, ...] if noise.ndim == 1 else noise
            image.append(src)
            image_power.append(np.mean(src[0, :dur]**2))
        else:
            rir = noise_rir[i]
            if rir.ndim == 1:
                rir = rir[None, ...]
            if channel >= 0:
                if rir.ndim == 2:
                    rir = rir[channel:channel + 1]
            revb, revb_power = add_room_response(noise[:dur], rir, sr=sr)
            image.append(revb)
            image_power.append(revb_power)
    # make noise mix
    N, _ = image[0].shape
    mix = np.zeros([N, mix_nsamps])
    # start mixing
    for i, img in enumerate(image):
        beg = noise_begin[i]
        coeff = coeff_snr(image_power[i], ref_power, snr[i])
        mix[..., beg:beg + dur] += coeff * img[..., :dur]
    return mix


def run(args):
    def arg_audio(src_args, beg=None):
        if src_args:
            src_path = src_args.split(",")
            if beg:
                beg = [int(v) for v in beg.split(",")]
                return [
                    read_wav(s, sr=args.sr, beg=b)
                    for s, b in zip(src_path, beg)
                ]
            else:
                return [read_wav(s, sr=args.sr) for s in src_path]
        else:
            return None

    def arg_float(src_args):
        return [float(s) for s in src_args.split(",")] if src_args else None

    src_spk = arg_audio(args.src_spk)
    src_rir = arg_audio(args.src_rir)
    if src_rir:
        if len(src_rir) != len(src_spk):
            raise RuntimeError(
                f"Number of --src-rir={args.src_rir} do not match with " +
                f"--src-spk={args.src_spk} option")
    sdr = arg_float(args.src_sdr)
    if len(src_spk) > 1 and not sdr:
        raise RuntimeError("--src-sdr need to be assigned for " +
                           f"--src-spk={args.src_spk}")
    if sdr:
        if len(src_spk) - 1 != len(sdr):
            raise RuntimeError("Number of --src-snr - 1 do not match with " +
                               "--src-snr option")
        sdr = [0] + sdr

    src_begin = arg_float(args.src_begin)
    if src_begin:
        src_begin = [int(v) for v in src_begin]
    else:
        src_begin = [0 for _ in src_spk]

    # number samples of the mixture
    mix_nsamps = max([b + s.size for b, s in zip(src_begin, src_spk)])

    point_noise = arg_audio(args.point_noise, beg=args.point_noise_offset)
    point_noise_rir = arg_audio(args.point_noise_rir)
    if point_noise:
        if point_noise_rir:
            if len(point_noise) != len(point_noise_rir):
                raise RuntimeError(
                    f"Number of --point-noise-rir={args.point_noise_rir} do not match with "
                    + f"--point-noise={args.point_noise} option")
        point_snr = arg_float(args.point_noise_snr)
        if not point_snr:
            raise RuntimeError("--point-noise-snr need to be assigned for " +
                               f"--point-noise={args.point_noise}")
        if len(point_noise) != len(point_snr):
            raise RuntimeError(
                f"Number of --point-noise-snr={args.point_noise_snr} do not match with "
                + f"--point-noise={args.point_noise} option")

        point_begin = arg_float(args.point_noise_begin)
        if point_begin:
            point_begin = [int(v) for v in point_begin]
        else:
            point_begin = [0 for _ in point_noise]

    isotropic_noise = arg_audio(args.isotropic_noise,
                                beg=str(args.isotropic_noise_offset))
    if isotropic_noise:
        isotropic_noise = isotropic_noise[0]
        isotropic_snr = arg_float(args.isotropic_noise_snr)
        if not isotropic_snr:
            raise RuntimeError(
                "--isotropic-snr need to be assigned for " +
                f"--isotropic-noise={args.isotropic_noise} option")
        isotropic_snr = isotropic_snr[0]
    else:
        isotropic_snr = None

    # add speakers
    spk = add_speaker(mix_nsamps,
                      src_spk,
                      src_begin,
                      sdr,
                      src_rir=src_rir,
                      channel=args.dump_channel,
                      sr=args.sr)
    spk_utt = sum(spk)
    mix = spk_utt.copy()

    spk_power = np.mean(spk_utt[0]**2)
    if point_noise:
        noise = add_point_noise(mix_nsamps,
                                spk_power,
                                point_noise,
                                point_begin,
                                point_snr,
                                noise_rir=point_noise_rir,
                                channel=args.dump_channel,
                                sr=args.sr)
        if spk_utt.shape[0] != noise.shape[0]:
            raise RuntimeError("Channel mismatch between source speaker " +
                               "configuration and pointsource noise's, " +
                               f"{spk_utt.shape[0]} vs {noise.shape[0]}")
        mix = spk_utt + noise
    else:
        noise = None

    ch = args.dump_channel
    if isotropic_noise is not None:
        N, _ = spk_utt.shape
        if N == 1:
            if isotropic_noise.ndim == 1:
                isotropic_noise = isotropic_noise[None, ...]
            else:
                if ch >= 0:
                    isotropic_noise = isotropic_noise[ch:ch + 1]
                else:
                    raise RuntimeError(
                        "Single channel mixture vs multi-channel "
                        "isotropic noise")
        else:
            if isotropic_noise.shape[0] != N:
                raise RuntimeError(
                    "Channel number mismatch between mixture and isotropic noise, "
                    + f"{N} vs {isotropic_noise.shape[0]}")

        dur = min(mix_nsamps, isotropic_noise.shape[-1])
        isotropic_chunk = isotropic_noise[0, :dur]
        power = np.mean(isotropic_chunk**2)
        coeff = coeff_snr(power, spk_power, isotropic_snr)
        mix[..., :dur] += coeff * isotropic_chunk

        if noise is None:
            noise = coeff * isotropic_chunk
        else:
            noise[..., :dur] += coeff * isotropic_chunk

    factor = args.norm_factor / (np.max(np.abs(mix)) + EPSILON)

    write_wav(args.mix, factor * mix, sr=args.sr)

    if args.dump_ref_dir:
        basename = os.path.basename(args.mix)
        ref_dir = pathlib.Path(args.dump_ref_dir)
        ref_dir.mkdir(parents=True, exist_ok=True)
        # has noise
        if noise is not None:
            write_wav(ref_dir / "noise" / basename, factor * noise, sr=args.sr)
        # one speaker
        if len(spk) == 1:
            write_wav(ref_dir / "clean" / basename,
                      factor * spk[0],
                      sr=args.sr)
        else:
            for i, s in enumerate(spk):
                write_wav(ref_dir / f"spk{i + 1}" / basename,
                          factor * s,
                          sr=args.sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do audio data simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("mix", type=str, help="Audio to output")
    parser.add_argument("--dump-ref-dir",
                        type=str,
                        default="",
                        help="Directory of the reference audio to output")
    parser.add_argument("--src-spk",
                        type=str,
                        required=True,
                        help="Source speakers, e.g., spk1.wav,spk2.wav")
    parser.add_argument("--src-rir",
                        type=str,
                        default="",
                        help="RIRs for each source speakers")
    parser.add_argument("--src-sdr",
                        type=str,
                        default="",
                        help="SDR for each speakers (if needed)")
    parser.add_argument("--src-begin",
                        type=str,
                        default="",
                        help="Begining samples on the mixture utterances")
    parser.add_argument("--point-noise",
                        type=str,
                        default="",
                        help="Add pointsource noises")
    parser.add_argument("--point-noise-rir",
                        type=str,
                        default="",
                        help="RIRs of the pointsource noises (if needed)")
    parser.add_argument("--point-noise-snr",
                        type=str,
                        default="",
                        help="SNR of the pointsource noises")
    parser.add_argument("--point-noise-begin",
                        type=str,
                        default="",
                        help="Begining samples of the "
                        "pointsource noises on the mixture "
                        "utterances (if needed)")
    parser.add_argument("--point-noise-offset",
                        type=str,
                        default="",
                        help="Add from the offset position "
                        "of the pointsource noise")
    parser.add_argument("--isotropic-noise",
                        type=str,
                        default="",
                        help="Add isotropic noises")
    parser.add_argument("--isotropic-noise-snr",
                        type=str,
                        default="",
                        help="SNR of the isotropic noises")
    parser.add_argument("--isotropic-noise-offset",
                        type=int,
                        default=0,
                        help="Add noise from the offset position "
                        "of the isotropic noise")
    parser.add_argument("--dump-channel",
                        type=int,
                        default=-1,
                        help="Index of the channel to dump out (-1 means all)")
    parser.add_argument('--norm-factor',
                        type=float,
                        default=0.9,
                        help="Normalization factor of the final output")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Value of the sample rate")
    args = parser.parse_args()
    run(args)