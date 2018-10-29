#!/usr/bin/env python
# wujian@2018

import os
import warnings
import logging
import argparse

import librosa as audio_lib
# using wf to handle wave IO because it support better than librosa
import scipy.io.wavfile as wf
import numpy as np

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps

__all__ = ["stft", "istft", "get_logger"]


def nfft(window_size):
    # nextpow2
    return int(2**np.ceil(int(np.log2(window_size))))


def write_wav(fname, samps, fs=16000, normalize=True):
    if normalize:
        samps = samps * MAX_INT16
    # scipy.io.wavfile.write could write single/multi-channel files
    # for multi-channel, accept ndarray [Nsamples, Nchannels]
    if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
        samps = np.transpose(samps)
        samps = np.squeeze(samps)
    # same as MATLAB and kaldi
    samps_int16 = samps.astype(np.int16)
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile instead
    wf.write(fname, fs, samps_int16)


def read_wav(fname, normalize=True, return_rate=False):
    """
    Read wave files using scipy.io.wavfile(support multi-channel)
    """
    # samps_int16: N x C or N
    #   N: number of samples
    #   C: number of channels
    samp_rate, samps_int16 = wf.read(fname)
    # N x C => C x N
    samps = samps_int16.astype(np.float)
    # tranpose because I used to put channel axis first
    if samps.ndim != 1:
        samps = np.transpose(samps)
    # normalize like MATLAB and librosa
    if normalize:
        samps = samps / MAX_INT16
    if return_rate:
        return samp_rate, samps
    return samps


# return F x T or T x F(tranpose=True)
def stft(samps,
         frame_length=1024,
         frame_shift=256,
         center=False,
         window="hann",
         apply_abs=False,
         apply_log=False,
         apply_pow=False,
         transpose=True):
    if apply_log and not apply_abs:
        warnings.warn("Ignore apply_abs=False because apply_log=True")
        apply_abs = True
    if samps.ndim != 1:
        raise RuntimeError("Invalid shape, librosa.stft accepts mono input")
    # orignal stft accept samps(vector) and return matrix shape as F x T
    # NOTE for librosa.stft:
    # 1) win_length <= n_fft
    # 2) if win_length is None, win_length = n_fft
    # 3) if win_length < n_fft, pad window to n_fft
    stft_mat = audio_lib.stft(
        samps,
        nfft(frame_length),
        frame_shift,
        frame_length,
        window=window,
        center=center)
    # stft_mat: F x T or N x F x T
    if apply_abs:
        stft_mat = np.abs(stft_mat)
    if apply_pow:
        stft_mat = np.power(stft_mat, 2)
    if apply_log:
        stft_mat = np.log(np.maximum(stft_mat, EPSILON))
    if transpose:
        stft_mat = np.transpose(stft_mat)
    return stft_mat


# accept F x T or T x F(tranpose=True)
def istft(file,
          stft_mat,
          frame_length=1024,
          frame_shift=256,
          center=False,
          window="hann",
          transpose=True,
          normalize=True,
          norm=None,
          fs=16000,
          nsamps=None):
    if transpose:
        stft_mat = np.transpose(stft_mat)
    # orignal istft accept stft result(matrix, shape as FxT)
    samps = audio_lib.istft(
        stft_mat,
        frame_shift,
        frame_length,
        window=window,
        center=center,
        length=nsamps)
    # renorm if needed
    if norm:
        samps_norm = np.linalg.norm(samps, np.inf)
        samps = samps * norm / samps_norm
    write_wav(file, samps, fs=fs, normalize=normalize)


def griffin_lim(magnitude,
                frame_length=1024,
                frame_shift=256,
                window="hann",
                center=True,
                transpose=True,
                epochs=100):
    # TxF -> FxT
    if transpose:
        magnitude = np.transpose(magnitude)
    n_fft = nfft(frame_length)
    angle = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    samps = audio_lib.istft(
        magnitude * angle,
        frame_shift,
        frame_length,
        window=window,
        center=center)
    for _ in range(epochs):
        stft_mat = audio_lib.stft(
            samps,
            n_fft,
            frame_shift,
            frame_length,
            window=window,
            center=center)
        angle = np.exp(1j * np.angle(stft_mat))
        samps = audio_lib.istft(
            magnitude * angle,
            frame_shift,
            frame_length,
            window=window,
            center=center)
    return samps


def filekey(path):
    fname = os.path.basename(path)
    if not fname:
        raise ValueError("{}(Is directory path?)".format(path))
    token = fname.split(".")
    if len(token) == 1:
        return token[0]
    else:
        return '.'.join(token[:-1])


def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_stft_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--frame-length",
        type=int,
        default=1024,
        dest="frame_length",
        help="Frame length in number of samples(related to sample frequency)")
    parser.add_argument(
        "--frame-shift",
        type=int,
        default=256,
        dest="frame_shift",
        help="Frame shift in number of samples(related to sample frequency)")
    parser.add_argument(
        "--center",
        action="store_true",
        default=False,
        dest="center",
        help="Value of parameter \'center\' in librosa.stft functions")
    parser.add_argument(
        "--window",
        default="hann",
        dest="window",
        help="Type of window function, see scipy.signal.get_window")
    return parser