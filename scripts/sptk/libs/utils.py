#!/usr/bin/env python
# wujian@2018

import os
import errno
import warnings
import logging

import librosa as audio_lib
# using wf to handle wave IO because it support better than librosa
import scipy.io.wavfile as wf
import numpy as np
import scipy as sp

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps

__all__ = [
    "stft", "istft", "get_logger", "make_dir", "filekey", "write_wav",
    "read_wav"
]


def nfft(window_size):
    # nextpow2
    return int(2**np.ceil(np.log2(window_size)))


def cmat_abs(cmat):
    """
    In [4]: c = np.random.rand(500, 513) + np.random.rand(500, 513)*1j
    In [5]: %timeit np.abs(c)
    5.62 ms +- 1.75 us per loop (mean +- std. dev. of 7 runs, 100 loops each)
    In [6]: %timeit np.sqrt(c.real**2 + c.imag**2)
    2.4 ms +- 4.25 us per loop (mean +- std. dev. of 7 runs, 100 loops each)
    """
    if not np.iscomplexobj(cmat):
        raise RuntimeError(
            "function cmat_abs expect complex as input, but got {}".format(
                cmat.dtype))
    return np.sqrt(cmat.real**2 + cmat.imag**2)


def write_wav(fname, samps, fs=16000, normalize=True):
    """
    Write wav files in int16, support single/multi-channel
    """
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
    """
    STFT wrapper, using librosa
    """
    if apply_log and not apply_abs:
        warnings.warn("Ignore apply_abs=False because apply_log=True")
        apply_abs = True
    if samps.ndim != 1:
        raise RuntimeError("Invalid shape, librosa.stft accepts mono input")
    n_fft = nfft(frame_length)
    if window == "sqrthann":
        window = np.sqrt(sp.signal.hann(frame_length, sym=False))
    # orignal stft accept samps(vector) and return matrix shape as F x T
    # NOTE for librosa.stft:
    # 1) win_length <= n_fft
    # 2) if win_length is None, win_length = n_fft
    # 3) if win_length < n_fft, pad window to n_fft
    stft_mat = audio_lib.stft(
        samps,
        n_fft,
        frame_shift,
        win_length=frame_length,
        window=window,
        center=center)
    # stft_mat: F x T or N x F x T
    if apply_abs:
        stft_mat = cmat_abs(stft_mat)
    if apply_pow:
        stft_mat = np.power(stft_mat, 2)
    if apply_log:
        stft_mat = np.log(np.maximum(stft_mat, EPSILON))
    if transpose:
        stft_mat = np.transpose(stft_mat)
    return stft_mat


# accept F x T or T x F(tranpose=True)
def istft(stft_mat,
          frame_length=1024,
          frame_shift=256,
          center=False,
          window="hann",
          transpose=True,
          norm=None,
          power=None,
          nsamps=None):
    """
    iSTFT wrapper, using librosa
    """
    if transpose:
        stft_mat = np.transpose(stft_mat)
    if window == "sqrthann":
        window = np.sqrt(sp.signal.hann(frame_length, sym=False))
    # orignal istft accept stft result(matrix, shape as FxT)
    samps = audio_lib.istft(
        stft_mat,
        frame_shift,
        win_length=frame_length,
        window=window,
        center=center,
        length=nsamps)
    # keep same amplitude
    if norm:
        samps_norm = np.linalg.norm(samps, np.inf)
        samps = samps * norm / (samps_norm + EPSILON)
    # keep same power
    if power:
        samps_pow = np.linalg.norm(samps, 2)**2 / samps.size
        samps = samps * np.sqrt(power / samps_pow)
    return samps


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
    """
    Return unique index from file name
    """
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
        date_format="%Y-%m-%d %H:%M:%S",
        file=False):
    """
    Return python logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def make_dir(fdir):
    """
    Make directory 
    """
    if not fdir or os.path.exists(fdir):
        return
    try:
        os.makedirs(fdir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise RuntimeError("Error exists when mkdir -p {}".format(fdir))