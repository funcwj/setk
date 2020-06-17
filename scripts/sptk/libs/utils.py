#!/usr/bin/env python
# wujian@2018

import os
import sys
import math
import codecs
import logging
import warnings

import librosa
# using wf to handle wave IO because it support better than librosa
import soundfile as sf
import scipy.io.wavfile as wf
import scipy.signal as ss

import numpy as np

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps
default_format_str = "%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s"

__all__ = [
    "forward_stft", "inverse_stft", "get_logger", "filekey", "write_wav",
    "read_wav"
]


def nextpow2(window_size):
    # next power of two
    return 2**math.ceil(math.log2(window_size))


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


def write_wav(fname, samps, sr=16000, normalize=True):
    """
    Write wav files, support single/multi-channel
    """
    samps = samps.astype("float32" if normalize else "int16")
    # scipy.io.wavfile/soundfile could write single/multi-channel files
    # for multi-channel, accept ndarray [Nsamples, Nchannels]
    if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
        samps = np.transpose(samps)
        samps = np.squeeze(samps)
    # make dirs
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile/soundfile instead
    # wf.write(fname, sr, samps_int16)
    sf.write(fname, samps, sr)


def read_wav(fname, beg=0, end=None, normalize=True, sr=16000):
    """
    Read wave files using soundfile (support multi-channel & chunk)
    args:
        fname: file name or object
        beg, end: begin and end index for chunk-level reading
        norm: normalized samples between -1 and 1
        sr: sample rate of the audio
    return:
        samps: in shape C x N
        sr: sample rate
    """
    # samps: N x C or N
    #   N: number of samples
    #   C: number of channels
    samps, ret_sr = sf.read(fname,
                            start=beg,
                            stop=end,
                            dtype="float32" if normalize else "int16")
    if sr != ret_sr:
        raise RuntimeError(f"Expect sr={sr} of {fname}, get {ret_sr} instead")
    if not normalize:
        samps = samps.astype("float32")
    # put channel axis first
    # N x C => C x N
    if samps.ndim != 1:
        samps = np.transpose(samps)
    return samps


# return F x T or T x F (tranpose=True)
def forward_stft(samps,
                 frame_len=1024,
                 frame_hop=256,
                 round_power_of_two=True,
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
    # pad fft size to power of two or left it same as frame length
    n_fft = nextpow2(frame_len) if round_power_of_two else frame_len
    if window == "sqrthann":
        window = ss.hann(frame_len, sym=False)**0.5
    # orignal stft accept samps(vector) and return matrix shape as F x T
    # NOTE for librosa.stft:
    # 1) win_length <= n_fft
    # 2) if win_length is None, win_length = n_fft
    # 3) if win_length < n_fft, pad window to n_fft
    stft_mat = librosa.stft(samps,
                            n_fft,
                            frame_hop,
                            win_length=frame_len,
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


# accept F x T or T x F (tranpose=True)
def inverse_stft(stft_mat,
                 frame_len=1024,
                 frame_hop=256,
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
        window = ss.hann(frame_len, sym=False)**0.5
    # orignal istft accept stft result(matrix, shape as FxT)
    samps = librosa.istft(stft_mat,
                          frame_hop,
                          win_length=frame_len,
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


def griffin_lim(mag,
                frame_len=1024,
                frame_hop=256,
                round_power_of_two=True,
                window="hann",
                center=True,
                transpose=True,
                norm=None,
                epoches=30):
    """
    Griffin Lim Algothrim
    """
    # TxF -> FxT
    if transpose:
        mag = np.transpose(mag)
    n_fft = nextpow2(frame_len) if round_power_of_two else frame_len
    stft_kwargs = {
        "hop_length": frame_hop,
        "win_length": frame_len,
        "window": window,
        "center": center
    }
    phase = np.exp(2j * np.pi * np.random.rand(*mag.shape))
    samps = librosa.istft(mag * phase, **stft_kwargs)
    for _ in range(epoches):
        stft_mat = librosa.stft(samps, n_fft=n_fft, **stft_kwargs)
        phase = np.exp(1j * np.angle(stft_mat))
        samps = librosa.istft(mag * phase, **stft_kwargs)
    if norm:
        samps_norm = np.linalg.norm(samps, np.inf)
        samps = samps * norm / (samps_norm + EPSILON)
    return samps


def filekey(path):
    """
    Return unique index from file name
    """
    fname = os.path.basename(path)
    if not fname:
        raise ValueError(f"{path}: is directory path?")
    token = fname.split(".")
    if len(token) == 1:
        return token[0]
    else:
        return '.'.join(token[:-1])


def get_logger(name,
               format_str=default_format_str,
               date_format="%Y-%m-%d %H:%M:%S",
               file=False):
    """
    Get logger instance
    """
    def get_handler(handler):
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
        handler.setFormatter(formatter)
        return handler

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if file:
        logger.addHandler(get_handler(logging.FileHandler(name)))
        logger.addHandler(logging.StreamHandler())
    else:
        logger.addHandler(logging.StreamHandler())
    return logger