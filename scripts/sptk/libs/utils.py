#!/usr/bin/env python
# wujian@2018

import os
import warnings
import logging
import argparse

import librosa as audio_lib
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
    # same as MATLAB and kaldi
    samps_int16 = samps.astype(np.int16)
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile instead
    wf.write(fname, fs, samps_int16)


def read_wav(fname, normalize=True):
    _, samps_int16 = wf.read(fname)
    if samps_int16.ndim != 1:
        raise ValueError(
            "Input file contains {:d} channels, now support mono only".format(
                samps_int16.ndim))
    samps = samps_int16.astype(np.float)
    if normalize:
        samps = samps / MAX_INT16
    return samps


# return F x T or T x F
def stft(file,
         frame_length=1024,
         frame_shift=256,
         center=False,
         window="hann",
         return_samps=False,
         apply_abs=False,
         apply_log=False,
         apply_pow=False,
         transpose=True,
         normalize=True):
    if not os.path.exists(file):
        raise FileNotFoundError("Input file {} do not exists!".format(file))
    if apply_log and not apply_abs:
        apply_abs = True
        warnings.warn(
            "Ignore apply_abs=False cause function return real values")
    # sr=None, using default sample frequency
    # samps, _ = audio_lib.load(file, sr=None)
    samps = read_wav(file, normalize=normalize)
    stft_mat = audio_lib.stft(
        samps,
        nfft(frame_length),
        frame_shift,
        frame_length,
        window=window,
        center=center)
    if apply_abs:
        stft_mat = np.abs(stft_mat)
    if apply_pow:
        stft_mat = np.power(stft_mat, 2)
    if apply_log:
        stft_mat = np.log(np.maximum(stft_mat, EPSILON))
    if transpose:
        stft_mat = np.transpose(stft_mat)
    return stft_mat if not return_samps else (samps, stft_mat)


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


def parse_scps(scp_path, addr_processor=lambda x: x):
    assert os.path.exists(scp_path)
    scp_dict = dict()
    with open(scp_path, 'r') as f:
        for scp in f:
            scp_tokens = scp.strip().split()
            if len(scp_tokens) != 2:
                raise RuntimeError(
                    "Error format of context \'{}\'".format(scp))
            key, addr = scp_tokens
            if key in scp_dict:
                raise ValueError("Duplicate key \'{}\' exists!".format(key))
            scp_dict[key] = addr_processor(addr)
    return scp_dict


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
        type=float,
        default=1024,
        dest="frame_length",
        help="Frame length in number of samples(related to sample frequency)")
    parser.add_argument(
        "--frame-shift",
        type=float,
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