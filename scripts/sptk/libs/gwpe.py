#!/usr/bin/env python

# wujian@2019
"""
GWPE dereverbration algorithm
Reference: 
    [1] https://github.com/fgnt/nara_wpe
    [2] Yoshioka, Takuya, and Tomohiro Nakatani. "Generalization of multi-channel 
        linear prediction methods for blind MIMO impulse response shortening." IEEE 
        Transactions on Audio, Speech, and Language Processing 20.10 (2012): 2707-2720.
"""

import numpy as np


def _compute_tap_mat(spectra, taps, delay):
    """
    Arguments:
        spectra: F x N x T
    Return:
        shape as F x NK x T, k means taps
    """
    F, N, T = spectra.shape
    y_ = np.zeros([F, N * taps, T], dtype=spectra.dtype)
    for k in range(taps):
        d = k + delay
        if d < T:
            y_[:, k * N:k * N + N, d:] = spectra[:, :, :T - d]
        else:
            break
    return y_


def wpe(reverb, taps=10, delay=3, context=0, num_iters=3):
    """
    Arguments:
        reverb: complex spectrogram, F x N x T
    Return:
        dereverb: F x N x T
    """

    def her(mat):
        return mat.transpose(0, 2, 1).conj()

    def cpw(mat):
        return mat.real**2 + mat.imag**2

    # F x NK x T
    yt = _compute_tap_mat(reverb, taps, delay)
    # F x N x T
    dereverb = reverb
    # for num_iters
    for _ in range(num_iters):
        # F x N x T => F x T
        lamba = np.mean(cpw(dereverb), axis=1)
        # F x NK x T
        yn = yt / lamba[:, None, :]
        # F x NK x NK
        R = np.matmul(yn, her(yt))
        # F x NK x N
        r = np.matmul(yn, her(reverb))
        # F x NK x N
        G = np.linalg.solve(R, r)
        # filter
        dereverb = reverb - np.matmul(her(G), yt)
    return dereverb