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


def _compute_lambda(dereverb, ctx=0):
    """
    Compute spatial correlation matrix, using scaled identity matrix method
    Arguments:
        dereverb: F x N x T
        ctx: left/right context used to compute lambda
    Returns:
        lambda: F x T
    """

    def cpw(mat):
        return mat.real**2 + mat.imag**2

    L = np.mean(cpw(dereverb), axis=1)
    _, T = L.shape
    counts_ = np.zeros(T)
    lambda_ = np.zeros_like(L)
    for c in range(-ctx, ctx + 1):
        s = max(c, 0)
        e = min(T, T + c)
        lambda_[:, s:e] += L[:, max(-c, 0):min(T, T - c)]
        counts_[s:e] += 1
    return lambda_ / counts_


def wpe(reverb, taps=10, delay=3, context=1, num_iters=3):
    """
    Arguments:
        reverb: complex spectrogram, F x N x T
        taps: number of taps for filter matrix
        delay: frame delay
        context: left/right context used to compute lambda
        num_iters: number of iterations to filter signals
    Return:
        dereverb: F x N x T
    """

    def her(mat):
        return mat.transpose(0, 2, 1).conj()

    # F x NK x T
    yt = _compute_tap_mat(reverb, taps, delay)
    # F x N x T
    dereverb = reverb
    # for num_iters
    for _ in range(num_iters):
        # F x N x T => F x T
        lambda_ = _compute_lambda(dereverb, ctx=context)
        # F x NK x T
        yn = yt / lambda_[:, None, :]
        # F x NK x NK
        R = np.matmul(yn, her(yt))
        # F x NK x N
        r = np.matmul(yn, her(reverb))
        # F x NK x N
        G = np.linalg.solve(R, r)
        # filter
        dereverb = reverb - np.matmul(her(G), yt)
    return dereverb