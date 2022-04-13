#!/usr/bin/env python

# wujian@2020

import numpy as np

from .beamformer import compute_covar, solve_pevd
from .cluster import CgmmTrainer
from .utils import EPSILON, get_logger

logger = get_logger(__name__)


def compute_tap_mat(obs, taps, delay):
    """
    Arguments:
        obs: F x N x T
    Return:
        shape as F x NK x T, k means taps
    """
    F, N, T = obs.shape
    y_ = np.zeros([F, N * taps, T], dtype=obs.dtype)
    for k in range(taps):
        d = k + delay
        if d < T:
            y_[:, k * N:k * N + N, d:] = obs[:, :, :T - d]
        else:
            break
    return y_


def compute_lambda(dereverb, ctx=0):
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

    # F x T
    L = np.mean(cpw(dereverb), axis=1)
    _, T = L.shape
    counts_ = np.zeros(T)
    lambda_ = np.zeros_like(L)
    for c in range(-ctx, ctx + 1):
        s = max(c, 0)
        e = min(T, T + c)
        lambda_[:, s:e] += L[:, max(-c, 0):min(T, T - c)]
        counts_[s:e] += 1
    return np.maximum(lambda_ / counts_, EPSILON)


def wpe_step(reverb, yt, lambda_):
    """
    Args:
        reverb: reveberated observations, F x N x T
        yt: F x NK x T
        lambda_: F x T
    Return:
        dereverb: dereverbrated result, F x N x T
    """
    # F x NK x T
    yn = yt / lambda_[:, None, :]
    # F x NK x NK
    R = np.einsum("...mt,...nt->...mn", yn, yt.conj())
    # F x NK x N
    r = np.einsum("...mt,...nt->...mn", yn, reverb.conj())
    # F x NK x N
    G = np.linalg.solve(R, r)
    # filter
    dereverb = reverb - np.einsum("...na,...nb->...ab", G.conj(), yt)
    return dereverb


def wpe(reverb, taps=10, delay=3, context=1, num_iters=3):
    """
    GWPE dereverbration algorithm
    Reference:
        1.  https://github.com/fgnt/nara_wpe
        2.  Yoshioka, Takuya, and Tomohiro Nakatani. "Generalization of multi-channel
            linear prediction methods for blind MIMO impulse response shortening." IEEE
            Transactions on Audio, Speech, and Language Processing 20.10 (2012): 2707-2720.
    Arguments:
        reverb: complex spectrogram, F x N x T
        taps: number of taps for filter matrix
        delay: frame delay
        context: left/right context used to compute lambda
        num_iters: number of iterations to filter signals
    Return:
        dereverb: F x N x T
    """
    F, N, T = reverb.shape
    logger.info(f"WPE: F = {F}, N = {N}, T = {T}")
    # F x NK x T
    yt = compute_tap_mat(reverb, taps, delay)
    # F x N x T
    dereverb = reverb
    # for num_iters
    for i in range(num_iters):
        logger.info(f"WPE: iter = {i + 1}/{num_iters}...")
        # time-varying variance: F x N x T => F x T
        lambda_ = compute_lambda(dereverb, ctx=context)
        # wpe step
        dereverb = wpe_step(reverb, yt, lambda_)
    return dereverb


def facted_wpd(obs,
               cgmm_iters=10,
               wpd_iters=3,
               taps=10,
               delay=3,
               context=1,
               update_alpha=False):
    """
    Joint dereverberation & denoising
    Reference:
        1.  Nakatani, Tomohiro, and Keisuke Kinoshita. "A unified convolutional beamformer for simultaneous
            denoising and dereverberation." IEEE Signal Processing Letters 26.6 (2019): 903-907.
        2.  Boeddeker, Christoph, et al. "Jointly optimal dereverberation and beamforming." ICASSP 2020-2020
            IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.
        3.  Nakatani, Tomohiro, and Keisuke Kinoshita. "Maximum likelihood convolutional beamformer for
            simultaneous denoising and dereverberation." 2019 27th European Signal Processing Conference
            (EUSIPCO). IEEE, 2019.
    Args:
        obs: N x T x F
    Return:
        wpd_enh: T x F
    """
    obs = np.einsum("ntf->fnt", obs)
    F, N, T = obs.shape
    logger.info(f"Facted WPD: F = {F}, N = {N}, T = {T}")
    # F x T
    wpd_enh = None
    # F x NK x T
    yt = compute_tap_mat(obs, taps, delay)
    # iterations
    for i in range(wpd_iters):
        logger.info(f"Facted WPD: iter = {i + 1}/{wpd_iters}...")
        logger.info("Facted WPD: perform wpe...")
        # compute lambda: F x T
        if i == 0:
            lambda_ = compute_lambda(obs, ctx=context)
        else:
            lambda_ = np.abs(wpd_enh)**2
        # lower bound
        lambda_ = np.maximum(lambda_, EPSILON)
        # F x N x T
        der = wpe_step(obs, yt, lambda_)
        # N x F x T
        der_r = np.einsum("fnt->nft", der)
        logger.info("Facted WPD: mask estimation...")
        # TF-mask
        trainer = CgmmTrainer(der_r, 2, update_alpha=update_alpha)
        # F x T
        tf_mask = trainer.train(cgmm_iters)
        logger.info("Facted WPD: perform weighted mvdr...")
        # F x N x N
        Rd = np.einsum("...nt,...mt->...nm", der / lambda_[:, None],
                       der.conj()) / der.shape[-1]
        # F x N x N
        Rs = compute_covar(der_r, tf_mask[0].T)
        # F x N
        sv = solve_pevd(Rs)
        # F x N
        Rd_inv_sv = np.linalg.solve(Rd, sv)
        denominator = np.einsum("...d,...d->...", sv.conj(), Rd_inv_sv)
        # F x N
        weight = Rd_inv_sv / denominator[:, None]
        # F x T
        wpd_enh = np.einsum("...n,...nt->...t", weight.conj(), der)
    return tf_mask.T, wpd_enh.T
