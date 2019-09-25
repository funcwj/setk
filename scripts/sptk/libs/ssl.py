#!/usr/bin/env python

# wujian@2019
"""
Sound Source Localization (SSL) Module
"""
import numpy as np

from .utils import cmat_abs


def ml_ssl(stft, sv, compression=0, eps=1e-8, norm=False, mask=None):
    """
    Maximum likelihood SSL
    Arguments:
        stft: STFT transform result, M x T x F
        sv: steer vector in each directions, A x M x F
        norm: normalze STFT or not
        mask: TF-mask for source, T x F x (N)
    Return:
        loglike: likelihood on each directions
    """
    _, T, F = stft.shape
    if mask is None:
        mask = np.ones([T, F])
    # make sure sv is normalized
    sv = sv / np.linalg.norm(sv, axis=1, keepdims=True)
    if norm:
        stft = stft / np.maximum(cmat_abs(stft), eps)
    ssh_cor = np.abs(np.einsum("mtf,mtf->tf", stft, stft.conj()))
    ssv_cor = np.abs(np.einsum("amf,mtf->atf", sv, stft.conj()))**2
    # A x T x F
    delta = ssh_cor[None, ...] - ssv_cor / (1 + eps)
    if compression <= 0:
        tf_loglike = -np.log(np.maximum(delta, eps))
    else:
        tf_loglike = -np.power(delta, compression)
    # masking
    if mask.ndim == 2:
        loglike = np.sum(mask[None, ...] * tf_loglike, (1, 2))
    else:
        loglike = np.einsum("ntf,atf->na", mask, tf_loglike)
    return np.argmax(loglike, axis=-1)