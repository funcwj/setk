#!/usr/bin/env python
# coding=utf-8
# wujian@2018

import numpy as np
import scipy as sp

from .utils import EPSILON, cmat_abs
"""
Implement for some classic beamformer
"""

__all__ = [
    "FixedBeamformer", "DSBeamformer", "SupperDirectiveBeamformer",
    "MvdrBeamformer", "GevdBeamformer", "PmwfBeamformer",
    "OnlineMvdrBeamformer", "OnlineGevdBeamformer"
]


def do_ban(weight, Rvv):
    """
    Do Blind Analytical Normalization(BAN)
    Arguments: (for N: num_mics, F: num_bins)
        weight: shape as F x N
        Rvv: shape as F x N x N
    Return:
        ban_weight: shape as F x N
    """
    nominator = np.einsum("...a,...ab,...bc,...c->...", np.conj(weight), Rvv,
                          Rvv, weight)
    denominator = np.einsum("...a,...ab,...b->...", np.conj(weight), Rvv,
                            weight)
    filters = np.sqrt(cmat_abs(nominator)) / np.maximum(
        np.real(denominator), EPSILON)
    return filters[:, None] * weight


def solve_pevd(Rxx, Rvv=None):
    """
    Return principle eigenvector of covariance matrix (pair)
    Arguments: (for N: num_mics, F: num_bins)
        Rxx: shape as F x N x N
        Rvv: same as Rxx if not None
    Return:
        pvector: shape as F x N
    """
    if Rvv is None:
        # batch(faster) version
        # eigenvals: F x N, ascending order
        # eigenvecs: F x N x N on each columns, |vec|_2 = 1
        # NOTE: eigenvalues computed by np.linalg.eig is not necessarily ordered.
        _, eigenvecs = np.linalg.eigh(Rxx)
        return eigenvecs[:, :, -1]
    else:
        F, N, _ = Rxx.shape
        pvec = np.zeros((F, N), dtype=np.complex)
        for f in range(F):
            try:
                # sp.linalg.eigh returns eigen values in ascending order
                _, eigenvecs = sp.linalg.eigh(Rxx[f], Rvv[f])
                pvec[f] = eigenvecs[:, -1]
            except np.linalg.LinAlgError:
                try:
                    eigenvals, eigenvecs = sp.linalg.eig(Rxx[f], Rvv[f])
                    pvec[f] = eigenvecs[:, np.argmax(eigenvals)]
                except np.linalg.LinAlgError:
                    raise RuntimeError(
                        "LinAlgError when computing eig on frequency "
                        "{:d}: \nRxx = {}, \nRvv = {}".format(
                            f, Rxx[f], Rvv[f]))
        return pvec


def rank1_constraint(Rxx, Rvv=None):
    """
    Return generalized rank1 approximation of covariance matrix
    Arguments: (for N: num_mics, F: num_bins)
        Rxx: shape as F x N x N
        Rvv: same as Rxx if not None
    Return:
        rank1_mat: shape as F x N x N
    """
    if Rvv is None:
        eigenvals, eigenvecs = np.linalg.eigh(Rxx)
        pvals = eigenvals[:, -1]
        pvecs = eigenvecs[:, :, -1]
        rank1_appro = np.einsum("...a,...b->...ab", pvals[:, None] * pvecs,
                                pvecs.conj())
    else:
        num_bins = Rxx.shape[0]
        rank1_appro = np.zeros_like(Rxx, dtype=Rxx.dtype)
        for f in range(num_bins):
            try:
                # sp.linalg.eigh returns eigen values in ascending order
                eigenvals, eigenvecs = sp.linalg.eigh(Rxx[f], Rvv[f])
                rank1_appro[f] = eigenvals[-1] * np.outer(
                    eigenvecs[:, -1], eigenvecs[:, -1].conj())
            except np.linalg.LinAlgError:
                raise RuntimeError(
                    "LinAlgError when computing eig on frequency "
                    "{:d}: \nRxx = {}, \nRvv = {}".format(f, Rxx[f], Rvv[f]))
    return rank1_appro


def beam_pattern(weight, steer_vector):
    """
    Compute beam pattern of the fixed beamformer
    Arguments (for N: num_mics, F: num_bins, D: num_doas, B: num_beams)
        weight: B x F x N or F x N (single or multiple beams)
        steer_vector: F x D x N
    Return
        pattern: [F x D, ...] or F x D
    """

    if weight.shape[-1] != steer_vector.shape[-1] or weight.shape[
            -2] != steer_vector.shape[0]:
        raise RuntimeError("Shape mismatch between weight and steer_vector")

    def single_beam(weight, sv):
        # F x D x 1
        bp = sv @ np.expand_dims(weight.conj(), -1)
        return np.squeeze(np.abs(bp))

    if weight.ndim == 2:
        return single_beam(weight, steer_vector)
    elif weight.ndim == 3:
        return [single_beam(w, steer_vector) for w in weight]
    else:
        raise RuntimeError(f"Expect 2/3D beam weights, got {weight.ndim}")


class Beamformer(object):
    def __init__(self):
        pass

    def beamform(self, weight, spectrogram):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            weight: shape as F x N
            spectrogram: shape as N x F x T
        Return:
            stft_enhan: shape as F x T
        """
        # N x F x T => F x N x T
        if weight.shape[0] != spectrogram.shape[1] or weight.shape[
                1] != spectrogram.shape[0]:
            raise ValueError(
                "Input spectrogram do not match with weight, {} vs "
                "{}".format(weight.shape, spectrogram.shape))
        spectrogram = np.transpose(spectrogram, (1, 0, 2))
        spectrogram = np.einsum("...n,...nt->...t", weight.conj(), spectrogram)
        return spectrogram

    def run(self, spectrogram):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            spectrogram: shape as N x F x T
        Return:
            stft_enhan: shape as F x T
        """
        raise NotImplementedError


class SupervisedBeamformer(Beamformer):
    """
    BaseClass for TF-mask based beamformer
    """

    def __init__(self, num_bins):
        super(SupervisedBeamformer, self).__init__()
        self.num_bins = num_bins

    def compute_covar_mat(self, target_mask, spectrogram):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            target_mask: shape as T x F, same shape as network output
            spectrogram: shape as N x F x T
        Return:
            covar_mat: shape as F x N x N
        """
        if target_mask.shape[1] != self.num_bins or target_mask.ndim != 2:
            raise ValueError(
                "Input mask matrix should be shape as [num_frames x num_bins], now is {}"
                .format(target_mask.shape))
        if spectrogram.shape[1] != target_mask.shape[1] or spectrogram.shape[
                2] != target_mask.shape[0]:
            raise ValueError(
                "Shape of input spectrogram do not match with mask matrix, {} vs {}"
                .format(spectrogram.shape, target_mask.shape))
        # num_bins x num_mics x num_frames
        spectrogram = np.transpose(spectrogram, (1, 0, 2))
        # num_bins x 1 x num_frames
        mask = np.expand_dims(np.transpose(target_mask), axis=1)
        denominator = np.maximum(np.sum(mask, axis=-1, keepdims=True), 1e-6)
        # num_bins x num_mics x num_mics
        covar_mat = np.einsum("...dt,...et->...de", mask * spectrogram,
                              spectrogram.conj()) / denominator
        return covar_mat

    def weight(self, Rxx, Rvv):
        """
        Need reimplement for different beamformer
        """
        raise NotImplementedError

    def run(self, speech_mask, spectrogram, noise_mask=None, normalize=False):
        """
        Run beamformer based on TF-mask
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            speech_mask: shape as T x F, same shape as network output
            spectrogram: shape as N x T x F
        Returns:
            stft_enhan: shape as F x T
        """
        Rvv = self.compute_covar_mat(
            1 - speech_mask if noise_mask is None else noise_mask, spectrogram)
        Rxx = self.compute_covar_mat(speech_mask, spectrogram)
        # Rxx = rank1_constraint(Rxx)
        weight = self.weight(Rxx, Rvv)
        return self.beamform(
            do_ban(weight, Rvv) if normalize else weight, spectrogram)


class OnlineSupervisedBeamformer(SupervisedBeamformer):
    """
    Online version of SupervisedBeamformer
    """

    def __init__(self, num_bins, num_channels, alpha=0.8):
        super(OnlineSupervisedBeamformer, self).__init__(num_bins)
        self.covar_mat_shape = (num_bins, num_channels, num_channels)
        self.reset_stats(alpha=alpha)

    def reset_stats(self, alpha=0.8):
        self.Rxx = np.zeros(self.covar_mat_shape, dtype=np.complex)
        self.Rvv = np.zeros(self.covar_mat_shape, dtype=np.complex)
        self.alpha = alpha
        self.reset = True

    def run(self, speech_mask, spectrogram, noise_mask=None, normalize=False):
        """
        Run beamformer based on TF-mask, online version
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            speech_mask: shape as T x F, same shape as network output
            spectrogram: shape as N x T x F
        Returns:
            stft_enhan: shape as F x T
        """
        Rvv = self.compute_covar_mat(
            1 - speech_mask if noise_mask is None else noise_mask, spectrogram)
        Rxx = self.compute_covar_mat(speech_mask, spectrogram)
        # update stats
        phi = 1 if self.reset else (1 - self.alpha)
        self.Rxx = self.Rxx * self.alpha + phi * Rxx
        self.Rvv = self.Rvv * self.alpha + phi * Rvv
        # do beamforming
        weight = self.weight(self.Rxx, self.Rvv)
        return self.beamform(
            do_ban(weight, Rvv) if normalize else weight, spectrogram)


class FixedBeamformer(Beamformer):
    """
    Fixed Beamformer, need predefined weights
    """

    def __init__(self, weight):
        super(FixedBeamformer, self).__init__()
        # F x N
        self.weight = weight

    def run(self, spectrogram):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            spectrogram: shape as N x F x T
        Return:
            stft_enhan: shape as F x T
        """
        return self.beamform(self.weight, spectrogram)


class DSBeamformer(Beamformer):
    """
    Delay and Sum Beamformer
    """

    def __init__(self, linear_topo):
        super(DSBeamformer, self).__init__()
        if type(linear_topo) is not list:
            raise TypeError(
                "type of parameter \'linear_topo\' should be python list")
        self.linear_topo = np.array(linear_topo)
        self.num_mics = len(linear_topo)

    def weight(self, doa, num_bins, c=340, sample_rate=16000):
        """
        Arguments:
            doa: direction of arrival, in angle
            num_bins: number of frequency bins
        Return:
            weight: F x N
        """
        # e^{-j \omega \tau}, \omega = 2 \pi f
        tau = np.cos(doa * np.pi / 180) * self.linear_topo / c
        omega = np.pi * np.arange(num_bins) * sample_rate / (num_bins - 1)
        return np.exp(-1j * np.outer(omega, tau))

    def run(self, doa, spectrogram, c=340, sample_rate=16000):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            doa: direction of arrival, in angle
            spectrogram: shape as N x F x T
        Return:
            stft_enhan: shape as F x T
        """
        if spectrogram.shape[0] != self.num_mics:
            raise ValueError(
                "Shape of spectrogram do not match with number of microphones, {} vs {}"
                .format(self.num_mics, spectrogram.shape[0]))
        num_bins = spectrogram.shape[1]
        weight = self.weight(doa, num_bins, c=c, sample_rate=sample_rate)
        return self.beamform(weight, spectrogram)


class SupperDirectiveBeamformer(DSBeamformer):
    """
    SupperDirective Beamformer in diffused noise field
    """

    def __init__(self, linear_topo):
        super(SupperDirectiveBeamformer, self).__init__(linear_topo)

    def compute_diffuse_covar(self, num_bins, c=340, sample_rate=16000):
        """
        Compute coherence matrix of diffuse field noise
            \\Gamma(\\omega)_{ij} = \\sinc(\\omega \\tau_{ij}) = \\sinc(2 \\pi f \\tau_{ij})
        """
        covar = np.zeros([num_bins, self.num_mics, self.num_mics])
        dist = np.tile(self.linear_topo, (4, 1))
        for f in range(num_bins):
            omega = np.pi * f * sample_rate / (num_bins - 1)
            covar[f] = np.sinc((dist - np.transpose(dist)) * omega /
                               c) + np.eye(self.num_mics) * 1.0e-5
        return covar

    def weight(self, doa, num_bins, c=340, sample_rate=16000):
        """
        Arguments:
            doa: direction of arrival, in angle
            num_bins: number of frequency bins
        Return:
            weight: shape as F x N
        """
        steer_vector = super(SupperDirectiveBeamformer,
                             self).weight(doa,
                                          num_bins,
                                          c=c,
                                          sample_rate=sample_rate)
        Rvv = self.compute_diffuse_covar(num_bins,
                                         c=c,
                                         sample_rate=sample_rate)
        numerator = np.linalg.solve(Rvv, steer_vector)
        denominator = np.einsum("...d,...d->...", steer_vector.conj(),
                                numerator)
        return numerator / np.expand_dims(denominator, axis=-1)


class MvdrBeamformer(SupervisedBeamformer):
    """
    MVDR(Minimum Variance Distortionless Response) Beamformer
    Formula:
        h_mvdr(f) = R(f)_{vv}^{-1}*d(f) / [d(f)^H*R(f)_{vv}^{-1}*d(f)]
    where
        d(f) = P(R(f)_{xx}) P: principle eigenvector
    """

    def __init__(self, num_bins):
        super(MvdrBeamformer, self).__init__(num_bins)

    def weight(self, Rxx, Rvv):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            Rxx: shape as F x N x N
            Rvv: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        steer_vector = self.compute_steer_vector(Rxx)
        numerator = np.linalg.solve(Rvv, steer_vector)
        denominator = np.einsum("...d,...d->...", steer_vector.conj(),
                                numerator)
        return numerator / np.expand_dims(denominator, axis=-1)

    def compute_steer_vector(self, Rxx):
        """
        Compute steer vector using PCA methods
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            Rxx: shape as F x N x N
        Returns:
            steer_vector: shape as F x N
        """
        return solve_pevd(Rxx)


class PmwfBeamformer(SupervisedBeamformer):
    """
    PMWF(Parameterized Multichannel Non-Causal Wiener Filter)
    Reference:
        1) Erdogan H, Hershey J R, Watanabe S, et al. Improved MVDR Beamforming Using 
            Single-Channel Mask Prediction Networks[C]//Interspeech. 2016: 1981-1985.
        2) Souden M, Benesty J, Affes S. On optimal frequency-domain multichannel 
            linear filtering for noise reduction[J]. IEEE Transactions on audio, speech, 
            and language processing, 2010, 18(2): 260-276.
    Formula:
        h_pmwf(f) = numerator(f)*u(f) / (beta + trace(numerator(f)))
            beta = 0 => mvdr
            beta = 1 => mcwf
    where
        numerator(f) = R(f)_vv^{-1}*R(f)_xx = R(f)_vv^{-1}*(R(f)_yy^{-1} - R(f)_vv^{-1})
                     = R(f)_vv^{-1}*R(f)_yy^{-1} - I
        trace(numerator(f)) = trace(R(f)_vv^{-1}*R(f)_yy^{-1} - I)
                            = trace(R(f)_vv^{-1}*R(f)_yy^{-1}) - N
        u(f): pre-assigned or estimated using snr in 1)
    """

    def __init__(self, num_bins, beta=0, ref_channel=None):
        super(PmwfBeamformer, self).__init__(num_bins)
        self.ref_channel = ref_channel
        self.beta = beta

    def _snr(self, weight, Rxx, Rvv):
        """
        Estimate post-snr suppose we got weight, along whole frequency band
        Formula:
            snr(w) = \\sum_f w(f)^H*R(f)_xx*w(f) / \\sum_f w(f)^H*R(f)_vv*w(f) 
        """
        pow_s = np.einsum("...fa,...fab,...fb->...", np.conj(weight), Rxx,
                          weight)
        pow_n = np.einsum("...fa,...fab,...fb->...", np.conj(weight), Rvv,
                          weight)
        return np.real(pow_s) / np.maximum(EPSILON, np.real(pow_n))

    def weight(self, Rxx, Rvv):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            Rxx: shape as F x N x N
            Rvv: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        _, N, _ = Rxx.shape
        numerator = np.linalg.solve(Rvv, Rxx)
        if self.ref_channel is None:
            # using snr to select channel
            ref_channel = np.argmax(
                [self._snr(numerator[:, :, c], Rxx, Rvv) for c in range(N)])
        else:
            ref_channel = self.ref_channel
        if ref_channel >= N:
            raise RuntimeError(
                "Reference channel ID exceeds total channels: {:d} vs {:d}".
                format(ref_channel, N))
        denominator = self.beta + np.trace(numerator, axis1=1, axis2=2)
        return numerator[:, :, ref_channel] / denominator[:, None]


class GevdBeamformer(SupervisedBeamformer):
    """
    Max-SNR/GEV(Generalized Eigenvalue Decomposition) Beamformer
    Formula:
        h_gevd(f) = P(R(f)_xx, R(f)_vv) P: max generalzed eigenvector
    which maximum:
        snr(f) = h(f)^H*R(f)_xx^H*h(f) / h(f)^H*R(f)_vv^H*h(f)
    """

    def __init__(self, num_bins):
        super(GevdBeamformer, self).__init__(num_bins)

    def weight(self, Rxx, Rvv):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            Rxx: shape as F x N x N
            Rvv: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        return solve_pevd(Rxx, Rvv)


class OnlineGevdBeamformer(OnlineSupervisedBeamformer):
    """
    Online version of GEVD beamformer
    """

    def __init__(self, num_bins, num_channels, alpha=0.8):
        super(OnlineGevdBeamformer, self).__init__(num_bins,
                                                   num_channels,
                                                   alpha=alpha)

    def weight(self, Rxx, Rvv):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            Rxx: shape as F x N x N
            Rvv: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        return solve_pevd(Rxx, Rvv)


class OnlineMvdrBeamformer(OnlineSupervisedBeamformer):
    """
    Online version of MVDR beamformer
    """

    def __init__(self, num_bins, num_channels, alpha=0.8):
        super(OnlineMvdrBeamformer, self).__init__(num_bins,
                                                   num_channels,
                                                   alpha=alpha)

    def weight(self, Rxx, Rvv):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            Rxx: shape as F x N x N
            Rvv: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        steer_vector = solve_pevd(Rxx)
        numerator = np.linalg.solve(Rvv, steer_vector)
        denominator = np.einsum("...d,...d->...", steer_vector.conj(),
                                numerator)
        return numerator / np.expand_dims(denominator, axis=-1)