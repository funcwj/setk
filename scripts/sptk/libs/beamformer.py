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
    "MvdrBeamformer", "GevdBeamformer", "PmwfBeamformer"
]


def do_ban(weight, noise_covar):
    """
    Do Blind Analytical Normalization(BAN)
    Arguments: (for N: num_mics, F: num_bins)
        weight: shape as F x N
        noise_covar: shape as F x N x N
    Return:
        ban_weight: shape as F x N
    """
    nominator = np.einsum("...a,...ab,...bc,...c->...", np.conj(weight),
                          noise_covar, noise_covar, weight)
    denominator = np.einsum("...a,...ab,...b->...", np.conj(weight),
                            noise_covar, weight)
    filters = np.sqrt(cmat_abs(nominator)) / np.maximum(
        np.real(denominator), EPSILON)
    return filters[:, None] * weight


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


class FixedBeamformer(Beamformer):
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
        return np.exp(-1j * np.outer(tau, omega))

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
        steer_vector = super(SupperDirectiveBeamformer, self).weight(
            doa, num_bins, c=c, sample_rate=sample_rate)
        noise_covar = self.compute_diffuse_covar(
            num_bins, c=c, sample_rate=sample_rate)
        numerator = np.linalg.solve(noise_covar, steer_vector)
        denominator = np.einsum("...d,...d->...", steer_vector.conj(),
                                numerator)
        return numerator / np.expand_dims(denominator, axis=-1)


class MvdrBeamformer(SupervisedBeamformer):
    """
    MVDR(Minimum Variance Distortionless Response) Beamformer
    Formula:
        h_mvdr(f) = R(f)_{vv}^{-1}*d(f) / [d(f)^H*R(f)_{vv}^{-1}*d(f)]
    where
        d(f) = P(R(f)_{xx}) P: max eigenvector
    """

    def __init__(self, num_bins):
        super(MvdrBeamformer, self).__init__(num_bins)

    def weight(self, steer_vector, noise_covar):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            steer_vector: shape as F x N
            noise_covar: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        numerator = np.linalg.solve(noise_covar, steer_vector)
        denominator = np.einsum("...d,...d->...", steer_vector.conj(),
                                numerator)
        return numerator / np.expand_dims(denominator, axis=-1)

    def compute_steer_vector(self, speech_covar):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            speech_covar: shape as F x N x N
        Returns:
            steer_vector: shape as F x N
        """
        # batch(faster) version
        # eigenvals: F x N, ascending order
        # eigenvecs: F x N x N on each columns, |vec|_2 = 1
        # NOTE: eigenvalues computed by np.linalg.eig is not necessarily ordered.
        _, eigenvecs = np.linalg.eigh(speech_covar)
        steer_vector = eigenvecs[:, :, -1]
        return steer_vector

    def run(self,
            speech_mask,
            spectrogram,
            noise_mask=None,
            normalize=False,
            method="v1"):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            speech_mask: shape as T x F, same shape as network output
            spectrogram: shape as N x T x F
            method: different way to compute speech covariance matrix
        Returns:
            stft_enhan: shape as F x T
        """
        noise_covar = self.compute_covar_mat(
            1 - speech_mask if noise_mask is None else noise_mask, spectrogram)
        if method == "v1":
            speech_covar = self.compute_covar_mat(speech_mask, spectrogram)
        else:
            # seems bad this branch
            speech_covar = self.compute_covar_mat(
                np.ones_like(speech_mask), spectrogram) - noise_covar
        steer_vector = self.compute_steer_vector(speech_covar)
        weight = self.weight(steer_vector, noise_covar)
        return self.beamform(
            do_ban(weight, noise_covar) if normalize else weight, spectrogram)


class PmwfBeamformer(SupervisedBeamformer):
    """
    PMWF(Parameterized Multichannel Non-Causal Wiener Filter), treat beta = 0 now
    Reference:
        1) Erdogan H, Hershey J R, Watanabe S, et al. Improved MVDR Beamforming Using 
            Single-Channel Mask Prediction Networks[C]//Interspeech. 2016: 1981-1985.
        2) Souden M, Benesty J, Affes S. On optimal frequency-domain multichannel 
            linear filtering for noise reduction[J]. IEEE Transactions on audio, speech, 
            and language processing, 2010, 18(2): 260-276.
    Formula:
        h_pmwf(f) = numerator(f)*u(f) / trace(numerator(f))
    where
        numerator(f) = R(f)_vv^{-1}*R(f)_xx = R(f)_vv^{-1}*(R(f)_yy^{-1} - R(f)_vv^{-1})
                     = R(f)_vv^{-1}*R(f)_yy^{-1} - I
        trace(numerator(f)) = trace(R(f)_vv^{-1}*R(f)_yy^{-1} - I)
                            = trace(R(f)_vv^{-1}*R(f)_yy^{-1}) - N
        u(f): pre-assigned or estimated using snr in 1)
    """

    def __init__(self, num_bins, ref_channel=None):
        super(PmwfBeamformer, self).__init__(num_bins)
        self.ref_channel = ref_channel

    def _snr(self, weight, speech_covar, noise_covar):
        """
        Estimate post-snr suppose we got weight, along whole frequency band
        Formula:
            snr(w) = \\sum_f w(f)^H*R(f)_xx*w(f) / \\sum_f w(f)^H*R(f)_vv*w(f) 
        """
        pow_s = np.einsum("...fa,...fab,...fb->...", np.conj(weight),
                          speech_covar, weight)
        pow_n = np.einsum("...fa,...fab,...fb->...", np.conj(weight),
                          noise_covar, weight)
        return np.real(pow_s) / np.maximum(EPSILON, np.real(pow_n))

    def weight(self, speech_covar, noise_covar):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            speech_covar: shape as F x N x N
            noise_covar: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        _, N, _ = speech_covar.shape
        numerator = np.linalg.solve(noise_covar, speech_covar)
        if self.ref_channel is None:
            # using snr to select channel
            ref_channel = np.argmax([
                self._snr(numerator[:, :, c], speech_covar, noise_covar)
                for c in range(N)
            ])
        else:
            ref_channel = self.ref_channel
        if ref_channel >= N:
            raise RuntimeError(
                "Reference channel ID exceeds total channels: {:d} vs {:d}".
                format(ref_channel, N))
        denominator = np.trace(numerator, axis1=1, axis2=2)
        return numerator[:, :, ref_channel] / denominator[:, None]

    def run(self, speech_mask, spectrogram, noise_mask=None, normalize=False):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            speech_mask: shape as T x F, same shape as network output
            spectrogram: shape as N x T x F
        Returns:
            stft_enhan: shape as F x T
        """
        noise_covar = self.compute_covar_mat(
            1 - speech_mask if noise_mask is None else noise_mask, spectrogram)
        speech_covar = self.compute_covar_mat(speech_mask, spectrogram)
        weight = self.weight(speech_covar, noise_covar)
        return self.beamform(
            do_ban(weight, noise_covar) if normalize else weight, spectrogram)


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

    def weight(self, speech_covar, noise_covar):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            speech_covar: shape as F x N x N
            noise_covar: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        num_mics = speech_covar.shape[1]
        weight = np.zeros((self.num_bins, num_mics), dtype=np.complex)
        for f in range(self.num_bins):
            try:
                # sp.linalg.eigh returns eigen values in ascending order
                _, eigenvecs = sp.linalg.eigh(speech_covar[f], noise_covar[f])
                weight[f] = eigenvecs[:, -1]
            except np.linalg.LinAlgError:
                raise RuntimeError(
                    "LinAlgError when computing eign on frequency "
                    "{:d}: \nRxx = {}, \nRvv = {}".format(
                        f, speech_covar[f], noise_covar[f]))
                # weight[f] = num_mics * np.ones(num_mics) / np.trace(
                #     noise_covar[f])
        return weight

    def run(self,
            speech_mask,
            spectrogram,
            noise_mask=None,
            normalize=False,
            method="v1"):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            speech_mask: shape as T x F, same shape as network output
            spectrogram: shape as N x T x F
        Returns:
            stft_enhan: shape as F x T
        """
        noise_covar = self.compute_covar_mat(
            1 - speech_mask if noise_mask is None else noise_mask, spectrogram)
        if method == "v1":
            speech_covar = self.compute_covar_mat(speech_mask, spectrogram)
        else:
            speech_covar = self.compute_covar_mat(
                np.ones_like(speech_mask), spectrogram) - noise_covar
        weight = self.weight(speech_covar, noise_covar)
        return self.beamform(
            do_ban(weight, noise_covar) if normalize else weight, spectrogram)
