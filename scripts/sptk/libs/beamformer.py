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
    "FixedBeamformer", "LinearDSBeamformer", "LinearSDBeamformer",
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
                        "{f}: \nRxx = {Rxx[f]}, \nRvv = {Rvv[f]}")
        return pvec


def rank1_constraint(Rxx, Rvv=None):
    """
    Return generalized rank1 approximation of covariance matrix
    Arguments: (for N: num_mics, F: num_bins)
        Rxx: shape as F x N x N
        Rvv: same as Rxx if not None
    Return:
        rank1_appro: shape as F x N x N
    """
    pvecs = solve_pevd(Rxx, Rvv=Rvv)
    if Rvv is not None:
        pvecs = np.einsum('...ab,...b->...a', Rvv, pvecs)
    # rank1 approximation
    rank1_appro = np.einsum("...a,...b->...ab", pvecs, pvecs.conj())
    # scale back
    rank1_scale = np.trace(Rxx, axis1=-1, axis2=-2) / np.maximum(
        np.trace(rank1_appro, axis1=-1, axis2=-2), EPSILON)
    rank1_appro = rank1_scale[..., None, None] * rank1_appro
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


def diffuse_covar(num_bins, dist_mat, sr=16000, c=340, diag_eps=0.1):
    """
    Compute covarance matrices of the spherically isotropic noise field
        Gamma(omega)_{ij} = sinc(omega * tau_{ij}) = sinc(2 * pi * f * tau_{ij})
    Arguments:
        num_bins: number of the FFT points
        dist_mat: distance matrix
        sr: sample rate
        c: sound of the speed
    Return:
        covar: covariance matrix, F x N x N
    """
    N, _ = dist_mat.shape
    eps = np.eye(N) * diag_eps
    covar = np.zeros([num_bins, N, N])
    for f in range(num_bins):
        omega = np.pi * f * sr / (num_bins - 1)
        covar[f] = np.sinc(dist_mat * omega / c) + eps
    return covar


def plane_steer_vector(distance, num_bins, c=340, sr=16000):
    """
    Compute steer vector given projected distance on DoA:
    Arguments:
        distance: numpy array, N
        num_bins: number of frequency bins
    Return:
        steer_vector: F x N
    """
    omega = np.pi * np.arange(num_bins) * sr / (num_bins - 1)
    steer_vector = np.exp(-1j * np.outer(omega, distance / c))
    return steer_vector


def linear_steer_vector(topo, doa, num_bins, c=340, sr=16000):
    """
    Compute steer vector for linear array:
        [..., e^{-j omega tau_i}, ...], where omega = 2*pi * f
    0   1   ...     N - 1
    *   *   ...     *
    0   d1  ...     d(N-1)
    Arguments:
        topo: linear topo, N
        doa: direction of arrival, in degree
        num_bins: number of frequency bins
    Return:
        steer_vector: F x N
    """
    dist = np.cos(doa * np.pi / 180) * topo
    # 180 degree <---------> 0 degree
    return plane_steer_vector(dist, num_bins, c=c, sr=sr)


def circular_steer_vector(redius,
                          num_arounded,
                          doa,
                          num_bins,
                          c=349,
                          sr=16000,
                          center=False):
    """
    Compute steer vector for circle array:
        [..., e^{-j omega tau_i}, ...], where omega = 2*pi * f
    Arguments:
        redius: redius for circular array
        num_arounded: number of microphones on the circle
        doa: direction of arrival, in degree
        num_bins: number of frequency bins
        center: is there a microphone in the centor?
    Return:
        steer_vector: F x N
    """
    # N
    dirc = np.arange(num_arounded) * 2 * np.pi / num_arounded
    dist = np.cos(dirc - doa * np.pi / 180) * redius
    if center:
        # 1 + N
        dist = np.concatenate([np.array([0]), dist])
    return plane_steer_vector(-dist, num_bins, c=c, sr=sr)


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
            raise ValueError("Input spectrogram do not match with weight, " +
                             f"{weight.shape} vs {spectrogram.shape}")
        spectrogram = np.transpose(spectrogram, (1, 0, 2))
        spectrogram = np.einsum("...n,...nt->...t", weight.conj(), spectrogram)
        return spectrogram


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
                "Input mask matrix should be shape as " +
                f"[num_frames x num_bins], now is {target_mask.shape}")
        if spectrogram.shape[1] != target_mask.shape[1] or spectrogram.shape[
                2] != target_mask.shape[0]:
            raise ValueError(
                "Shape of input spectrogram do not match with " +
                f"mask matrix, {spectrogram.shape} vs {target_mask.shape}")
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

    def run(self, speech_mask, spectrogram, noise_mask=None, ban=False):
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
            do_ban(weight, Rvv) if ban else weight, spectrogram)


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

    def run(self, speech_mask, spectrogram, noise_mask=None, ban=False):
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
            do_ban(weight, Rvv) if ban else weight, spectrogram)


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
    Base DS beamformer
    """
    def __init__(self, num_mics):
        super(DSBeamformer, self).__init__()
        self.num_mics = num_mics

    def weight(self, doa, num_bins, c=340, sr=16000):
        """
        Arguments:
            doa: direction of arrival, in degree
            num_bins: number of frequency bins
        Return:
            weight: F x N
        """
        raise NotImplementedError

    def run(self, doa, spectrogram, c=340, sr=16000):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            doa: direction of arrival, in degree
            spectrogram: shape as N x F x T
        Return:
            stft_enhan: shape as F x T
        """
        if spectrogram.shape[0] != self.num_mics:
            raise ValueError(
                "Shape of spectrogram do not match with number" +
                f"of microphones, {self.num_mics} vs {spectrogram.shape[0]}")
        num_bins = spectrogram.shape[1]
        weight = self.weight(doa, num_bins, c=c, sr=sr)
        return self.beamform(weight, spectrogram)


class LinearDSBeamformer(DSBeamformer):
    """
    Delay and Sum Beamformer (for linear array)
    """
    def __init__(self, linear_topo):
        super(LinearDSBeamformer, self).__init__(len(linear_topo))
        self.linear_topo = np.array(linear_topo)

    def weight(self, doa, num_bins, c=340, sr=16000):
        """
        Arguments:
            doa: direction of arrival, in degree
            num_bins: number of frequency bins
        Return:
            weight: F x N
        """
        sv = linear_steer_vector(self.linear_topo, doa, num_bins, c=c, sr=sr)
        return sv / self.num_mics


class CircularDSBeamformer(DSBeamformer):
    """
    Delay and Sum Beamformer (for circular array)
    """
    def __init__(self, radius, num_arounded, center=False):
        super(CircularDSBeamformer,
              self).__init__(num_arounded + 1 if center else num_arounded)
        self.radius = radius
        self.center = center
        self.num_arounded = num_arounded

    def weight(self, doa, num_bins, c=340, sr=16000):
        """
        Arguments:
            doa: direction of arrival, in degree
            num_bins: number of frequency bins
        Return:
            weight: F x N
        """
        sv = circular_steer_vector(self.radius,
                                   self.num_arounded,
                                   doa,
                                   num_bins,
                                   c=c,
                                   sr=sr,
                                   center=self.center)
        return sv / self.num_mics


class LinearSDBeamformer(LinearDSBeamformer):
    """
    Linear SupperDirective Beamformer in diffused noise field
    """
    def __init__(self, linear_topo):
        super(LinearSDBeamformer, self).__init__(linear_topo)
        mat = np.tile(self.linear_topo, (self.num_mics, 1))
        self.distance_mat = np.abs(mat - np.transpose(mat))

    def weight(self, doa, num_bins, c=340, sr=16000, diag_eps=0.1):
        """
        Arguments:
            doa: direction of arrival, in degree
            num_bins: number of frequency bins
        Return:
            weight: shape as F x N
        """
        steer_vector = super(LinearSDBeamformer, self).weight(doa,
                                                              num_bins,
                                                              c=c,
                                                              sr=sr)
        Rvv = diffuse_covar(num_bins,
                            self.distance_mat,
                            sr=sr,
                            c=c,
                            diag_eps=diag_eps)
        numerator = np.linalg.solve(Rvv, steer_vector)
        denominator = np.einsum("...d,...d->...", steer_vector.conj(),
                                numerator)
        return numerator / np.expand_dims(denominator, axis=-1)


class CircularSDBeamformer(CircularDSBeamformer):
    """
    Circular SupperDirective Beamformer in diffused noise field
    """
    def __init__(self, radius, num_arounded, center=False):
        super(CircularSDBeamformer, self).__init__(radius,
                                                   num_arounded,
                                                   center=center)
        self.distance_mat = self._compute_distance_mat()

    def _compute_distance_mat(self):
        """
        Compute distance matrix D = [d_{ij}...]
        """
        distance_mat = np.zeros((self.num_mics, self.num_mics))
        if self.center:
            distance_mat[0, 1:] = self.radius
            raw = 1
        else:
            raw = 0
        ang = np.pi / self.num_arounded
        for r in range(raw, self.num_mics):
            for c in range(r + 1, self.num_mics):
                distance_mat[r, c] = np.abs(
                    np.sin((c - r) * ang) * 2 * self.radius)
        distance_mat += distance_mat.T
        return distance_mat

    def weight(self, doa, num_bins, c=340, sr=16000, diag_eps=1e-5):
        """
        Arguments:
            doa: direction of arrival, in degree
            num_bins: number of frequency bins
        Return:
            weight: shape as F x N
        """
        steer_vector = super(CircularSDBeamformer, self).weight(doa,
                                                                num_bins,
                                                                c=c,
                                                                sr=sr)
        Rvv = diffuse_covar(num_bins,
                            self.distance_mat,
                            sr=sr,
                            c=c,
                            diag_eps=diag_eps)
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
    def __init__(self, num_bins, beta=0, ref_channel=-1, rank1_appro=""):
        super(PmwfBeamformer, self).__init__(num_bins)
        self.ref_channel = ref_channel
        self.rank1_appro = rank1_appro
        self.beta = beta

    def _snr(self, weight, Rxx, Rvv):
        """
        Estimate SNR suppose we have beam weight
        Formula:
            snr(w) = sum_f w(f)^H*R(f)_xx*w(f) / sum_f w(f)^H*R(f)_vv*w(f) 
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
        # use rank1 approximation
        if self.rank1_appro == "eig":
            Rxx = rank1_constraint(Rxx)
        if self.rank1_appro == "gev":
            Rxx = rank1_constraint(Rxx, Rvv=Rvv)
        numerator = np.linalg.solve(Rvv, Rxx)
        denominator = self.beta + np.trace(numerator, axis1=1, axis2=2)
        # F x N x N
        weight_mat = numerator / denominator[..., None, None]
        if self.ref_channel < 0:
            # using snr to select channel
            est_snr = [
                self._snr(weight_mat[..., c], Rxx, Rvv) for c in range(N)
            ]
            ref_channel = np.argmax(est_snr)
        else:
            ref_channel = self.ref_channel
        if ref_channel >= N:
            raise RuntimeError("Reference channel ID exceeds total " +
                               f"channels: {ref_channel} vs {N}")
        return weight_mat[..., ref_channel]


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