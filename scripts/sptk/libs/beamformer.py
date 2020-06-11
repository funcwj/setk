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


def do_ban(weight, Rn):
    """
    Do Blind Analytical Normalization(BAN)
    Arguments: (for N: num_mics, F: num_bins)
        weight: shape as F x N
        Rn: shape as F x N x N
    Return:
        ban_weight: shape as F x N
    """
    nominator = np.einsum("...a,...ab,...bc,...c->...", np.conj(weight), Rn,
                          Rn, weight)
    denominator = np.einsum("...a,...ab,...b->...", np.conj(weight), Rn,
                            weight)
    filters = np.sqrt(cmat_abs(nominator)) / np.maximum(
        np.real(denominator), EPSILON)
    return filters[:, None] * weight


def solve_pevd(Rs, Rn=None):
    """
    Return principle eigenvector of covariance matrix (pair)
    Arguments: (for N: num_mics, F: num_bins)
        Rs: shape as F x N x N
        Rn: same as Rs if not None
    Return:
        pvector: shape as F x N
    """
    if Rn is None:
        # batch(faster) version
        # eigenvals: F x N, ascending order
        # eigenvecs: F x N x N on each columns, |vec|_2 = 1
        # NOTE: eigenvalues computed by np.linalg.eig is not necessarily ordered.
        _, eigenvecs = np.linalg.eigh(Rs)
        return eigenvecs[:, :, -1]
    else:
        F, N, _ = Rs.shape
        pvec = np.zeros((F, N), dtype=np.complex)
        for f in range(F):
            try:
                # sp.linalg.eigh returns eigen values in ascending order
                _, eigenvecs = sp.linalg.eigh(Rs[f], Rn[f])
                pvec[f] = eigenvecs[:, -1]
            except np.linalg.LinAlgError:
                try:
                    eigenvals, eigenvecs = sp.linalg.eig(Rs[f], Rn[f])
                    pvec[f] = eigenvecs[:, np.argmax(eigenvals)]
                except np.linalg.LinAlgError:
                    raise RuntimeError(
                        "LinAlgError when computing eig on frequency "
                        "{f}: \nRs = {Rs[f]}, \nRn = {Rn[f]}")
        return pvec


def rank1_constraint(Rs, Rn=None):
    """
    Return generalized rank1 approximation of covariance matrix
    Arguments: (for N: num_mics, F: num_bins)
        Rs: shape as F x N x N
        Rn: same as Rs if not None
    Return:
        rank1_appro: shape as F x N x N
    """
    pvecs = solve_pevd(Rs, Rn=Rn)
    if Rn is not None:
        pvecs = np.einsum('...ab,...b->...a', Rn, pvecs)
    # rank1 approximation
    rank1_appro = np.einsum("...a,...b->...ab", pvecs, pvecs.conj())
    # scale back
    rank1_scale = np.trace(Rs, axis1=-1, axis2=-2) / np.maximum(
        np.trace(rank1_appro, axis1=-1, axis2=-2), EPSILON)
    rank1_appro = rank1_scale[..., None, None] * rank1_appro
    return rank1_appro


def compute_covar(obs, tf_mask):
    """
    Arguments: (for N: num_mics, F: num_bins, T: num_frames)
        tf_mask: shape as T x F, same shape as network output
        obs: shape as N x F x T
    Return:
        covar_mat: shape as F x N x N
    """
    # num_bins x num_mics x num_frames
    obs = np.transpose(obs, (1, 0, 2))
    # num_bins x 1 x num_frames
    mask = np.expand_dims(np.transpose(tf_mask), axis=1)
    denominator = np.maximum(np.sum(mask, axis=-1, keepdims=True), 1e-6)
    # num_bins x num_mics x num_mics
    covar_mat = np.einsum("...dt,...et->...de", mask * obs,
                          obs.conj()) / denominator
    return covar_mat


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

    def beamform(self, weight, obs):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            weight: shape as F x N
            obs: shape as N x F x T
        Return:
            stft_enhan: shape as F x T
        """
        # N x F x T => F x N x T
        if weight.shape[0] != obs.shape[1] or weight.shape[1] != obs.shape[0]:
            raise ValueError("Input obs do not match with weight, " +
                             f"{weight.shape} vs {obs.shape}")
        obs = np.transpose(obs, (1, 0, 2))
        obs = np.einsum("...n,...nt->...t", weight.conj(), obs)
        return obs


class SupervisedBeamformer(Beamformer):
    """
    BaseClass for TF-mask based beamformer
    """
    def __init__(self, num_bins):
        super(SupervisedBeamformer, self).__init__()
        self.num_bins = num_bins

    def compute_covar_mat(self, target_mask, obs):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            target_mask: shape as T x F, same shape as network output
            obs: shape as N x F x T
        Return:
            covar_mat: shape as F x N x N
        """
        if target_mask.shape[1] != self.num_bins or target_mask.ndim != 2:
            raise ValueError(
                "Input mask matrix should be shape as " +
                f"[num_frames x num_bins], now is {target_mask.shape}")
        if obs.shape[1] != target_mask.shape[1] or obs.shape[
                2] != target_mask.shape[0]:
            raise ValueError(
                "Shape of input obs do not match with " +
                f"mask matrix, {obs.shape} vs {target_mask.shape}")
        return compute_covar(obs, target_mask)

    def weight(self, Rs, Rn):
        """
        Need reimplement for different beamformer
        """
        raise NotImplementedError

    def run(self, mask_s, obs, mask_n=None, ban=False):
        """
        Run beamformer based on TF-mask
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            mask_s: shape as T x F, same shape as network output
            obs: shape as N x T x F
        Returns:
            stft_enhan: shape as F x T
        """
        Rn = self.compute_covar_mat(1 - mask_s if mask_n is None else mask_n,
                                    obs)
        Rs = self.compute_covar_mat(mask_s, obs)
        # Rs = rank1_constraint(Rs)
        weight = self.weight(Rs, Rn)
        return self.beamform(do_ban(weight, Rn) if ban else weight, obs)


class OnlineSupervisedBeamformer(SupervisedBeamformer):
    """
    Online version of SupervisedBeamformer
    """
    def __init__(self, num_bins, num_channels, alpha=0.8):
        super(OnlineSupervisedBeamformer, self).__init__(num_bins)
        self.covar_mat_shape = (num_bins, num_channels, num_channels)
        self.reset_stats(alpha=alpha)

    def reset_stats(self, alpha=0.8):
        self.Rs = np.zeros(self.covar_mat_shape, dtype=np.complex)
        self.Rn = np.zeros(self.covar_mat_shape, dtype=np.complex)
        self.alpha = alpha
        self.reset = True

    def run(self, mask_s, obs, mask_n=None, ban=False):
        """
        Run beamformer based on TF-mask, online version
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            mask_s: shape as T x F, same shape as network output
            obs: shape as N x T x F
        Returns:
            stft_enhan: shape as F x T
        """
        Rn = self.compute_covar_mat(1 - mask_s if mask_n is None else mask_n,
                                    obs)
        Rs = self.compute_covar_mat(mask_s, obs)
        # update stats
        phi = 1 if self.reset else (1 - self.alpha)
        self.Rs = self.Rs * self.alpha + phi * Rs
        self.Rn = self.Rn * self.alpha + phi * Rn
        # do beamforming
        weight = self.weight(self.Rs, self.Rn)
        return self.beamform(do_ban(weight, Rn) if ban else weight, obs)


class FixedBeamformer(Beamformer):
    """
    Fixed Beamformer, need predefined weights
    """
    def __init__(self, weight):
        super(FixedBeamformer, self).__init__()
        # F x N
        self.weight = weight

    def run(self, obs):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            obs: shape as N x F x T
        Return:
            stft_enhan: shape as F x T
        """
        return self.beamform(self.weight, obs)


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

    def run(self, doa, obs, c=340, sr=16000):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            doa: direction of arrival, in degree
            obs: shape as N x F x T
        Return:
            stft_enhan: shape as F x T
        """
        if obs.shape[0] != self.num_mics:
            raise ValueError(
                "Shape of obs do not match with number" +
                f"of microphones, {self.num_mics} vs {obs.shape[0]}")
        num_bins = obs.shape[1]
        weight = self.weight(doa, num_bins, c=c, sr=sr)
        return self.beamform(weight, obs)


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
        Rn = diffuse_covar(num_bins,
                           self.distance_mat,
                           sr=sr,
                           c=c,
                           diag_eps=diag_eps)
        numerator = np.linalg.solve(Rn, steer_vector)
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
        Rn = diffuse_covar(num_bins,
                           self.distance_mat,
                           sr=sr,
                           c=c,
                           diag_eps=diag_eps)
        numerator = np.linalg.solve(Rn, steer_vector)
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

    def weight(self, Rs, Rn):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            Rs: shape as F x N x N
            Rn: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        steer_vector = self.compute_steer_vector(Rs)
        numerator = np.linalg.solve(Rn, steer_vector)
        denominator = np.einsum("...d,...d->...", steer_vector.conj(),
                                numerator)
        return numerator / np.expand_dims(denominator, axis=-1)

    def compute_steer_vector(self, Rs):
        """
        Compute steer vector using PCA methods
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            Rs: shape as F x N x N
        Returns:
            steer_vector: shape as F x N
        """
        return solve_pevd(Rs)


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

    def _snr(self, weight, Rs, Rn):
        """
        Estimate SNR suppose we have beam weight
        Formula:
            snr(w) = sum_f w(f)^H*R(f)_xx*w(f) / sum_f w(f)^H*R(f)_vv*w(f) 
        """
        pow_s = np.einsum("...fa,...fab,...fb->...", np.conj(weight), Rs,
                          weight)
        pow_n = np.einsum("...fa,...fab,...fb->...", np.conj(weight), Rn,
                          weight)
        return np.real(pow_s) / np.maximum(EPSILON, np.real(pow_n))

    def weight(self, Rs, Rn):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            Rs: shape as F x N x N
            Rn: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        _, N, _ = Rs.shape
        # use rank1 approximation
        if self.rank1_appro == "eig":
            Rs = rank1_constraint(Rs)
        if self.rank1_appro == "gev":
            Rs = rank1_constraint(Rs, Rn=Rn)
        numerator = np.linalg.solve(Rn, Rs)
        denominator = self.beta + np.trace(numerator, axis1=1, axis2=2)
        # F x N x N
        weight_mat = numerator / denominator[..., None, None]
        if self.ref_channel < 0:
            # using snr to select channel
            est_snr = [self._snr(weight_mat[..., c], Rs, Rn) for c in range(N)]
            ref_channel = np.argmax(est_snr)
        else:
            ref_channel = self.ref_channel
        if ref_channel >= N:
            raise RuntimeError("Reference channel ID exceeds total " +
                               f"channels: {ref_channel} vs {N}")
        return weight_mat[..., ref_channel]


class GevdBeamformer(SupervisedBeamformer):
    """
    Max-SNR/GEV (Generalized Eigenvalue Decomposition) Beamformer
    Formula:
        h_gevd(f) = P(R(f)_xx, R(f)_vv) P: max generalzed eigenvector
    which maximum:
        snr(f) = h(f)^H*R(f)_xx^H*h(f) / h(f)^H*R(f)_vv^H*h(f)
    """
    def __init__(self, num_bins):
        super(GevdBeamformer, self).__init__(num_bins)

    def weight(self, Rs, Rn):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            Rs: shape as F x N x N
            Rn: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        return solve_pevd(Rs, Rn)


from .wpe import compute_tap_mat


class WpdBeamformer(object):
    """
    Weighted Power minimization Distortionless response (WPD) beamformer
    NOTE: on testing
    """
    def __init__(self, num_bins, delay=3, taps=10):
        self.num_bins = num_bins
        self.delay = delay
        self.taps = taps

    def compute_lambda(self, obs, mask_s):
        """
        Compute time-varying power of the desired signal
        Arguments:
            obs: STFT of observed signals, shape as N x F x T
            mask_s: shape as T x F
        Return:
            lambda_: F x T
        """
        # N x F x T
        mask_obs = obs * np.transpose(mask_s)
        # F x T
        lambda_ = np.maximum(np.mean(mask_obs**2, 0), EPSILON)
        return lambda_

    def compute_steer_vector(self, obs, mask_s):
        """
        Compute steer vector
        Arguments:
            obs: STFT of observed signals, shape as N x F x T
            mask_s: shape as T x F
        Return:
            sv: F x N
        """
        # F x N x N
        Rs = compute_covar(obs, mask_s)
        # F x N
        sv = solve_pevd(Rs)
        return sv

    def run(self, mask_s, obs, mask_n=None, ban=False):
        """
        Run beamformer based on TF-mask
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            mask_s: shape as T x F
            obs: STFT of observed signals, shape as N x F x T
        Returns:
            wpe_enh: shape as F x T
        """
        # F x T
        lambda_ = self.compute_lambda(obs, mask_s)
        # F x N x T
        obs = np.transpose(obs, (1, 0, 2))
        # F x NK x T
        xt = compute_tap_mat(obs, self.taps, self.delay)
        # F x NK x NK
        R = np.einsum("...mt,...nt->...mn", xt / lambda_[:, None, :],
                      xt.conj())
        # N x F x T
        obs = np.transpose(obs, (1, 0, 2))
        # F x N
        sv = self.compute_steer_vector(obs, mask_s)
        # F x NK
        sv = np.pad(sv, ((0, 0), (0, (self.taps - 1) * sv.shape[-1])))
        # F x NK
        numerator = np.linalg.solve(R, sv)
        # F
        denominator = np.einsum("...d,...d->...", sv.conj(), numerator)
        # F x NK
        weight = numerator / denominator[:, None]
        # F x T
        wpd_enh = np.einsum("...n,...nt->...t", weight.conj(), xt)
        return wpd_enh


class OnlineGevdBeamformer(OnlineSupervisedBeamformer):
    """
    Online version of GEVD beamformer
    """
    def __init__(self, num_bins, num_channels, alpha=0.8):
        super(OnlineGevdBeamformer, self).__init__(num_bins,
                                                   num_channels,
                                                   alpha=alpha)

    def weight(self, Rs, Rn):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            Rs: shape as F x N x N
            Rn: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        return solve_pevd(Rs, Rn)


class OnlineMvdrBeamformer(OnlineSupervisedBeamformer):
    """
    Online version of MVDR beamformer
    """
    def __init__(self, num_bins, num_channels, alpha=0.8):
        super(OnlineMvdrBeamformer, self).__init__(num_bins,
                                                   num_channels,
                                                   alpha=alpha)

    def weight(self, Rs, Rn):
        """
        Arguments: (for N: num_mics, F: num_bins, T: num_frames)
            Rs: shape as F x N x N
            Rn: shape as F x N x N
        Return:
            weight: shape as F x N
        """
        steer_vector = solve_pevd(Rs)
        numerator = np.linalg.solve(Rn, steer_vector)
        denominator = np.einsum("...d,...d->...", steer_vector.conj(),
                                numerator)
        return numerator / np.expand_dims(denominator, axis=-1)