#!/usr/bin/env python

# wujian@2018
"""
Trainer for some spatial clustering algorithm

CGMM Trainer
Reference:
    Higuchi T, Ito N, Yoshioka T, et al. Robust MVDR beamforming using time-frequency masks
    for online/offline ASR in noise[C]//Acoustics, Speech and Signal Processing (ICASSP),
    2016 IEEE International Conference on. IEEE, 2016: 5210-5214.

CACGMM Trainer
Reference:
    N. Ito, S. Araki, and T. Nakatani, “Complex angular central Gaussian mixture model for 
    directional statistics in mask-based microphone array signal processing,” in European 
    Signal Processing Conference (EUSIPCO). IEEE, 2016, pp. 1153–1157.
"""
import pickle
import numpy as np

from .utils import get_logger, EPSILON

logger = get_logger(__name__)

# cgmm trainer profile log
# 424.9570s/369.6120s/333.9310s/275.4740s/12.4840s
# faster = True
theta = 1e-4


# for all frames on frequency f
def CgmmLoglikelihoodFaster(R, phi):
    """
    Make computation faster
        for:
            N(y, sigma) = e^(-y^H * sigma^{-1} * y) / det(pi*sigma)
        since:
            phi = trace(y * y^H * sigma^{-1}) / N = y^H * sigma^{-1} * y / N
        then:
            N(y, phi*sigma) = e^(-y^H * sigma^{-1} * y / phi) / det(pi*sigma*phi)
                            = e^{-N} / (det(sigma) * (phi * pi)^N)
    """
    # kernels = np.trace(np.linalg.solve(R, C), axis1=1, axis2=2).real / phi
    N = R.shape[0]
    # make sure hermitian
    R = (R + np.transpose(np.conj(R))) / 2
    det = np.linalg.det(R).real
    # TODO: check why sometimes negative here
    if det <= 0:
        raise RuntimeError(
            f"Determinant of R is negative, det = {det:.4f}, R = {R}")
    loglikes = np.exp(-N) / (det * ((np.pi * phi)**N) + EPSILON)
    if np.any(np.isnan(loglikes)) or np.any(np.isinf(loglikes)):
        raise RuntimeError("Encounter loglike = NAN/INF in "
                           "CgmmLoglikehoodV() on some time axis")
    return loglikes


class CgmmTrainer(object):
    """
    CgmmTrainer for two components only: speech & noise
    """
    def __init__(self, X, Ms=None):
        # N x F x T => F x N x T
        X = X.transpose([1, 0, 2])

        # self.X = X
        self.num_bins, self.num_channels, self.num_frames = X.shape

        # init R{n,s}
        if Ms is None:
            self.Rs = np.einsum("...dt,...et->...de", X,
                                X.conj()) / self.num_frames
            self.Rn = np.array([
                np.eye(self.num_channels, self.num_channels, dtype=np.complex)
                for f in range(self.num_bins)
            ])
        else:
            # Ms: T x F => F x 1 x T
            Ms = np.expand_dims(np.transpose(Ms), axis=1)
            denominator_s = np.maximum(np.sum(Ms, axis=-1, keepdims=True),
                                       1e-6)
            self.Rs = np.einsum("...dt,...et->...de", Ms * X,
                                X.conj()) / denominator_s
            denominator_n = np.maximum(np.sum(1 - Ms, axis=-1, keepdims=True),
                                       1e-6)
            self.Rn = np.einsum("...dt,...et->...de",
                                (1 - Ms) * X, X.conj()) / denominator_n

        # F x N x T => F x T x N
        X = X.transpose([0, 2, 1])
        # init covariance-matrix on each T-F bins
        # F x T x N x N
        self.Rtf = np.einsum("...a,...b->...ab", X, np.conj(X))

        # init phi_{n,s}
        self.phi_n = np.zeros([self.num_bins, self.num_frames])
        self.phi_s = np.zeros([self.num_bins, self.num_frames])

        # tr(A*B^{-1}) = tr(B^{-1}*A) = tr(B\A)
        for f in range(self.num_bins):
            # N x N, T x N x N => T x N x N
            self.phi_n[f] = np.trace(np.linalg.solve(self.Rn[f], self.Rtf[f]),
                                     axis1=1,
                                     axis2=2).real / self.num_channels
            self.phi_s[f] = np.trace(np.linalg.solve(self.Rs[f], self.Rtf[f]),
                                     axis1=1,
                                     axis2=2).real / self.num_channels

        logger.info(
            "CGMM initialize: {:d} channels, {:d} frames, {:d} frequency bins".
            format(self.num_channels, self.num_frames, self.num_bins))

    def train(self, num_epoches=20):
        """
            Start cgmm training for some epoches
        """
        # likelihoods
        pn = np.ones([self.num_bins, self.num_frames])
        ps = np.ones([self.num_bins, self.num_frames])
        # masks
        mn = np.ones([self.num_bins, self.num_frames])
        ms = np.ones([self.num_bins, self.num_frames])
        # component weights
        # wn = np.ones([self.num_bins, 1]) / 2
        # ws = np.ones([self.num_bins, 1]) / 2
        I = np.eye(self.num_channels, self.num_channels, dtype=np.complex)

        for e in range(num_epoches):

            for f in range(self.num_bins):

                self.Rn[f] += theta * np.trace(
                    self.Rn[f]).real / self.num_channels * I
                self.Rs[f] += theta * np.trace(
                    self.Rs[f]).real / self.num_channels * I

                pn[f] = CgmmLoglikelihoodFaster(self.Rn[f], self.phi_n[f])
                ps[f] = CgmmLoglikelihoodFaster(self.Rs[f], self.phi_s[f])

                # update phi
                self.phi_n[f] = np.trace(
                    np.linalg.solve(self.Rn[f], self.Rtf[f]), axis1=1,
                    axis2=2).real / self.num_channels
                self.phi_s[f] = np.trace(
                    np.linalg.solve(self.Rs[f], self.Rtf[f]), axis1=1,
                    axis2=2).real / self.num_channels

                # update masks
                # mn[f] = wn[f] * pn[f] / (wn[f] * pn[f] + ws[f] * ps[f])
                mn[f] = pn[f] / (pn[f] + ps[f])
                ms[f] = 1 - mn[f]
                # update alpha
                # wn[f] = np.sum(mn[f]) / self.num_frames
                # ws[f] = np.sum(ms[f]) / self.num_frames

                cn = mn[f] / (self.phi_n[f] * np.sum(mn[f]))
                cs = ms[f] / (self.phi_s[f] * np.sum(ms[f]))

                # update R{n,s}
                self.Rn[f] = np.sum(cn[:, None, None] * self.Rtf[f], axis=0)
                self.Rs[f] = np.sum(cs[:, None, None] * self.Rtf[f], axis=0)

            Qn = np.sum(
                mn * np.log(pn + EPSILON)) / (self.num_bins * self.num_frames)
            Qs = np.sum(
                ms * np.log(ps + EPSILON)) / (self.num_bins * self.num_frames)
            logger.info("Epoch {:02d}: Q = {:.4f} + {:.4f} = {:.4f}".format(
                e + 1, Qn, Qs, Qn + Qs))

        return ms


class CacgmDistribution(object):
    """
    Complex Angular Central Gaussian Distribution (K classes, F bins)
    """
    def __init__(self, covar_eigval=None, covar_eigvec=None):
        self.covar_eigval = covar_eigval
        self.covar_eigvec = covar_eigvec

    def update(self, covar, force_hermitian=True):
        """
        Update covariance matrix (K x F x M x M)
        """
        if force_hermitian:
            covar_h = np.einsum("...xy->...yx", covar.conj())
            covar = (covar + covar_h) / 2
        try:
            eig_val, eig_vec = np.linalg.eigh(covar)
        except np.linalg.LinAlgError:
            eig_val, eig_vec = np.linalg.eig(covar)
        # scaled eigen values
        self.covar_eigval = eig_val / np.maximum(
            np.amax(eig_val, axis=-1, keepdims=True),
            EPSILON,
        )
        self.covar_eigvec = eig_vec

    def _check_status(self):
        """
        Check if model is initialized
        """
        for s in [self.covar_eigval, self.covar_eigvec]:
            if s is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} is not initialized")

    def covar(self, inv=False):
        """
        Return B or B^{-1}
        """
        # K x F x M x M
        if not inv:
            return np.einsum("...xy,...y,...zy->...xz", self.covar_eigvec,
                             self.covar_eigval, self.covar_eigvec.conj())
        else:
            return np.einsum("...xy,...y,...zy->...xz", self.covar_eigvec,
                             1 / self.covar_eigval, self.covar_eigvec.conj())

    def log_pdf(self, obs, return_kernel=False):
        """
        Formula:
            A(z, B) = (M - 1)!/(2 * pi^M * det(B)) * 1 / (z^H * B^{-1} * z)^M
            log A = const - log[det(B)] - M * log(z^H * B^{-1} * z)
        Arguments
            obs: normalized mixture observation, F x M x T
        Return:
            logpdf: K x F x T
            zh_B_inv_z: K x F x T, z^H * B^{-1} * z
        """
        self._check_status()
        _, _, M = self.covar_eigval.shape
        # K x F x M x M
        B_inv = self.covar(inv=True)
        # K x F x T
        zh_B_inv_z = np.einsum("...xt,...xy,...yt->...t", obs.conj(), B_inv,
                               obs)
        zh_B_inv_z = np.maximum(np.abs(zh_B_inv_z), EPSILON)
        # K x F x M => K x F
        log_det = np.sum(np.log(self.covar_eigval), axis=-1, keepdims=True)
        log_pdf = -M * np.log(zh_B_inv_z) - log_det
        # K x F x T
        if not return_kernel:
            return log_pdf
        else:
            return log_pdf, zh_B_inv_z


class Cacgmm(object):
    """
    Complex Angular Central Gaussian Mixture Model (CACGMM)
    """
    def __init__(self):
        self.cacgm = CacgmDistribution()
        # K x F
        self.alpha = None

    def update(self, obs, gamma, kernel):
        """
        Update parameters in Cacgmm
        Arguments:
            obs: normalized mixture observation, F x M x T
            gamma: K x F x T
            kernel: K x F x T, z^H * B^{-1} * z
        """
        # K x F
        denominator = np.sum(gamma, -1)
        M, _, _ = obs.shape
        # K x F x M x M
        covar = M * np.einsum("...t,...xt,...yt->...xy", gamma / kernel, obs,
                              obs.conj())
        covar = covar / np.maximum(denominator[..., None, None], EPSILON)
        self.alpha = denominator / obs.shape[-1]
        self.cacgm.update(covar, force_hermitian=True)

    def predict(self, obs, return_Q=False):
        """
        Compute gamma (posterior) using Cacgmm
        Arguments:
            obs: normalized mixture observation, F x M x T
        Return:
            gamma: posterior, K x F x T
        """
        # K x F x T
        log_pdf, kernel = self.cacgm.log_pdf(obs, return_kernel=True)
        Q = None
        if return_Q:
            # K x F x T => F x T
            pdf_tf = np.sum(np.exp(log_pdf) * self.alpha[..., None], 0)
            # each TF-bin
            Q = np.mean(np.log(pdf_tf))
        log_pdf = log_pdf - np.amax(log_pdf, 0, keepdims=True)
        # K x F x T
        pdf = np.exp(log_pdf)
        # K x F x T
        nominator = pdf * self.alpha[..., None]
        denominator = np.sum(nominator, 0, keepdims=True)
        gamma = nominator / np.maximum(denominator, EPSILON)
        if return_Q:
            return gamma, kernel, Q
        else:
            return gamma, kernel


class CacgmmTrainer(object):
    """
    Cacgmm Trainer
    """
    def __init__(self, obs, num_classes, gamma=None, cacgmm=None):
        """
        Arguments:
            obs: mixture observation, M x F x T
            num_classes: number of the cluster
            gamma: initial gamma, K x F x T
        """
        self.random_init = cacgmm is None
        # F x M x T
        self.obs = self._norm_obs(obs)

        if self.random_init:
            self.cacgmm = Cacgmm()
            logger.info(f"Random initialized, num_classes = {num_classes}")
            if gamma is None:
                F, _, T = self.obs.shape
                gamma = np.random.uniform(size=[num_classes, F, T])
                self.gamma = gamma / np.sum(gamma, 0, keepdims=True)
            else:
                self.gamma = gamma
            self.K = np.ones([num_classes, F, T])
        else:
            with open(cacgmm, "r") as pkl:
                self.cacgmm = pickle.load(pkl)
            logger.info(f"Resume cacgmm model from {cacgmm}")
            self.gamma, self.K = self.cacgmm.predict(obs)

    def train(self, num_epoches=20):
        """
        Train in EM progress
        """
        for e in range(num_epoches):
            self.cacgmm.update(self.obs, self.gamma, self.K)
            self.gamma, self.K, Q = self.cacgmm.predict(self.obs,
                                                        return_Q=True)
            logger.info(f"Epoch {e + 1:2d}: Q = {Q:.4f}")
        return self.gamma

    def _norm_obs(self, obs):
        """
        Normalize observations
        """
        # obs (M x F x T) => z (F x M x T)
        norm = np.maximum(EPSILON,
                          np.linalg.norm(obs, ord=2, axis=0, keepdims=True))
        obs = obs / norm
        obs = np.einsum("mft->fmt", obs)
        return obs