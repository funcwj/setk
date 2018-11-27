#!/usr/bin/env python

# wujian@2018
"""
Faster CGMM Trainer
Reference:
    Higuchi T, Ito N, Yoshioka T, et al. Robust MVDR beamforming using time-frequency masks
    for online/offline ASR in noise[C]//Acoustics, Speech and Signal Processing (ICASSP),
    2016 IEEE International Conference on. IEEE, 2016: 5210-5214.
"""

import numpy as np
from .utils import get_logger, EPSILON

logger = get_logger(__name__)

# profile log
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
            "Determinant of R is negative, det = {:.4f}, R = {}".format(
                det, R))
    loglikes = np.exp(-N) / (det * ((np.pi * phi)**N) + EPSILON)
    if np.any(np.isnan(loglikes)) or np.any(np.isinf(loglikes)):
        raise RuntimeError(
            "Encounter loglike = NAN/INF in CgmmLoglikehoodV() on some time axis"
        )
    return loglikes


class CgmmTrainer(object):
    """
        CgmmTrainer for two targets: speech & noise
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
            denominator_s = np.maximum(
                np.sum(Ms, axis=-1, keepdims=True), 1e-6)
            self.Rs = np.einsum("...dt,...et->...de", Ms * X,
                                X.conj()) / denominator_s
            denominator_n = np.maximum(
                np.sum(1 - Ms, axis=-1, keepdims=True), 1e-6)
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
            self.phi_n[f] = np.trace(
                np.linalg.solve(self.Rn[f], self.Rtf[f]), axis1=1,
                axis2=2).real / self.num_channels
            self.phi_s[f] = np.trace(
                np.linalg.solve(self.Rs[f], self.Rtf[f]), axis1=1,
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
