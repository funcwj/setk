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
from libs.utils import get_logger, EPSILON

logger = get_logger(__name__)

# profile log
# 424.9570s/369.6120s/333.9310s/275.4740s/12.4840s
# faster = True
theta = 1e-4


# for time t on frequency f
def CgmmLoglikelihood(C, R, phi):
    # for x' * R^{-1} / phi * x using  x' * (R \ x) / phi
    # kernel = np.inner(np.conj(x), np.linalg.solve(R, x) / phi).real
    kernel = np.trace(np.linalg.solve(R, C)).real / phi
    # det(pi * phi * R) = (pi * phi)^N * det(R)
    det = np.linalg.det(R).real * ((np.pi * phi)**R.shape[0])
    # TODO: check why sometimes negative here
    if det <= 0:
        raise RuntimeError("Determinant of R is negative, R = {}".format(
            phi * R))
    loglike = np.exp(-kernel) / (det + EPSILON)
    # check NAN
    if np.isnan(loglike) or np.isinf(loglike):
        raise RuntimeError(
            "Encounter loglike = {} in CgmmLoglikehood(), det = {:.4f}, kernel = {:.4f}"
            .format(loglike, det, kernel))
    return loglike


# for all frames on frequency f
def CgmmLoglikelihoodV(C, R, phi):
    kernels = np.trace(np.linalg.solve(R, C), axis1=1, axis2=2).real / phi
    # make sure hermitian
    R = (R + np.transpose(np.conj(R))) / 2
    det = np.linalg.det(R).real
    # TODO: check why sometimes negative here
    if det <= 0:
        raise RuntimeError(
            "Determinant of R is negative, det = {:.4f}, R = {}".format(
                det, R))
    loglikes = np.exp(-kernels) / (det * ((np.pi * phi)**R.shape[0]) + EPSILON)
    if np.any(np.isnan(loglikes)) or np.any(np.isinf(loglikes)):
        raise RuntimeError(
            "Encounter loglike = NAN/INF in CgmmLoglikehoodV() on some time axis"
        )
    return loglikes


class CgmmTrainer(object):
    """
        CgmmTrainer for two targets: speech & noise
    """

    def __init__(self, X):
        # N x F x T => F x T x N
        X = X.transpose([1, 2, 0])

        # self.X = X
        self.num_bins, self.num_frames, self.num_channels = X.shape

        # init R{n,s}
        self.Rs = np.array([
            # (N x T) * (T x N)
            np.dot(X[f].T, np.conj(X[f])) / self.num_frames
            for f in range(self.num_bins)
        ])
        self.Rn = np.array([
            np.eye(self.num_channels, self.num_channels, dtype=np.complex)
            for f in range(self.num_bins)
        ])

        # init covariance-matrix on each T-F bins
        self.Rtf = np.zeros([
            self.num_bins, self.num_frames, self.num_channels,
            self.num_channels
        ],
                            dtype=np.complex)
        # to optimize
        self.Rtf = np.einsum("...a,...b->...ab", X, np.conj(X))
        # for f in range(self.num_bins):
        #     for t in range(self.num_frames):
        #         self.Rtf[f, t] = np.outer(X[f, t], np.conj(X[f, t]))

        # init phi_{n,s}
        self.phi_n = np.zeros([self.num_bins, self.num_frames])
        self.phi_s = np.zeros([self.num_bins, self.num_frames])

        # tr(A*B^{-1}) = tr(B^{-1}*A) = tr(B\A)
        # for f in range(self.num_bins):
        #     for t in range(self.num_frames):
        #         self.phi_n[f, t] = np.trace(
        #             np.linalg.solve(self.Rn[f],
        #                             self.Rtf[f, t])).real / self.num_channels
        #         self.phi_s[f, t] = np.trace(
        #             np.linalg.solve(self.Rs[f],
        #                             self.Rtf[f, t])).real / self.num_channels
        # to optimized
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

        for e in range(num_epoches):

            for f in range(self.num_bins):

                self.Rn[f] += theta * np.trace(
                    self.Rn[f]).real / self.num_channels * np.eye(
                        self.num_channels, self.num_channels, dtype=np.complex)
                self.Rs[f] += theta * np.trace(
                    self.Rs[f]).real / self.num_channels * np.eye(
                        self.num_channels, self.num_channels, dtype=np.complex)

                # for t in range(self.num_frames):
                #     # obs vector
                #     pn[f, t] = CgmmLoglikelihood(self.Rtf[f, t], self.Rn[f],
                #                                  self.phi_n[f, t]) + EPSILON
                #     ps[f, t] = CgmmLoglikelihood(self.Rtf[f, t], self.Rs[f],
                #                                  self.phi_s[f, t]) + EPSILON

                #    # update phi
                #    self.phi_n[f, t] = np.trace(
                #        np.linalg.solve(
                #            self.Rn[f],
                #            self.Rtf[f, t])).real / self.num_channels
                #    self.phi_s[f, t] = np.trace(
                #        np.linalg.solve(
                #            self.Rs[f],
                #            self.Rtf[f, t])).real / self.num_channels

                pn[f] = CgmmLoglikelihoodV(self.Rtf[f], self.Rn[f],
                                           self.phi_n[f])
                ps[f] = CgmmLoglikelihoodV(self.Rtf[f], self.Rs[f],
                                           self.phi_s[f])

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
