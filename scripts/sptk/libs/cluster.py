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

from scipy.optimize import linear_sum_assignment
from .utils import get_logger, EPSILON

logger = get_logger(__name__)

supported_plan = {
    257: [[20, 70, 170], [2, 90, 190], [2, 50, 150], [2, 110, 210],
          [2, 30, 130], [2, 130, 230], [2, 0, 110], [2, 150, 257]],
    513: [[20, 100, 200], [2, 120, 220], [2, 80, 180], [2, 140, 240],
          [2, 60, 160], [2, 160, 260], [2, 40, 140], [2, 180, 280],
          [2, 0, 120], [2, 200, 300], [2, 220, 320], [2, 240, 340],
          [2, 260, 360], [2, 280, 380], [2, 300, 400], [2, 320, 420],
          [2, 340, 440], [2, 360, 460], [2, 380, 480], [2, 400, 513]]
}


def norm_observation(mat, axis=-1, eps=EPSILON):
    """
    L2 normalization for observation vectors
    """
    denorm = np.linalg.norm(mat, ord=2, axis=axis, keepdims=True)
    denorm = np.maximum(denorm, eps)
    return mat / denorm


def permu_aligner(masks, transpose=False):
    """
    Solve permutation problems for clustering based mask algorithm
    Reference: "https://raw.githubusercontent.com/fgnt/pb_bss/master/pb_bss/permutation_alignment.py"
    args:
        masks: K x T x F
    return:
        aligned_masks: K x T x F
    """
    if masks.ndim != 3:
        raise RuntimeError("Expect 3D TF-masks, K x T x F or K x F x T")
    if transpose:
        masks = np.transpose(masks, (0, 2, 1))
    K, _, F = masks.shape
    # normalized masks, for cos distance, K x T x F
    feature = norm_observation(masks, axis=1)
    # K x F
    mapping = np.stack([np.ones(F, dtype=np.int) * k for k in range(K)])

    if F not in supported_plan:
        raise ValueError(f"Unsupported num_bins: {F}")
    for itr, beg, end in supported_plan[F]:
        for _ in range(itr):
            # normalized centroid, K x T
            centroid = np.mean(feature[..., beg:end], axis=-1)
            centroid = norm_observation(centroid, axis=-1)
            go_on = False
            for f in range(beg, end):
                # K x K
                score = feature[..., f] @ centroid.T
                # derive permutation based on score matrix
                index, permu = linear_sum_assignment(score, maximize=True)
                # not ordered
                if np.sum(permu != index) != 0:
                    feature[..., f] = feature[permu, :, f]
                    mapping[..., f] = mapping[permu, f]
                    go_on = True
            if not go_on:
                break
    # K x T x F
    permu_masks = np.zeros_like(masks)
    for f in range(F):
        permu_masks[..., f] = masks[mapping[..., f], :, f]
    return permu_masks


class Distribution(object):
    """
    Basic distribution class
    """
    def __init__(self, covar_eigval=None, covar_eigvec=None):
        self.parameters = {
            "covar_eigval": covar_eigval,
            "covar_eigvec": covar_eigvec
        }

    def check_status(self):
        """
        Check if distribution is initialized
        """
        for key, value in self.parameters.items():
            if value is None:
                raise RuntimeError(
                    f"{key} is not initialized in the distribution")

    def update_covar(self, covar, force_hermitian=True):
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
        eig_val = eig_val / np.maximum(
            np.amax(eig_val, axis=-1, keepdims=True),
            EPSILON,
        )
        self.parameters["covar_eigval"] = np.maximum(eig_val, EPSILON)
        self.parameters["covar_eigvec"] = eig_vec

    def covar(self, inv=False):
        """
        Return R or R^{-1}
        """
        # K x F x M x M
        if not inv:
            return np.einsum("...xy,...y,...zy->...xz",
                             self.parameters["covar_eigvec"],
                             self.parameters["covar_eigval"],
                             self.parameters["covar_eigvec"].conj())
        else:
            return np.einsum("...xy,...y,...zy->...xz",
                             self.parameters["covar_eigvec"],
                             1 / self.parameters["covar_eigval"],
                             self.parameters["covar_eigvec"].conj())

    def log_pdf(self, obs, **kwargs):
        """
        Return value of log-pdf
        """
        raise NotImplementedError

    def update_parameters(self, *args, **kwargs):
        """
        Update distribution parameters
        """
        raise NotImplementedError


class CgDistribution(Distribution):
    """
    Complex Gaussian Distribution (K classes, F bins)
    """
    def __init__(self, phi=None, covar_eigval=None, covar_eigvec=None):
        super(CgDistribution, self).__init__(covar_eigval, covar_eigvec)
        self.parameters["phi"] = phi

    def update_parameters(self, obs, covar, force_hermitian=True):
        """
        Update covar & phi
        """
        _, M, _ = obs.shape
        self.update_covar(covar, force_hermitian=force_hermitian)
        R_inv = self.covar(inv=True)
        phi = np.einsum("...xt,...xy,...yt->...t", obs.conj(), R_inv, obs)
        phi = np.maximum(np.abs(phi), EPSILON)
        self.parameters["phi"] = phi / M

    def log_pdf(self, obs, return_kernel=False):
        """
        Formula:
            N(y, R) = e^(-y^H * R^{-1} * y) / det(pi*R)
        since:
            phi = trace(y * y^H * R^{-1}) / M = y^H * R^{-1} * y / M
        then:
            N(y, phi*R) = e^(-y^H * R^{-1} * y / phi) / det(pi*R*phi)
                        = e^{-M} / (det(R) * (phi * pi)^M)
        log N = const - log[det(R)] - M * log(phi)

        Arguments
            obs: normalized mixture observation, F x M x T
        Return:
            logpdf: K x F x T
        """
        self.check_status()
        _, M, _ = obs.shape
        log_det = np.sum(np.log(self.parameters["covar_eigval"]),
                         axis=-1,
                         keepdims=True)
        log_pdf = -M * np.log(self.parameters["phi"]) - log_det
        # K x F x T
        return log_pdf


class CacgDistribution(Distribution):
    """
    Complex Angular Central Gaussian Distribution (K classes, F bins)
    """
    def __init__(self, covar_eigval=None, covar_eigvec=None):
        super(CacgDistribution, self).__init__(covar_eigval, covar_eigvec)

    def update_parameters(self, covar, force_hermitian=True):
        """
        Update covar only
        """
        self.update_covar(covar, force_hermitian=force_hermitian)

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
        self.check_status()
        _, M, _ = obs.shape
        # K x F x M x M
        B_inv = self.covar(inv=True)
        # K x F x T
        zh_B_inv_z = np.einsum("...xt,...xy,...yt->...t", obs.conj(), B_inv,
                               obs)
        zh_B_inv_z = np.maximum(np.abs(zh_B_inv_z), EPSILON)
        # K x F x M => K x F x 1
        log_det = np.sum(np.log(self.parameters["covar_eigval"]),
                         axis=-1,
                         keepdims=True)
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
    def __init__(self, mdl, alpha=None):
        self.cacg = CacgDistribution() if mdl is None else mdl
        # K x F
        self.alpha = alpha

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
        _, M, T = obs.shape
        # K x F x M x M
        covar = M * np.einsum("...t,...xt,...yt->...xy", gamma / kernel, obs,
                              obs.conj())
        covar = covar / np.maximum(denominator[..., None, None], EPSILON)
        self.alpha = denominator / T
        self.cacg.update_parameters(covar, force_hermitian=True)

    def predict(self, obs, return_Q=False):
        """
        Compute gamma (posterior) using Cacgmm
        Arguments:
            obs: normalized mixture observation, F x M x T
        Return:
            gamma: posterior, K x F x T
        """
        # K x F x T
        log_pdf, kernel = self.cacg.log_pdf(obs, return_kernel=True)
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


class Cgmm(object):
    """
    Complex Gaussian Mixture Model (CGMM)
    """
    def __init__(self, mdl, alpha=None):
        self.cg = CgDistribution() if mdl is None else mdl
        # K x F
        self.alpha = alpha

    def update(self, obs, gamma):
        """
        Update parameters in Cgmm
        Arguments:
            obs: mixture observation, F x M x T
            gamma: K x F x T
        """
        # K x F x 1
        denominator = np.sum(gamma, -1, keepdims=True)
        # K x F x M x M
        R = np.einsum("...t,...xt,...yt->...xy",
                      gamma / self.cg.parameters["phi"], obs, obs.conj())
        R = R / np.maximum(denominator[..., None], EPSILON)
        # update R & phi
        self.cg.update_parameters(obs, R)
        # do not update alpha
        # self.alpha = np.mean(gamma, -1)

    def predict(self, obs, return_Q=False):
        """
        Compute gamma (posterior) using Cgmm
        Arguments:
            obs: mixture observation, F x M x T
        Return:
            gamma: posterior, K x F x T
        """
        # K x F x T
        log_pdf = self.cg.log_pdf(obs)
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
            return gamma, Q
        else:
            return gamma


class CgmmTrainer(object):
    """
    Cgmm Trainer
    """
    def __init__(self, obs, gamma=None, cgmm=None):
        """
        Arguments:
            obs: mixture observation, M x F x T
            gamma: initial gamma, K x F x T
        """
        self.random_init = cgmm is None
        # F x M x T
        self.obs = np.einsum("mft->fmt", obs)
        F, M, T = self.obs.shape
        logger.info(f"CGMM instance: F = {F:d}, T = {T:}, M = {M}")

        if self.random_init:
            cg = CgDistribution()
            if gamma is None:
                Rs = np.einsum("...dt,...et->...de", self.obs,
                               self.obs.conj()) / T
                Rn = np.stack(
                    [np.eye(M, M, dtype=np.complex) for _ in range(F)])
                R = np.stack([Rs, Rn])
            else:
                # 2 x F x T
                gamma = np.stack([gamma, 1 - gamma])
                # 2 x F
                den = np.maximum(np.sum(gamma, axis=-1, keepdims=True),
                                 EPSILON)
                # 2 x F x M x M
                R = np.einsum("...t,...xt,...yt->...xy", gamma, self.obs,
                              self.obs.conj()) / den[..., None]
            cg.update_parameters(self.obs, R)
            self.cgmm = Cgmm(cg, alpha=np.ones([2, F]))
            self.gamma = self.cgmm.predict(self.obs)
        else:
            with open(cgmm, "r") as pkl:
                self.cgmm = pickle.load(pkl)
            logger.info(f"Resume cgmm model from {cgmm}")
            self.gamma = self.cgmm.predict(self.obs)

    def train(self, num_epoches=20):
        """
        Train in EM progress
        """
        for e in range(num_epoches):
            self.cgmm.update(self.obs, self.gamma)
            self.gamma, Q = self.cgmm.predict(self.obs, return_Q=True)
            logger.info(f"Epoch {e + 1:2d}: Q = {Q:.4f}")
        return self.gamma[0]


class CacgmmTrainer(object):
    """
    Cacgmm Trainer
    """
    def __init__(self,
                 obs,
                 num_classes,
                 gamma=None,
                 cacgmm=None,
                 cgmm_init=False):
        """
        Arguments:
            obs: mixture observation, M x F x T
            num_classes: number of the cluster
            gamma: initial gamma, K x F x T
            cgmm_init: init like cgmm papers
        """
        self.random_init = cacgmm is None
        # obs (M x F x T) => z (F x M x T)
        self.obs = np.einsum("mft->fmt", norm_observation(obs, axis=0))

        F, M, T = self.obs.shape
        logger.info(f"CACGMM instance: F = {F:d}, T = {T:}, M = {M}")

        if self.random_init:
            if cgmm_init and num_classes == 2:
                # using init method like cgmm (not well)
                cacg = CacgDistribution()
                # init covar
                covar = np.stack([
                    np.einsum("...xt,...yt->...xy", self.obs, self.obs.conj())
                    / T,
                    np.stack([np.eye(M, M, dtype=obs.dtype) for _ in range(F)])
                ])
                cacg.update_parameters(covar)
                self.cacgmm = Cacgmm(cacg, alpha=np.ones([2, F]) / 2)
                self.gamma, self.K = self.cacgmm.predict(self.obs)
                logger.info("Using cgmm init for num_classes = 2")
            else:
                if gamma is None:
                    self.cacgmm = Cacgmm(None)
                    gamma = np.random.uniform(size=[num_classes, F, T])
                    self.gamma = gamma / np.sum(gamma, 0, keepdims=True)
                    logger.info(
                        f"Random initialized, num_classes = {num_classes}")
                else:
                    self.gamma = gamma
                    logger.info("Using external gamma, " +
                                f"num_classes = {num_classes}")
                self.K = np.ones([num_classes, F, T])
        else:
            with open(cacgmm, "r") as pkl:
                self.cacgmm = pickle.load(pkl)
            logger.info(f"Resume cacgmm model from {cacgmm}")
            self.gamma, self.K = self.cacgmm.predict(self.obs)

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