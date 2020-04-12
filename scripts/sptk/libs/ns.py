#!/usr/bin/env python

# wujian@2020

import numpy as np
import scipy.signal as ss
import scipy.integrate as si


class MCRA(object):
    """
    OM-LSA (Optimally Modified Log-Spectral Amplitude Estimator) with MCRA
    Reference:
        1) Cohen I, Berdugo B. Speech enhancement for non-stationary noise environments[J]. 
           Signal processing, 2001, 81(11): 2403-2418.
    """
    def __init__(self,
                 alpha=0.92,
                 delta=5,
                 beta=0.7,
                 alpha_s=0.9,
                 alpha_d=0.85,
                 alpha_p=0.2,
                 gmin_db=-10,
                 xi_min_db=-18,
                 w_mcra=1,
                 w_local=1,
                 w_global=15,
                 h_mcra="hann",
                 h_local="hann",
                 h_global="hann",
                 q_max=0.95,
                 zeta_min_db=-10,
                 zeta_max_db=-5,
                 zeta_p_max_db=10,
                 zeta_p_min_db=0,
                 L=60,
                 M=128):
        self.delta = delta
        self.alpha = {"s": alpha_s, "d": alpha_d, "p": alpha_p, "t": alpha}
        self.gmin = 10**(gmin_db / 10)
        self.beta = beta
        self.w_m = ss.get_window(h_mcra, w_mcra * 2 + 1)
        self.w_g = ss.get_window(h_global, w_global * 2 + 1)
        self.w_l = ss.get_window(h_local, w_local * 2 + 1)
        self.xi_min = 10**(xi_min_db / 10)
        self.zeta_min = 10**(zeta_min_db / 10)
        self.zeta_max = 10**(zeta_max_db / 10)
        self.zeta_p_min = 10**(zeta_p_min_db / 10)
        self.zeta_p_max = 10**(zeta_p_max_db / 10)
        self.L = L
        self.M = M
        self.q_max = q_max

    def run(self, stft, eps=1e-7):
        """
        Arguments:
            stft: complex STFT, T x F
        Return:
            gain: real array, T x F
        """
        obs_power = np.abs(stft)**2
        T, F = obs_power.shape

        def expint(v):
            return si.quad(lambda t: np.exp(-t) / t, v, np.inf)[0]

        exp_para = np.vectorize(expint)

        gh1 = 1
        p_hat = np.ones(F)
        zeta = np.ones(F)
        zeta_peak = 0
        zeta_frame_pre = 1000
        gamma = obs_power[0]
        lambda_d_hat = obs_power[0]

        g = []
        for t in range(T):
            # >>> eq.32
            var_sf = np.convolve(obs_power[t], self.w_m, mode="same")
            # <<< eq.32

            if t == 0:
                var_s = obs_power[t]
                var_s_min = var_s
                var_s_tmp = var_s
            else:
                # >>> eq.33
                var_s = self.alpha["s"] * var_s + (1 -
                                                   self.alpha["s"]) * var_sf
                # <<< eq.33

            if (t + 1) % self.L == 0:
                # >>> eq.34 & eq.35
                var_s_min = np.minimum(var_s_tmp, var_s)
                var_s_tmp = var_s
                # <<< eq.34 & eq.35
            else:
                # >>> eq.36 & eq.37
                var_s_min = np.minimum(var_s_min, var_s)
                var_s_tmp = np.minimum(var_s_tmp, var_s)
                # <<< eq.36 & eq.37

            # >>> eq.39
            var_sr = var_s / np.maximum(eps, var_s_min)
            sr_ind = var_sr > self.delta
            # <<< eq.39

            # >>> eq.40
            p_hat = self.alpha["p"] * p_hat + (1 - self.alpha["p"]) * sr_ind
            # >>> eq.40

            # >>> eq.31
            alpha_d_hat = self.alpha["d"] + (1 - self.alpha["d"]) * p_hat
            # <<< eq.31

            # >>> eq.30
            lambda_d_hat = alpha_d_hat * lambda_d_hat + (
                1 - alpha_d_hat) * obs_power[t]
            # <<< eq.30

            # >>> eq.18: a priori SNR
            xi_hat = self.alpha["t"] * gh1**2 * gamma + (
                1 - self.alpha["t"]) * np.maximum(gamma - 1, 0)
            xi_hat = np.maximum(xi_hat, self.xi_min)
            # <<< eq.18

            # >>> eq.23
            zeta = self.beta * zeta + (1 - self.beta) * xi_hat
            # <<< eq.23

            # >>> eq.24
            zeta_g = np.convolve(zeta, self.w_g, mode="same")
            zeta_l = np.convolve(zeta, self.w_l, mode="same")
            # <<< eq.24

            # >>> eq.25
            var_p_g = np.zeros(F)
            pg_idx = np.logical_and(zeta_g > self.zeta_min,
                                    zeta_g < self.zeta_max)
            var_p_g[pg_idx] = np.log10(
                zeta_g[pg_idx] / self.zeta_min) / np.log10(
                    self.zeta_max / self.zeta_min)
            pg_idx = zeta_g >= self.zeta_max
            var_p_g[pg_idx] = 1
            # <<< eq.25

            # >>> eq.25
            var_p_l = np.zeros(F)
            pl_idx = np.logical_and(zeta_l > self.zeta_min,
                                    zeta_l < self.zeta_max)
            var_p_l[pl_idx] = np.log10(
                zeta_l[pl_idx] / self.zeta_min) / np.log10(
                    self.zeta_max / self.zeta_min)
            pl_idx = zeta_l >= self.zeta_max
            var_p_l[pl_idx] = 1
            # <<< eq.25

            # >>> eq.26
            zeta_frame_cur = np.mean(zeta[:self.M // 2 + 1])
            # <<< eq.26

            # >>> eq.27
            if zeta_frame_cur > self.zeta_min:
                if zeta_frame_cur > zeta_frame_pre:
                    zeta_peak = min(max(zeta_frame_cur, self.zeta_p_min),
                                    self.zeta_p_max)
                    p_frame = 1
                elif zeta_frame_cur <= self.zeta_min * zeta_peak:
                    p_frame = 0
                elif zeta_frame_cur >= self.zeta_max * zeta_peak:
                    p_frame = 1
                else:
                    p_frame = np.log10(zeta_frame_cur /
                                       (self.zeta_min * zeta_peak))
                    p_frame = p_frame / np.log10(self.zeta_max / self.zeta_min)
            else:
                p_frame = 0
            # <<< eq.27

            # >>> eq.28
            q_hat = np.minimum(self.q_max, 1 - var_p_l * p_frame * var_p_g)
            # <<< eq.28

            zeta_frame_pre = zeta_frame_cur

            # >>> eq.10
            # a posteriori SNR
            gamma = obs_power[t] / np.maximum(lambda_d_hat, eps)
            gamma = np.maximum(gamma, eps)
            v = gamma * xi_hat / (1 + xi_hat)
            # <<< eq.10

            # >>> eq.9
            p_inv = 1 + q_hat * (1 + xi_hat) * np.exp(-v) / (1 + q_hat)
            p = 1 / p_inv
            # <<< eq.10

            # >>> eq.15
            gh1 = xi_hat * np.exp(0.5 * exp_para(v)) / (1 + xi_hat)
            # <<< eq.15

            # >>> eq.16
            gt = gh1**p * self.gmin**(1 - p)
            g.append(gt)
            # <<< eq.16
        return np.stack(g)


class iMCRA(object):
    """
    OM-LSA (Optimally Modified Log-Spectral Amplitude Estimator) with iMCRA
    Reference:
        1) Cohen I. Noise spectrum estimation in adverse environments: Improved minima controlled 
           recursive averaging[J]. IEEE Transactions on speech and audio processing, 2003, 11(5): 
           466-475.
    """
    def __init__(self,
                 alpha=0.92,
                 alpha_s=0.9,
                 alpha_d=0.85,
                 b_min=1.66,
                 gamma0=4.6,
                 gamma1=3,
                 zeta0=1.67,
                 xi_min_db=-18,
                 gmin_db=-10,
                 w_mcra=1,
                 h_mcra="hann",
                 lambda_d_scaler=1,
                 V=15,
                 U=8):
        self.alpha = {"s": alpha_s, "d": alpha_d, "t": alpha}
        self.lambda_d_scaler = lambda_d_scaler
        self.gamma0, self.gamma1 = gamma0, gamma1
        self.zeta0 = zeta0
        self.b_min = 1 / b_min
        self.xi_min = 10**(xi_min_db / 10)
        self.gain_min = 10**(gmin_db / 10)
        self.w_m = ss.get_window(h_mcra, w_mcra * 2 + 1)
        self.V = V
        self.U = U

    def _derive_var_v(self, power, lambda_d, gh1, eps=1e-7):
        # >>> eq.3 in ref{1}: posteriori SNR
        gamma = power / np.maximum(lambda_d, eps)
        # <<< eq.3 in ref{1}

        gain = gh1**2 * gamma
        # >>> eq.32 in ref{1} : a priori SNR
        xi_hat = self.alpha["t"] * gain + (1 - self.alpha["t"]) * np.maximum(
            gamma - 1, 0)
        xi_hat = np.maximum(xi_hat, self.xi_min)
        # <<< eq.32 in ref{1}

        # >>> eq.10 in ref{2}
        v = gamma * xi_hat / (1 + xi_hat)
        # <<< eq.10 in ref{2}
        return v, xi_hat

    def run(self, stft, eps=1e-7):
        """
        Arguments:
            stft: complex STFT, T x F
        Return:
            gain: real array, T x F
        """
        obs_power = np.abs(stft)**2
        T, F = obs_power.shape
        lambda_d_hat = obs_power[0]
        gh1 = 1
        xi_hat = np.ones(F) * self.alpha["t"]
        v = lambda_d_hat * xi_hat / (1 + xi_hat)

        def expint(v):
            return si.quad(lambda t: np.exp(-t) / t, v, np.inf)[0]

        exp_para = np.vectorize(expint)

        s_min_sw_hat = []
        s_min_sw = []
        g = []
        for t in range(T):

            # >>> eq.14 in ref{1}
            var_sf = np.convolve(obs_power[t], self.w_m, mode="same")
            # <<< eq.14 in ref{1}

            if t == 0:
                var_s = var_sf
                var_s_hat = var_sf
                var_s_min = var_sf
                var_s_min_sw = var_sf
            else:
                # >>> eq.15 in ref{1}
                var_s = self.alpha["s"] * var_s + (1 -
                                                   self.alpha["s"]) * var_sf
                # <<< eq.15 in ref{1}
                var_s_min = np.minimum(var_s_min, var_s)
                var_s_min_sw = np.minimum(var_s_min_sw, var_s)

            # >>> eq.21 in ref{1}
            gamma_min = obs_power[t] * self.b_min / np.maximum(var_s_min, eps)
            zeta = var_sf * self.b_min / np.maximum(var_s_min, eps)
            indicator = np.logical_and(gamma_min < self.gamma0,
                                       zeta < self.zeta0)
            # <<< eq.21 in ref{1}

            # >>> eq.26 in ref{1}
            ind_conv = np.convolve(indicator, self.w_m, mode="same")
            ind_nz_idx = (ind_conv > 0)
            obs_conv = np.convolve(obs_power[t] * indicator,
                                   self.w_m,
                                   mode="same")
            var_sf_hat = var_s_hat.copy()
            var_sf_hat[
                ind_nz_idx] = obs_conv[ind_nz_idx] / ind_conv[ind_nz_idx]
            # <<< eq.26 in ref{1}

            if t == 0:
                var_s_min_hat = var_s
                var_s_min_sw_hat = var_sf
            else:
                # <<< eq.27 in ref{1}
                var_s_hat = self.alpha["s"] * var_s_hat + (
                    1 - self.alpha["s"]) * var_sf_hat
                # >>> eq.27 in ref{1}
                var_s_min_hat = np.minimum(var_s_min_hat, var_s_hat)
                var_s_min_sw_hat = np.minimum(var_s_min_sw_hat, var_s_hat)

            # >>> eq.28 in ref{1}
            gamma_min_hat = obs_power[t] * self.b_min / np.maximum(
                var_s_min_hat, eps)
            zeta_hat = var_s * self.b_min / np.maximum(var_s_min_hat, eps)
            # <<< eq.28 in ref{1}

            # >>> eq.29 in ref{1}
            qhat_idx_c1 = gamma_min_hat < self.gamma1
            qhat_idx_c2 = gamma_min_hat > 1
            # 1 < gamma_min_hat < self.gamma1
            qhat_idx_c3 = np.logical_and(qhat_idx_c2, qhat_idx_c1)

            q_hat = np.zeros(F)
            qhat_idx = np.logical_and(qhat_idx_c3, zeta_hat < self.zeta0)
            # (0, 1)
            q_hat[qhat_idx] = (self.gamma1 -
                               gamma_min_hat[qhat_idx]) / (self.gamma1 - 1)
            # <<< eq.29 in ref{1}

            # >>> eq.7 in ref{1}
            p_hat = np.zeros(F)
            p_hat_den = 1 + q_hat[qhat_idx] * (1 + xi_hat[qhat_idx]) / (
                1 - q_hat[qhat_idx]) * np.exp(-v[qhat_idx])
            # (0, 1)
            p_hat[qhat_idx] = 1 / p_hat_den
            phat_idx = np.logical_and(gamma_min_hat >= self.gamma1,
                                      zeta_hat >= self.zeta0)
            p_hat[phat_idx] = 1
            # <<< eq.7 in ref{1}

            # >>> eq.11 in ref{1}
            alpha_d_hat = self.alpha["d"] + (1 - self.alpha["d"]) * p_hat
            # <<< eq.11 in ref{1}

            # >>> eq.10 in ref{1}
            lambda_d_hat = alpha_d_hat * lambda_d_hat + (
                1 - alpha_d_hat) * obs_power[t]
            # <<< eq.10 in ref{1}
            lambda_d = lambda_d_hat * self.lambda_d_scaler

            s_min_sw.append(var_s_min_sw)
            s_min_sw_hat.append(var_s_min_sw_hat)

            if (t + 1) % self.V == 0:
                # U x F
                u_s_min_sw = np.stack(s_min_sw[-self.U:])
                u_s_min_sw_hat = np.stack(s_min_sw_hat[-self.U:])
                var_s_min = np.min(u_s_min_sw, 0)
                var_s_min_hat = np.min(u_s_min_sw_hat, 0)
                var_s_min_sw = var_s
                var_s_min_sw_hat = var_s_hat

            v, xi_hat = self._derive_var_v(obs_power[t],
                                           lambda_d,
                                           gh1,
                                           eps=eps)

            # >>> eq.15 in ref{1}
            gh1 = xi_hat / (1 + xi_hat) * np.exp(0.5 * exp_para(v))
            # <<< eq.15 in ref{1}

            # >>> eq.16 in ref{1}
            gt = gh1**p_hat * self.gain_min**(1 - p_hat)
            g.append(gt)
            # <<< eq.16 in ref{1}

        return np.stack(g)