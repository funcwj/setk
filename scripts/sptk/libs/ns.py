#!/usr/bin/env python

# wujian@2020

import numpy as np
import scipy.signal as ss
import scipy.integrate as si


class OMLSA(object):
    """
    OM-LSA (Optimally Modified Log-Spectral Amplitude) with iMCRA
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
        self.gain_min = self.xi_min**0.5
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
        lambda_d = obs_power[0]
        lambda_d_hat = obs_power[0]
        gh1 = 1

        def expint(v):
            return si.quad(lambda t: np.exp(-t) / t, v, np.inf)[0]

        exp_para = np.vectorize(expint)

        s_min_sw_hat = []
        s_min_sw = []
        g = []
        for t in range(T):

            v, xi_hat = self._derive_var_v(obs_power[t],
                                           lambda_d,
                                           gh1,
                                           eps=eps)

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