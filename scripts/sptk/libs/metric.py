# wujian@2018

"""
SI-SNR(scale-invariant SNR/SDR) measure of speech separation
"""

import numpy as np

from itertools import permutations

def si_snr(x, s):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    def vec_l2normp(x):
        return (np.linalg.norm(x, 2)**2)

    # zero mean
    x_zm = x - np.mean(x)
    s_zm = s - np.mean(s)
    t = np.inner(x_zm, s_zm) * s_zm / vec_l2normp(s_zm)
    return 10 * np.log10(vec_l2normp(t) / vec_l2normp(x_zm - t))


def permute_si_snr(xlist, slist):
    """
    Compute SI-SNR between N pairs
    Arguments:
        x: list[vector], enhanced/separated signal
        s: list[vector], reference signal(ground truth)
    """

    def si_snr_avg(xlist, slist):
        return sum([si_snr(x, s) for x, s in zip(xlist, slist)]) / len(xlist)

    N = len(xlist)
    if N != len(slist):
        raise RuntimeError(
            "size do not match between xlist and slist: {:d} vs {:d}".format(
                N, len(slist)))
    si_snrs = []
    for order in permutations(range(N)):
        si_snrs.append(si_snr_avg(xlist, [slist[n] for n in order]))
    return max(si_snrs)
