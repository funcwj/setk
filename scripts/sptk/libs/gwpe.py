#!/usr/bin/env python
# wujian@2018
"""
GWPE Algorithm From fgnt's Implementation, I borrowed from https://github.com/fgnt/nara_wpe

Reference:
    Yoshioka T, Nakatani T. Generalization of multi-channel linear prediction 
                            methods for blind MIMO impulse response shortening
                            [J]. IEEE Transactions on Audio, Speech and 
                            Language Processing, 2012, 20(10): 2707-2720.
"""

import functools
import operator

import numpy as np

__all__ = ["wpe"]


def _segment_axis(
        x,
        length,
        shift,
        axis=-1,
        end='cut',  # in ['pad', 'cut', None]
        pad_mode='constant',
        pad_value=0):
    """Generate a new array that chops the given array along the given axis
     into overlapping frames.
    Args:
        x: The array to segment
        length: The length of each frame
        shift: The number of array elements by which to step forward
        axis: The axis to operate on; if None, act on the flattened array
        end: What to do with the last frame, if the array is not evenly
                divisible into pieces. Options are:
                * 'cut'   Simply discard the extra values
                * None    No end treatment. Only works when fits perfectly.
                * 'pad'   Pad with a constant value
        pad_mode:
        pad_value: The value to use for end='pad'
    """
    axis = axis % x.ndim
    elements = x.shape[axis]

    if shift <= 0:
        raise ValueError('Can not shift forward by less than 1 element.')

    if end == 'pad':
        npad = np.zeros([x.ndim, 2], dtype=np.int)
        pad_fn = functools.partial(
            np.pad, pad_width=npad, mode=pad_mode, constant_values=pad_value)
        if elements < length:
            npad[axis, 1] = length - elements
            x = pad_fn(x)
        elif not shift == 1 and not (elements + shift - length) % shift == 0:
            npad[axis, 1] = shift - ((elements + shift - length) % shift)
            x = pad_fn(x)
    elif end is None:
        assert (elements + shift - length) % shift == 0, \
            '{} = elements({}) + shift({}) - length({})) % shift({})' \
            ''.format((elements + shift - length) % shift,
                      elements, shift, length, shift)
    elif end == 'cut':
        pass
    else:
        raise ValueError(end)

    shape = list(x.shape)
    del shape[axis]
    shape.insert(axis, (elements + shift - length) // shift)
    shape.insert(axis + 1, length)

    strides = list(x.strides)
    strides.insert(axis, shift * strides[axis])

    return np.lib.stride_tricks.as_strided(x, strides=strides, shape=shape)


def _build_y_tilde(Y, taps, delay):
    """
    Note: The returned y_tilde consumes a similar amount of memory as Y, because
        of tricks with strides. Usually the memory consumprion is K times
        smaller than the memory consumprion of a contignous array
    """
    S = Y.shape[:-2]
    D = Y.shape[-2]
    T = Y.shape[-1]

    def pad(x, axis=-1, pad_width=taps + delay - 1):
        npad = np.zeros([x.ndim, 2], dtype=np.int)
        npad[axis, 0] = pad_width
        x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
        return x

    Y_ = pad(Y)
    Y_ = np.moveaxis(Y_, -1, -2)
    Y_ = np.flip(Y_, axis=-1)
    Y_ = np.ascontiguousarray(Y_)
    Y_ = np.flip(Y_, axis=-1)
    Y_ = _segment_axis(Y_, taps, 1, axis=-2)
    Y_ = np.flip(Y_, axis=-2)
    if delay > 0:
        Y_ = Y_[..., :-delay, :, :]
    Y_ = np.reshape(Y_, list(S) + [T, taps * D])
    Y_ = np.moveaxis(Y_, -2, -1)

    return Y_


def _window_mean(x, lr_context, axis=-1):
    """
    Take the mean of x at each index with a left and right context.
    Pseudo code for lr_context == (1, 1):
        y = np.zeros(...)
        for i in range(...):
            if not edge_case(i):
                y[i] = (x[i - 1] + x[i] + x[i + 1]) / 3
            elif i == 0:
                y[i] = (x[i] + x[i + 1]) / 2
            else:
                y[i] = (x[i - 1] + x[i]) / 2
        return y
    """
    if isinstance(lr_context, int):
        lr_context = [lr_context + 1, lr_context]
    else:
        assert len(lr_context) == 2, lr_context
        tmp_l_context, tmp_r_context = lr_context
        lr_context = tmp_l_context + 1, tmp_r_context

    x = np.asarray(x)

    window_length = sum(lr_context)
    if window_length == 0:
        return x

    pad_width = np.zeros((x.ndim, 2), dtype=np.int64)
    pad_width[axis] = lr_context

    first_slice = [slice(None)] * x.ndim
    first_slice[axis] = slice(sum(lr_context), None)
    second_slice = [slice(None)] * x.ndim
    second_slice[axis] = slice(None, -sum(lr_context))

    def foo(x):
        cumsum = np.cumsum(np.pad(x, pad_width, mode='constant'), axis=axis)
        return cumsum[first_slice] - cumsum[second_slice]

    ones_shape = [1] * x.ndim
    ones_shape[axis] = x.shape[axis]

    return foo(x) / foo(np.ones(ones_shape, np.int64))


def _get_power_inverse(signal, psd_context=0):
    """
    Assumes single frequency bin with shape (D, T).
    """
    power = np.mean(np.abs(signal)**2, axis=-2)

    if np.isposinf(psd_context):
        power = np.broadcast_to(
            np.mean(power, axis=-1, keepdims=True), power.shape)
    elif psd_context > 0:
        assert int(psd_context) == psd_context, psd_context
        psd_context = int(psd_context)
        # import bottleneck as bn
        # Handle the corner case correctly (i.e. sum() / count)
        # Use bottleneck when only left context is requested
        # power = bn.move_mean(power, psd_context*2+1, min_count=1)
        power = _window_mean(power, (psd_context, psd_context))
    elif psd_context == 0:
        pass
    else:
        raise ValueError(psd_context)
    eps = 1e-10 * np.max(power)
    inverse_power = 1 / np.maximum(power, eps)
    return inverse_power


def _get_working_shape(shape):
    "Flattens all but the last two dimension."
    product = functools.reduce(operator.mul, [1] + list(shape[:-2]))
    return [product] + list(shape[-2:])


def _stable_solve(A, B):
    """
    Use np.linalg.solve with fallback to np.linalg.lstsq.
    Equal to np.linalg.lstsq but faster.
    Note: limited currently by A.shape == B.shape
    This function try's np.linalg.solve with independent dimensions,
    when this is not working the function fall back to np.linalg.solve
    for each matrix. If one matrix does not work it fall back to
    np.linalg.lstsq.
    The reason for not using np.linalg.lstsq directly is the execution time.
    Examples:
    A and B have the shape (500, 6, 6), than a loop over lstsq takes
    108 ms and this function 28 ms for the case that one matrix is singular
    else 1 ms.
    """
    assert A.shape[:-2] == B.shape[:-2], (A.shape, B.shape)
    assert A.shape[-1] == B.shape[-2], (A.shape, B.shape)
    try:
        return np.linalg.solve(A, B)
    except np.linalg.linalg.LinAlgError:
        shape_A, shape_B = A.shape, B.shape
        assert shape_A[:-2] == shape_A[:-2]
        working_shape_A = _get_working_shape(shape_A)
        working_shape_B = _get_working_shape(shape_B)
        A = A.reshape(working_shape_A)
        B = B.reshape(working_shape_B)

        C = np.zeros_like(B)
        for i in range(working_shape_A[0]):
            # lstsq is much slower, use it only when necessary
            try:
                C[i] = np.linalg.solve(A[i], B[i])
            except np.linalg.linalg.LinAlgError:
                C[i] = np.linalg.lstsq(A[i], B[i])[0]
        return C.reshape(*shape_B)


def _hermite(x):
    return x.swapaxes(-2, -1).conj()


def _get_correlations(Y, Y_tilde, inverse_power):
    Y_tilde_inverse_power = Y_tilde * inverse_power[..., None, :]
    R = np.matmul(Y_tilde_inverse_power, _hermite(Y_tilde))
    P = np.matmul(Y_tilde_inverse_power, _hermite(Y))
    return R, P


def _get_filter_matrix(Y, Y_tilde, inverse_power):
    R, P = _get_correlations(Y, Y_tilde, inverse_power)
    G = _stable_solve(R, P)
    return G


def _perform_filter_operation(Y, Y_tilde, filter_matrix):
    X = Y - np.matmul(_hermite(filter_matrix), Y_tilde)
    return X


def wpe(Y, taps=10, delay=3, iters=3, psd_context=0, statistics_mode='full'):
    """
    Modular wpe version.
    """
    X = Y
    Y_tilde = _build_y_tilde(Y, taps, delay)

    if statistics_mode == 'full':
        s = Ellipsis
    elif statistics_mode == 'valid':
        s = (Ellipsis, slice(delay + taps - 1, None))
    else:
        raise ValueError(statistics_mode)

    for _ in range(iters):
        inverse_power = _get_power_inverse(X, psd_context=psd_context)
        G = _get_filter_matrix(
            Y=Y[s], Y_tilde=Y_tilde[s], inverse_power=inverse_power[s])
        X = _perform_filter_operation(Y=Y, Y_tilde=Y_tilde, filter_matrix=G)
    return X

