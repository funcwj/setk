#!/usr/bin/env python
# coding=utf-8
# wujian@2018
"""
Copy MATLAB's .mat into (C)Matrix binary format
"""

import argparse
import numpy as np
import scipy.io as sio

from libs.iobase import write_common_mat, write_token, write_int32, write_binary_symbol


def write_complex_mat(fd, cmat):
    assert cmat.dtype == np.complex64 or cmat.dtype == np.complex128
    mat_type = 'FCM' if cmat.dtype == np.complex64 else 'DCM'
    write_token(fd, mat_type)
    num_rows, num_cols = cmat.shape
    write_int32(fd, num_rows)
    write_int32(fd, num_cols)
    fd.write(cmat.tobytes())


def run(args):
    mdict = sio.loadmat(args.mmat)
    mmat = mdict[args.key]
    assert mmat.dtype in [np.float32, np.float64, np.complex64, np.complex128]
    print('Detect input dtype={}'.format(mmat.dtype))
    if args.transpose:
        mmat = np.transpose(mmat)

    if mmat.dtype == np.float64 or mmat.dtype == np.float32:
        # from float32 to float64
        if mmat.dtype == np.float32 and args.double:
            mmat = np.array(mmat, dtype=np.float64)
        if mmat.dtype == np.float64 and args.float:
            mmat = np.array(mmat, dtype=np.float32)
        with open(args.kmat, "wb") as f:
            write_binary_symbol(f)
            write_common_mat(f, mmat)
    else:
        if mmat.dtype == np.complex64 and args.double:
            mmat = np.array(mmat, dtype=np.complex128)
        if mmat.dtype == np.complex128 and args.float:
            mmat = np.array(mmat, dtype=np.complex64)
        with open(args.kmat, "wb") as f:
            write_binary_symbol(f)
            write_complex_mat(f, mmat)
    print("Copy from {} to {} in {}".format(args.mmat, args.kmat, mmat.dtype))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Command to copy MATLAB's (complex) "
        "matrix into (C)Matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mmat",
                        type=str,
                        help="Original matrix in matlab's format")
    parser.add_argument("kmat",
                        type=str,
                        help="Object matrix in kaldi's format")
    parser.add_argument("key",
                        type=str,
                        help="Key values to index matrix in mmat")
    parser.add_argument("--double",
                        action="store_true",
                        help="If true, then write matrix "
                        "in float64/complex128")
    parser.add_argument("--float",
                        action="store_true",
                        help="If true, then write matrix in float32/complex64")
    parser.add_argument("--transpose",
                        action="store_true",
                        help="If true, write transpose of "
                        "original matrix instead")
    args = parser.parse_args()
    run(args)
