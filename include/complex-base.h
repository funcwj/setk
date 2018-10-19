// include/complex-base.h
// wujian@2018

// Copyright 2018 Jian Wu

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef COMPLEX_BASE_H_
#define COMPLEX_BASE_H_

#include <complex>
#include "matrix/cblas-wrappers.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/kaldi-vector.h"
#include "matrix/matrix-common.h"

#include "include/cblas-cpl-wrappers.h"

namespace kaldi {

typedef enum { kReal, kImag } ComplexIndexType;

typedef enum {
  kConj,
  kNoConj,
} ConjugateType;

//
// typedef enum {
//     kNoTrans        = 111,  // CblasNoTrans
//     kTrans          = 112,  // CblasTrans
//     kConjTrans      = 113,  // CblasConjTrans
//     kConjNoTrans    = 114   // CblasConjNoTrans
// } CMatrixTransposeType;

template <typename Real>
struct Complex {
  Real real, imag;
  Complex(Real r, Real i) : real(r), imag(i) {}
  Complex() {}
};

template <typename Real>
class CVectorBase;
template <typename Real>
class CVector;
template <typename Real>
class SubCVector;

template <typename Real>
class CMatrixBase;
template <typename Real>
class CMatrix;
template <typename Real>
class SubCMatrix;

template <typename Real>
inline void ComplexDiv(const Real &a_re, const Real &a_im, Real *b_re,
                       Real *b_im) {
  Real d = a_re * a_re + a_im * a_im;
  Real tmp_re = (*b_re * a_re) + (*b_im * a_im);
  *b_im = (*b_re * a_im - *b_im * a_re) / d;
  *b_re = tmp_re / d;
}
}

#endif
