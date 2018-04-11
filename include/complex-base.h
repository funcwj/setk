// complex-base.h
// wujian@2018

#ifndef COMPLEX_MATH_BASE_H
#define COMPLEX_MATH_BASE_H


#include <complex>
#include "matrix/cblas-wrappers.h"
#include "matrix/matrix-common.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"

#include "include/cblas-cpl-wrappers.h"

namespace kaldi {

typedef enum {
    kReal,
    kImag
} ComplexIndexType;

typedef enum {
    kConj,
    kNoConj,
} ConjugateType;


template<typename Real>
struct Complex {
    Real real, imag;
    Complex(Real r, Real i): real(r), imag(i) {}
    Complex() {}
};

template<typename Real> class CVectorBase;
template<typename Real> class CVector;
template<typename Real> class SubCVector;

template<typename Real> class CMatrixBase;
template<typename Real> class CMatrix;
template<typename Real> class SubCMatrix;


template<typename Real> 
inline void ComplexDiv(const Real &a_re, const Real &a_im, Real *b_re, Real *b_im) {
    Real d = a_re * a_re + a_im * a_im;
    Real tmp_re = (*b_re * a_re) + (*b_im * a_im);
    *b_im = (*b_re * a_im - *b_im * a_re) / d;
    *b_re = tmp_re / d;
}

}

#endif
