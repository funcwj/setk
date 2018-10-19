// include/cblas-complex.h
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

#ifndef CBLAS_COMPLEX_WRAPPERS_H
#define CBLAS_COMPLEX_WRAPPERS_H

#include "cblas.h"

namespace kaldi {

#if defined(HAVE_OPENBLAS)
#define KaldiComplexFloat lapack_complex_float
#define KaldiComplexDouble lapack_complex_double
#elif defined(HAVE_CLAPACK)
#define KaldiComplexFloat complex
#define KaldiComplexDouble doublecomplex
#else
#error \
    "You must add definitions in CMakeLists.txt(-DHAVE_OPENBLAS or -DHAVE_CLAPACK)"
#endif

inline void cblas_CZscal(const int N, const void *alpha, float *data,
                         const int inc) {
  cblas_cscal(N, alpha, data, inc);
}

inline void cblas_CZscal(const int N, const void *alpha, double *data,
                         const int inc) {
  cblas_zscal(N, alpha, data, inc);
}

inline void cblas_CZaxpy(const int N, const void *alpha, const float *X,
                         const int incX, float *Y, const int incY) {
  cblas_caxpy(N, alpha, X, incX, Y, incY);
}

inline void cblas_CZaxpy(const int N, const void *alpha, const double *X,
                         const int incX, double *Y, const int incY) {
  cblas_zaxpy(N, alpha, X, incX, Y, incY);
}

inline void cblas_CZdot(const int N, const float *X, const int incX,
                        const float *Y, const int incY, bool conj, void *dot) {
  if (conj)
    cblas_cdotc_sub(N, X, incX, Y, incY, dot);
  else
    cblas_cdotu_sub(N, X, incX, Y, incY, dot);
}

inline void cblas_CZdot(const int N, const double *X, const int incX,
                        const double *Y, const int incY, bool conj, void *dot) {
  if (conj)
    cblas_zdotc_sub(N, X, incX, Y, incY, dot);
  else
    cblas_zdotu_sub(N, X, incX, Y, incY, dot);
}

inline void cblas_CZgemm(const void *alpha, MatrixTransposeType transA,
                         const float *Adata, MatrixIndexT a_num_rows,
                         MatrixIndexT a_num_cols, MatrixIndexT a_stride,
                         MatrixTransposeType transB, const float *Bdata,
                         MatrixIndexT b_stride, const void *beta, float *Mdata,
                         MatrixIndexT num_rows, MatrixIndexT num_cols,
                         MatrixIndexT stride) {
  cblas_cgemm(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(transA),
              static_cast<CBLAS_TRANSPOSE>(transB), num_rows, num_cols,
              transA == kNoTrans ? a_num_cols : a_num_rows, alpha, Adata,
              a_stride >> 1, Bdata, b_stride >> 1, beta, Mdata, stride >> 1);
}

inline void cblas_CZgemm(const void *alpha, MatrixTransposeType transA,
                         const double *Adata, MatrixIndexT a_num_rows,
                         MatrixIndexT a_num_cols, MatrixIndexT a_stride,
                         MatrixTransposeType transB, const double *Bdata,
                         MatrixIndexT b_stride, const void *beta, double *Mdata,
                         MatrixIndexT num_rows, MatrixIndexT num_cols,
                         MatrixIndexT stride) {
  cblas_zgemm(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(transA),
              static_cast<CBLAS_TRANSPOSE>(transB), num_rows, num_cols,
              transA == kNoTrans ? a_num_cols : a_num_rows, alpha, Adata,
              a_stride >> 1, Bdata, b_stride >> 1, beta, Mdata, stride >> 1);
}

inline void cblas_CZger(MatrixIndexT num_rows, MatrixIndexT num_cols,
                        void *alpha, const float *xdata, MatrixIndexT incX,
                        const float *ydata, MatrixIndexT incY, float *Mdata,
                        MatrixIndexT stride, bool conj) {
  if (conj)
    cblas_cgerc(CblasRowMajor, num_rows, num_cols, alpha, xdata, 1, ydata, 1,
                Mdata, stride >> 1);
  else
    cblas_cgeru(CblasRowMajor, num_rows, num_cols, alpha, xdata, 1, ydata, 1,
                Mdata, stride >> 1);
}

inline void cblas_CZger(MatrixIndexT num_rows, MatrixIndexT num_cols,
                        void *alpha, const double *xdata, MatrixIndexT incX,
                        const double *ydata, MatrixIndexT incY, double *Mdata,
                        MatrixIndexT stride, bool conj) {
  if (conj)
    cblas_zgerc(CblasRowMajor, num_rows, num_cols, alpha, xdata, 1, ydata, 1,
                Mdata, stride >> 1);
  else
    cblas_zgeru(CblasRowMajor, num_rows, num_cols, alpha, xdata, 1, ydata, 1,
                Mdata, stride >> 1);
}

inline void cblas_CZgemv(MatrixTransposeType trans, MatrixIndexT num_rows,
                         MatrixIndexT num_cols, void *alpha, const float *Mdata,
                         MatrixIndexT stride, const float *xdata,
                         MatrixIndexT incX, void *beta, float *ydata,
                         MatrixIndexT incY) {
  cblas_cgemv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), num_rows,
              num_cols, alpha, Mdata, stride >> 1, xdata, incX, beta, ydata,
              incY);
}

inline void cblas_CZgemv(MatrixTransposeType trans, MatrixIndexT num_rows,
                         MatrixIndexT num_cols, void *alpha,
                         const double *Mdata, MatrixIndexT stride,
                         const double *xdata, MatrixIndexT incX, void *beta,
                         double *ydata, MatrixIndexT incY) {
  cblas_zgemv(CblasRowMajor, static_cast<CBLAS_TRANSPOSE>(trans), num_rows,
              num_cols, alpha, Mdata, stride >> 1, xdata, incX, beta, ydata,
              incY);
}

// function prototype: compute eigen vector & value for hermite matrix
// int cheev_(char *jobz, char *uplo, integer *n, complex *a,
//            integer *lda, real *w, complex *work, integer *lwork, real *rwork,
//            integer *info);

inline void clapack_CZheev(KaldiBlasInt *num_rows, void *V,
                           KaldiBlasInt *stride, float *D, void *work,
                           KaldiBlasInt *lwork, float *rwork,
                           KaldiBlasInt *info) {
  cheev_(const_cast<char *>("V"), const_cast<char *>("U"), num_rows,
         reinterpret_cast<KaldiComplexFloat *>(V), stride, D,
         reinterpret_cast<KaldiComplexFloat *>(work), lwork, rwork, info);
}

inline void clapack_CZheev(KaldiBlasInt *num_rows, void *V,
                           KaldiBlasInt *stride, double *D, void *work,
                           KaldiBlasInt *lwork, double *rwork,
                           KaldiBlasInt *info) {
  zheev_(const_cast<char *>("V"), const_cast<char *>("U"), num_rows,
         reinterpret_cast<KaldiComplexDouble *>(V), stride, D,
         reinterpret_cast<KaldiComplexDouble *>(work), lwork, rwork, info);
}

// function prototype: compute generalized eigen vector & value for hermite
// matrix
// int chegv_(integer *itype, char *jobz, char *uplo, integer *n,
//            complex *a, integer *lda, complex *b, integer *ldb, real *w,
//            complex *work, integer *lwork, real *rwork, integer *info);

inline void clapack_CZhegv(KaldiBlasInt *itype, KaldiBlasInt *num_rows, void *A,
                           KaldiBlasInt *stride_a, void *B,
                           KaldiBlasInt *stride_b, float *D, void *work,
                           KaldiBlasInt *lwork, float *rwork,
                           KaldiBlasInt *info) {
  chegv_(itype, const_cast<char *>("V"), const_cast<char *>("U"), num_rows,
         reinterpret_cast<KaldiComplexFloat *>(A), stride_a,
         reinterpret_cast<KaldiComplexFloat *>(B), stride_b, D,
         reinterpret_cast<KaldiComplexFloat *>(work), lwork, rwork, info);
}

inline void clapack_CZhegv(KaldiBlasInt *itype, KaldiBlasInt *num_rows, void *A,
                           KaldiBlasInt *stride_a, void *B,
                           KaldiBlasInt *stride_b, double *D, void *work,
                           KaldiBlasInt *lwork, double *rwork,
                           KaldiBlasInt *info) {
  zhegv_(itype, const_cast<char *>("V"), const_cast<char *>("U"), num_rows,
         reinterpret_cast<KaldiComplexDouble *>(A), stride_a,
         reinterpret_cast<KaldiComplexDouble *>(B), stride_b, D,
         reinterpret_cast<KaldiComplexDouble *>(work), lwork, rwork, info);
}

inline void clapack_CZgetrf(KaldiBlasInt *num_rows, KaldiBlasInt *num_cols,
                            float *Mdata, KaldiBlasInt *stride,
                            KaldiBlasInt *pivot, KaldiBlasInt *result) {
  cgetrf_(num_rows, num_cols, reinterpret_cast<KaldiComplexFloat *>(Mdata),
          stride, pivot, result);
}

inline void clapack_CZgetrf(KaldiBlasInt *num_rows, KaldiBlasInt *num_cols,
                            double *Mdata, KaldiBlasInt *stride,
                            KaldiBlasInt *pivot, KaldiBlasInt *result) {
  zgetrf_(num_rows, num_cols, reinterpret_cast<KaldiComplexDouble *>(Mdata),
          stride, pivot, result);
}

// int cgetri_(integer *n, complex *a, integer *lda, integer *
//             ipiv, complex *work, integer *lwork, integer *info);
inline void clapack_CZgetri(KaldiBlasInt *num_rows, float *Mdata,
                            KaldiBlasInt *stride, KaldiBlasInt *pivot,
                            float *work, KaldiBlasInt *lwork,
                            KaldiBlasInt *result) {
  cgetri_(num_rows, reinterpret_cast<KaldiComplexFloat *>(Mdata), stride, pivot,
          reinterpret_cast<KaldiComplexFloat *>(work), lwork, result);
}

inline void clapack_CZgetri(KaldiBlasInt *num_rows, double *Mdata,
                            KaldiBlasInt *stride, KaldiBlasInt *pivot,
                            double *work, KaldiBlasInt *lwork,
                            KaldiBlasInt *result) {
  zgetri_(num_rows, reinterpret_cast<KaldiComplexDouble *>(Mdata), stride,
          pivot, reinterpret_cast<KaldiComplexDouble *>(work), lwork, result);
}

}  // namespace kaldi

#endif
