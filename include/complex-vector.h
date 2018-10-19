// include/complex-vector.h
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

#ifndef COMPLEX_VECTOR_H_
#define COMPLEX_VECTOR_H_

#include "include/complex-base.h"

namespace kaldi {

template <typename Real>
class CVectorBase {
 public:
  void SetZero();

  void Set(Real f);

  void SetRandn();

  inline MatrixIndexT Dim() const { return dim_; }

  inline Real *Data() { return data_; }

  inline const Real *Data() const { return data_; }

  inline Real operator()(MatrixIndexT i, ComplexIndexType kIndex) const {
    KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(i * 2) <
                          static_cast<UnsignedMatrixIndexT>(dim_));
    return *(data_ + (kIndex == kReal ? i * 2 : i * 2 + 1));
  }

  inline Real &operator()(MatrixIndexT i, ComplexIndexType kIndex) {
    KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(i * 2) <
                          static_cast<UnsignedMatrixIndexT>(dim_));
    return *(data_ + (kIndex == kReal ? i * 2 : i * 2 + 1));
  }

  SubCVector<Real> Range(const MatrixIndexT offset, const MatrixIndexT dim) {
    return SubCVector<Real>(*this, offset, dim);
  }

  const SubCVector<Real> Range(const MatrixIndexT offset,
                               const MatrixIndexT dim) const {
    return SubCVector<Real>(*this, offset, dim);
  }

  std::complex<Real> Sum() const;

  std::string Info() const;
  // ajust cblas mm results into normal form
  // (a, bi) => (b+a)/2 (b-a)/2
  void AdjustIn();

  // (a, bi) => (a-b) (a+b)
  void AdjustOut();

  // this += c
  void Add(Real cr, Real ci);

  // this = this + alpha * v
  void AddVec(Real alpha_r, Real alpha_i, CVectorBase<Real> &v);

  // this = beta * this + alpha * M * v
  void AddMatVec(const Real alpha_r, const Real alpha_i,
                 const CMatrixBase<Real> &M, const MatrixTransposeType trans,
                 const CVectorBase<Real> &v, const Real beta_r,
                 const Real beta_i);

  // this = this .* v
  void MulElements(const CVectorBase<Real> &v, ConjugateType conj = kNoConj,
                   bool mul_abs = false);

  // this = this ./ v
  void DivElements(const CVectorBase<Real> &v, ConjugateType conj = kNoConj,
                   bool div_abs = false);

  // this = this * alpha
  void Scale(const Real alpha_r, const Real alpha_i);

  // this = this'
  void Conjugate();

  void CopyFromVec(const CVectorBase<Real> &v, ConjugateType conj = kNoConj);

  void CopyFromVec(const VectorBase<Real> &v, ComplexIndexType kIndex);

  void CopyFromRealfft(const VectorBase<Real> &v);

  void Part(VectorBase<Real> *p, ComplexIndexType index);

  void Abs(VectorBase<Real> *p);

  void Exp(const VectorBase<Real> &e);

 protected:
  ~CVectorBase() {}

  explicit CVectorBase() : data_(NULL), dim_(0) {
    KALDI_ASSERT_IS_FLOATING_TYPE(Real);
  }

  Real *data_;
  MatrixIndexT dim_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(CVectorBase);
};

template <typename Real>
class CVector : public CVectorBase<Real> {
 public:
  CVector() : CVectorBase<Real>() {}

  explicit CVector(const MatrixIndexT s,
                   MatrixResizeType resize_type = kSetZero)
      : CVectorBase<Real>() {
    Resize(s, resize_type);
  }

  // keep them default zero
  CVector(const CVectorBase<Real> &v) : CVectorBase<Real>() {
    Resize(v.Dim(), kSetZero);
    this->CopyFromVec(v);
  }

  CVector(const VectorBase<Real> &v, ComplexIndexType kIndex = kReal)
      : CVectorBase<Real>() {
    Resize(v.Dim(), kSetZero);
    this->CopyFromVec(v, kIndex);
  }

  CVector<Real> &operator=(const CVector<Real> &other) {
    Resize(other.Dim(), kSetZero);
    this->CopyFromVec(other);
    return *this;
  }

  CVector(const CVector<Real> &v) : CVectorBase<Real>() {
    Resize(v.Dim(), kSetZero);
    this->CopyFromVec(v);
  }

  void Swap(CVector<Real> *other);

  ~CVector() { Destroy(); }

  void Resize(MatrixIndexT length, MatrixResizeType resize_type = kSetZero);

 private:
  void Init(const MatrixIndexT dim);

  void Destroy();
};

template <typename Real>
class SubCVector : public CVectorBase<Real> {
 public:
  SubCVector(const CVectorBase<Real> &t, const MatrixIndexT offset,
             const MatrixIndexT dim)
      : CVectorBase<Real>() {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(offset) +
                     static_cast<UnsignedMatrixIndexT>(dim) <=
                 static_cast<UnsignedMatrixIndexT>(t.Dim()));
    CVectorBase<Real>::data_ = const_cast<Real *>(t.Data() + offset * 2);
    CVectorBase<Real>::dim_ = dim;
  }

  SubCVector(const SubCVector &other) : CVectorBase<Real>() {
    CVectorBase<Real>::data_ = other.data_;
    CVectorBase<Real>::dim_ = other.dim_;
  }

  SubCVector(Real *data, MatrixIndexT size) : CVectorBase<Real>() {
    CVectorBase<Real>::data_ = data;
    CVectorBase<Real>::dim_ = size;
  }

  SubCVector(const CMatrixBase<Real> &matrix, MatrixIndexT row) {
    CVectorBase<Real>::data_ = const_cast<Real *>(matrix.RowData(row));
    CVectorBase<Real>::dim_ = matrix.NumCols();
  }

  ~SubCVector() {}

 private:
  SubCVector &operator=(const SubCVector &other) {}
};

template <typename Real>
std::complex<Real> VecVec(const CVectorBase<Real> &v1,
                          const CVectorBase<Real> &v2,
                          ConjugateType conj = kNoConj) {
  MatrixIndexT dim = v1.Dim();
  KALDI_ASSERT(dim == v2.Dim());
  Complex<Real> dot;
  cblas_CZdot(dim, v1.Data(), 1, v2.Data(), 1, conj == kConj ? true : false,
              &dot);
  return std::complex<Real>(dot.real, dot.imag);
}

// I do not implement Write/Read function cause I refused to write them into
// disk.
// This function is only used for debug.
template <typename Real>
std::ostream &operator<<(std::ostream &os, const CVectorBase<Real> &cv) {
  os << " [ ";
  for (MatrixIndexT i = 0; i < cv.Dim(); i++)
    os << cv(i, kReal) << (cv(i, kImag) >= 0 ? "+" : "") << cv(i, kImag)
       << "i ";
  os << "]\n";
  return os;
}
}

#endif
