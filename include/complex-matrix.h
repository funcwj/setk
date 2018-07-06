// include/complex-matrix.h
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


#ifndef COMPLEX_MATRIX_H_
#define COMPLEX_MATRIX_H_

#include "include/complex-base.h"
#include "include/complex-vector.h"

namespace kaldi {


template<typename Real>
class CMatrixBase {

public:
    friend class CMatrix<Real>;
    friend class SubCMatrix<Real>;

    inline MatrixIndexT NumRows() const { return num_rows_; }

    inline MatrixIndexT NumCols() const { return num_cols_; }

    inline MatrixIndexT Stride() const {  return stride_; }

    inline const Real* Data() const { return data_; }

    inline Real* Data() { return data_; }

    inline const Real* RowData(MatrixIndexT i) const {
        KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_));
        return data_ + i * stride_;
    }

    inline Real* RowData(MatrixIndexT i) {
        KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_));
        return data_ + i * stride_;
    }

    inline Real&  operator() (MatrixIndexT r, MatrixIndexT c, ComplexIndexType kIndex) {
        KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                              static_cast<UnsignedMatrixIndexT>(num_rows_) &&
                              static_cast<UnsignedMatrixIndexT>(c) <
                              static_cast<UnsignedMatrixIndexT>(num_cols_));
        return *(data_ + r * stride_ + (kIndex == kReal ? c * 2: c * 2 + 1));
    }

    inline const Real operator() (MatrixIndexT r, MatrixIndexT c, ComplexIndexType kIndex) const {
        KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                              static_cast<UnsignedMatrixIndexT>(num_rows_) &&
                              static_cast<UnsignedMatrixIndexT>(c) <
                              static_cast<UnsignedMatrixIndexT>(num_cols_));
        return *(data_ + r * stride_ + (kIndex == kReal ? c * 2: c * 2 + 1));
    }

    inline SubCMatrix<Real> Range(const MatrixIndexT row_offset,
                                  const MatrixIndexT num_rows,
                                  const MatrixIndexT col_offset,
                                  const MatrixIndexT num_cols) const {
        return SubCMatrix<Real>(*this, row_offset, num_rows, col_offset, num_cols);
    }

    inline SubCMatrix<Real> RowRange(const MatrixIndexT row_offset,
                                    const MatrixIndexT num_rows) const {
        return SubCMatrix<Real>(*this, row_offset, num_rows, 0, num_cols_);
    }

    inline SubCMatrix<Real> ColRange(const MatrixIndexT col_offset,
                                     const MatrixIndexT num_cols) const {
        return SubCMatrix<Real>(*this, 0, num_rows_, col_offset, num_cols);
    }

    inline const SubCVector<Real> Row(MatrixIndexT i) const {
        KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) < 
                     static_cast<UnsignedMatrixIndexT>(num_rows_));
        return SubCVector<Real>(data_ + (i * stride_), num_cols_);
    }

    inline SubCVector<Real> Row(MatrixIndexT i) {
        KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                     static_cast<UnsignedMatrixIndexT>(num_rows_));
        return SubCVector<Real>(data_ + (i * stride_), num_cols_);
    }

    void SetZero();

    void SetRandn();
    
    void SetUnit();

    void Add(Real alpha_r, Real alpha_i);
    
    // this = this * alpha
    void Scale(Real alpha_r, Real alpha_i);
    
    // ajust cblas mm results into normal form
    // (a, bi) => (b+a)/2 (b-a)/2
    void AdjustOut();

    // (a, bi) => (a-b) (a+b)
    void AdjustIn();

    // this = this'
    void Conjugate();
    
    // this = this^T
    void Transpose();
    
    // this = this^H
    void Hermite();

    // this = this.Real/Imag
    void Part(MatrixBase<Real> *P, ComplexIndexType index);

    // this = |this|
    void Abs(MatrixBase<Real> *P);

    // this = exp(Ej)
    void Exp(const MatrixBase<Real> &E);

    // this == this^H
    bool IsHermitian(Real cutoff = 1.0e-5);

    bool IsHermitianPosDef();
    
    // init from complex matrix, enable conjugate & transpose
    void CopyFromMat(const CMatrixBase<Real> &M, 
                     MatrixTransposeType trans = kNoTrans);

    // init from real matrix
    void CopyFromMat(const MatrixBase<Real> &M, 
                     ComplexIndexType index = kReal);

    void CopyFromRealfft(const MatrixBase<Real> &M);

    // copy vector into rindex row of matrix
    void CopyRowFromVec(const CVectorBase<Real> &v, const MatrixIndexT rindex);

    // copy vector into cindex col of matrix
    void CopyColFromVec(const CVectorBase<Real> &v, const MatrixIndexT cindex);


    std::string Info() const;

    void AddToDiag(const Real alpha_r, const Real alpha_i);

    // this = this .* A
    void MulElements(const CMatrixBase<Real> &A, 
                     ConjugateType conj = kNoConj,
                     bool mul_abs = false);

    // this = this ./ A
    void DivElements(const CMatrixBase<Real> &A, 
                     ConjugateType conj = kNoConj,
                     bool div_abs = false);
    
    // this = this * beta + alpha * A * B
    void AddMatMat(const Real alpha_r, const Real alpha_i,
                   const CMatrixBase<Real>& A, MatrixTransposeType transA,
                   const CMatrixBase<Real>& B, MatrixTransposeType transB,
                   const Real beta_r, const Real beta_i);

    // this = this + alpha * a * b^{T or H}
    void AddVecVec(const Real alpha_r, const Real alpha_i, 
                   const CVectorBase<Real> &a,
                   const CVectorBase<Real> &b,
                   ConjugateType conj = kNoConj);

    // this = this * alpha + M
    void AddMat(const Real alpha_r, const Real alpha_i, 
                const CMatrixBase<Real> &M,
                MatrixTransposeType trans = kNoTrans);

    // this = this^{-1}
    void Invert();

    // For Hermite matrix, eigen values are all real.
    // And eig_value is in ascend order.
    // To get same results as MATLAB, call eig_vector.Hermite(）after
    void Hed(VectorBase<Real> *D, CMatrixBase<Real> *V);


    void Hged(CMatrixBase<Real> *B, VectorBase<Real> *D,
              CMatrixBase<Real> *V);

    // Now binary must be true
    void Read(std::istream &in, bool binary);
    
    void Write(std::ostream &out, bool binary) const;

protected:


    CMatrixBase(Real *data, MatrixIndexT cols, MatrixIndexT rows, MatrixIndexT stride) :
        data_(data), num_cols_(cols), num_rows_(rows), stride_(stride) {
        KALDI_ASSERT_IS_FLOATING_TYPE(Real);
    }

    CMatrixBase(): data_(NULL) {
        KALDI_ASSERT_IS_FLOATING_TYPE(Real);
    }

    inline Real*  Data_workaround() const {
        return data_;
    }

    ~CMatrixBase() { }

    Real*   data_;

    MatrixIndexT    num_cols_;
    MatrixIndexT    num_rows_;
    MatrixIndexT    stride_;

private:
    KALDI_DISALLOW_COPY_AND_ASSIGN(CMatrixBase);

};


template<typename Real>
class CMatrix: public CMatrixBase<Real> {
public:
    CMatrix() { };

    CMatrix(const MatrixIndexT r, const MatrixIndexT c,
            MatrixResizeType resize_type = kSetZero,
            MatrixStrideType stride_type = kDefaultStride):
            CMatrixBase<Real>() { Resize(r, c, resize_type, stride_type); }
    
    // copy constructor, from complex matrix
    explicit CMatrix(const CMatrixBase<Real> &M, 
                     MatrixTransposeType trans = kNoTrans);
    
    CMatrix(const CMatrix<Real> &M);

    CMatrix(const VectorBase<Real> &v);

    // copy constructor, from real matrix
    CMatrix(const MatrixBase<Real> &M, ComplexIndexType index = kReal);
    
    // copy constructor
    CMatrix<Real> &operator = (const CMatrixBase<Real> &other) {
        if (CMatrixBase<Real>::NumRows() != other.NumRows() ||
            CMatrixBase<Real>::NumCols() != other.NumCols())
            Resize(other.NumRows(), other.NumCols(), kUndefined);
        CMatrixBase<Real>::CopyFromMat(other);
        return *this;
    }

    CMatrix<Real> &operator = (const CMatrix<Real> &other) {
        if (CMatrixBase<Real>::NumRows() != other.NumRows() ||
            CMatrixBase<Real>::NumCols() != other.NumCols())
            Resize(other.NumRows(), other.NumCols(), kUndefined);
        CMatrixBase<Real>::CopyFromMat(other);
        return *this;
    }

    void Swap(CMatrix<Real> *other);

    void Transpose();

    void Hermite();

    ~CMatrix() { Destroy(); }

    void Resize(const MatrixIndexT rows, const MatrixIndexT cols,
                MatrixResizeType resize_type = kSetZero,
                MatrixStrideType stride_type = kDefaultStride);

    void Read(std::istream &in, bool binary);

private:
    void Destroy();

    void Init(const MatrixIndexT rows, const MatrixIndexT cols,
              const MatrixStrideType stride_type);
};


template<typename Real>
class SubCMatrix: public CMatrixBase<Real> {
public:

    SubCMatrix(const CMatrixBase<Real>& T, 
               const MatrixIndexT row_offset, const MatrixIndexT rows,
               const MatrixIndexT col_offset, const MatrixIndexT cols); 

    SubCMatrix(Real *data, 
               MatrixIndexT num_rows, MatrixIndexT num_cols,
               MatrixIndexT stride);

    SubCMatrix<Real> (const SubCMatrix &other): CMatrixBase<Real> 
             (other.data_, other.num_cols_, other.num_rows_, other.stride_) {}

    ~SubCMatrix<Real>() {}

private:
    SubCMatrix<Real> &operator = (const SubCMatrix<Real> &other);
};

// This function is only used for debug.
template<typename Real>
std::ostream & operator << (std::ostream &os, const CMatrixBase<Real> &cm) {
    cm.Write(os, false);
    return os;
}

}

#endif
