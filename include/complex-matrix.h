// complex-matrix.h
// wujian@2018
// reference from kaldi-matrix.h
// 

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
    
    // this = this * alpha
    void Scale(Real alpha_r, Real alpha_i);
    
    // ajust cblas mm results into normal form
    // (a, bi) => (b+a)/2 (b-a)/2
    void Adjust();

    // this = this'
    void Conjugate();
    
    // this = this^T
    void Transpose();
    
    // this = this^H
    void Hermite();

    // this == this^H
    bool IsHermitian(Real cutoff = 1.0e-5);
    
    // init from complex matrix, enable conjugate & transpose
    void CopyFromMat(const CMatrixBase<Real> &M, 
                     MatrixTransposeType trans = kNoTrans,
                     ConjugateType conj = kNoConj);
    // init from real matrix
    void CopyFromMat(const MatrixBase<Real> &M, 
                     ComplexIndexType index = kReal);

    void CopyFromRealfft(const MatrixBase<Real> &M);

    std::string Info() const;

    // this = this .* A
    void MulElements(const CMatrixBase<Real> &A);

    // this = this ./ A
    void DivElements(const CMatrixBase<Real> &A);
    
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

    void HEig(Vector<Real> *eig_value, CMatrix<Real> *eig_vector);

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
                     MatrixTransposeType trans = kNoTrans,
                     ConjugateType conj = kNoConj);
    
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


template<typename Real>
std::ostream & operator << (std::ostream &os, const CMatrixBase<Real> &cm) {
    if (cm.NumCols() == 0) {
        os << " [ ]\n";
    } else {
        os << " [";
        for (MatrixIndexT i = 0; i < cm.NumRows(); i++) {
            os << "\n  ";
            for (MatrixIndexT j = 0; j < cm.NumCols(); j++)
            os << cm(i, j, kReal) << (cm(i, j, kImag) >=0 ? "+": "") << cm(i, j, kImag) << "i ";
        }
        os << "]\n";
    }
    return os;
}

}

#endif
