// complex-matrix.cc
// wujian@2018

#include "include/complex-matrix.h"

namespace kaldi {
    
// Implement for CMatrixBase

template<typename Real>
void CMatrixBase<Real>::SetZero() {
    if (num_cols_ * 2 == stride_)
        memset(data_, 0, sizeof(Real) * num_rows_ * num_cols_ * 2);
    else
        for (MatrixIndexT row = 0; row < num_rows_; row++)
            memset(data_ + row * stride_, 0, sizeof(Real) * num_cols_ * 2);
}


template<typename Real>
void CMatrixBase<Real>::SetRandn() {
    kaldi::RandomState rstate;
    for (MatrixIndexT row = 0; row < num_rows_; row++) {
        Real *row_data = this->RowData(row);        
        for (MatrixIndexT col = 0; col < num_cols_ * 2; col += 2) {
            kaldi::RandGauss2(row_data + col, row_data + col + 1, &rstate);
        }
    }
}

template<typename Real>
void CMatrixBase<Real>::SetUnit() {
    SetZero();
    for (MatrixIndexT row = 0; row < std::min(num_rows_, num_cols_); row++)
        (*this)(row, row, kReal) = 1.0;
}

template<typename Real>
void CMatrixBase<Real>::Adjust() {
    for (MatrixIndexT i = 0; i < num_rows_; i++)
        for (MatrixIndexT j = 0; j < num_cols_; j++) {
            (*this)(i, j, kReal) += (*this)(i, j, kImag);
            (*this)(i, j, kReal) /= 2;
            (*this)(i, j, kImag) -= (*this)(i, j, kReal);
        }
}

template<typename Real>
void CMatrixBase<Real>::Conjugate() {
    for (MatrixIndexT i = 0; i < num_rows_; i++) {
        for (MatrixIndexT j = 0; j < num_cols_; j++)
            (*this)(i, j, kImag) *= (-1.0);
    }
}

template<typename Real>
void CMatrixBase<Real>::Transpose() {
    KALDI_ASSERT(num_cols_ == num_rows_);
    for (MatrixIndexT i = 0; i < num_rows_; i++) {
        for (MatrixIndexT j = 0; j < i; j++) {
            Real &ar = (*this)(i, j, kReal), &ai = (*this)(i, j, kImag),
                 &br = (*this)(j, i, kReal), &bi = (*this)(j, i, kImag);
            std::swap(ar, br);
            std::swap(ai, bi);
        }
    }
}

template<typename Real>
void CMatrixBase<Real>::Hermite() {
    Transpose();
    Conjugate();
}

template<typename Real>
void CMatrixBase<Real>::MulElements(const CMatrixBase<Real> &A) {
    KALDI_ASSERT(num_cols_ == A.NumCols() && num_rows_ == A.NumRows());
    for (MatrixIndexT i = 0; i < num_cols_; i++) {
        for (MatrixIndexT j = 0; j < num_rows_; j++)
            ComplexMul(A(i, j, kReal), A(i, j, kImag), &(*this)(i, j, kReal), &(*this)(i, j, kImag));
    } 
}

template<typename Real>
void CMatrixBase<Real>::DivElements(const CMatrixBase<Real> &A) {
    KALDI_ASSERT(num_cols_ == A.NumCols() && num_rows_ == A.NumRows());
    Real denomintor = 0;
    for (MatrixIndexT i = 0; i < num_cols_; i++) {
        for (MatrixIndexT j = 0; j < num_rows_; j++) {
            ComplexDiv(A(i, j, kReal), A(i, j, kImag), &(*this)(i, j, kReal), &(*this)(i, j, kImag));
        }
    } 
}

template<typename Real>
void CMatrixBase<Real>::Scale(Real alpha_r, Real alpha_i) {
    if (alpha_r == 1.0 && alpha_i == 0.0)
        return;
    if (num_rows_ == 0)
        return;
    Complex<Real> alpha(alpha_r, alpha_i);
    if (num_cols_ * 2 == stride_) {
        cblas_CZscal(static_cast<size_t>(num_cols_) * static_cast<size_t>(num_rows_), 
                    &alpha, data_, 1); 
    } else {
        for (MatrixIndexT i = 0; i < num_rows_; i++)
            cblas_CZscal(num_cols_, &alpha, data_ + stride_ * i, 1);
    }
}


template<typename Real>
std::string CMatrixBase<Real>::Info() const {
    std::ostringstream ostr;
    ostr << "Size: " << num_rows_ << " x " << num_cols_ 
         << " --stride = " << stride_ << ", --addr = " 
         << data_ << std::endl;
    return ostr.str();
}

template<typename Real>
void CMatrixBase<Real>::CopyFromMat(const MatrixBase<Real> &M, ComplexIndexType index) {
    KALDI_ASSERT(static_cast<const void*>(data_) != static_cast<const void*>(M.Data()));
    KALDI_ASSERT(num_cols_ == M.NumCols() && num_rows_ == M.NumRows());
    for (MatrixIndexT i = 0; i < num_rows_; i++)
        for (MatrixIndexT j = 0; j < num_cols_; j++)
            (*this)(i, j, index) = M(i, j);
}

template<typename Real>
void CMatrixBase<Real>::CopyFromMat(const CMatrixBase<Real> &M,
                                   MatrixTransposeType Trans, 
                                   ConjugateType conj) {
    if (static_cast<const void*>(M.Data()) == static_cast<const void*>(this->Data())) {
        KALDI_ASSERT(Trans == kNoTrans && M.NumRows() == NumRows() &&
                    M.NumCols() == NumCols() && M.Stride() == Stride());
        return;
    }
    if (Trans == kNoTrans) {
        KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == M.NumCols());
        for (MatrixIndexT i = 0; i < num_rows_; i++) {
            (*this).Row(i).CopyFromVec(M.Row(i));
            if (conj == kConj)
                (*this).Row(i).Conjugate();
        }
    } else {
        KALDI_ASSERT(num_cols_ == M.NumRows() && num_rows_ == M.NumCols());
        for (MatrixIndexT i = 0; i < num_rows_; i++)
            for (MatrixIndexT j = 0; j < num_cols_; j++) {
                (*this)(i, j, kReal) = M(j, i, kReal);
                (*this)(i, j, kImag) = (conj == kConj ? -M(j, i, kImag): M(j, i, kImag));
        }
    }
}

template<typename Real>
void CMatrixBase<Real>::CopyFromRealfft(const MatrixBase<Real> &M) {
    KALDI_ASSERT(num_rows_ == M.NumRows() && (num_cols_ - 1) * 2 == M.NumCols());
    for (MatrixIndexT i = 0; i < num_rows_; i++)
        this->Row(i).CopyFromRealfft(M.Row(i)); 
}

template<typename Real>
void CMatrixBase<Real>::AddMatMat(const Real alpha_r, const Real alpha_i,
                                  const CMatrixBase<Real>& A, MatrixTransposeType transA,
                                  const CMatrixBase<Real>& B, MatrixTransposeType transB,
                                  const Real beta_r, const Real beta_i) {
    KALDI_ASSERT((transA == kNoTrans && transB == kNoTrans && A.num_cols_ == B.num_rows_ && A.num_rows_ == num_rows_ && B.num_cols_ == num_cols_)
                 || (transA == kTrans && transB == kNoTrans && A.num_rows_ == B.num_rows_ && A.num_cols_ == num_rows_ && B.num_cols_ == num_cols_)
                 || (transA == kNoTrans && transB == kTrans && A.num_cols_ == B.num_cols_ && A.num_rows_ == num_rows_ && B.num_rows_ == num_cols_)
                 || (transA == kTrans && transB == kTrans && A.num_rows_ == B.num_cols_ && A.num_cols_ == num_rows_ && B.num_rows_ == num_cols_));
    KALDI_ASSERT(&A !=  this && &B != this);
    if (num_rows_ == 0) return;
    Complex<Real> alpha(alpha_r, alpha_i), beta(beta_r, beta_i);
    cblas_CZgemm(&alpha, transA, A.data_, A.num_rows_, A.num_cols_, A.stride_,
                transB, B.data_, B.stride_, &beta, data_, num_rows_, num_cols_, stride_);
    Adjust();
}


template<typename Real>
void CMatrixBase<Real>::AddVecVec(const Real alpha_r, const Real alpha_i, 
                                  const CVectorBase<Real> &a,
                                  const CVectorBase<Real> &b,
                                  ConjugateType conj) {
    KALDI_ASSERT(num_rows_ == a.Dim() && num_cols_ == b.Dim());
    if (num_rows_ == 0)
        return;
    Complex<Real> alpha(alpha_r, alpha_i);
    cblas_CZger(a.Dim(), b.Dim(), &alpha, a.Data(), 1, b.Data(), 1, data_, stride_, (conj == kConj ? true: false));
    Adjust();
}


template<typename Real>
void CMatrixBase<Real>::AddMat(const Real alpha_r, const Real alpha_i, 
                               const CMatrixBase<Real> &M,
                               MatrixTransposeType trans) {
    KALDI_ASSERT(&M != this);
    Complex<Real> alpha(alpha_r, alpha_i);
    Real *mdata = M.data_;
    MatrixIndexT mstride = M.stride_;
    if (trans == kNoTrans) {
        KALDI_ASSERT(M.num_cols_ == num_cols_ && M.num_rows_ == num_rows_);
        if (num_rows_ == 0) return;
        for (MatrixIndexT row = 0; row < num_rows_; row++) {
            cblas_CZaxpy(num_cols_, &alpha, mdata + mstride * row, 1, data_ + stride_ * row, 1);
        }
    } else {
        KALDI_ASSERT(M.num_rows_ == num_cols_ && M.num_cols_ == num_rows_);
        if (num_rows_ == 0) return;
        // NOTE: mdata shift by row * 2
        for (MatrixIndexT row = 0; row < num_rows_; row++) {
            cblas_CZaxpy(num_cols_, &alpha, mdata + row * 2, mstride >> 1, data_ + stride_ * row, 1);
        }
    }
    Adjust();
}

// using clapack
template<typename Real>
void CMatrixBase<Real>::Invert() {
    KaldiBlasInt M = num_rows_, N = num_cols_;
    // NOTE: stride_ / 2
    KaldiBlasInt stride = stride_ / 2, result = -1;
    // NOTE: double it
    KaldiBlasInt lwork = std::max<KaldiBlasInt>(1, N * 2);

    Real *work; void *temp;
    if ((work = static_cast<Real*>(KALDI_MEMALIGN(16, 
            sizeof(Real) * lwork, &temp))) == NULL)
        throw std::bad_alloc();

    KaldiBlasInt *pivot = new KaldiBlasInt[num_rows_];
    clapack_CZgetrf(&M, &N, data_, &stride, pivot, &result);

    // free memory
    if (result != 0) {
        delete [] pivot;
        if (result < 0)
            KALDI_ERR << "clapack_CZgetrf(): " << -result << "-th parameter had an illegal value";
        else
            KALDI_ERR << "Cannot invert: matrix is singular";
    }

    clapack_CZgetri(&M, data_, &stride, pivot, work, &lwork, &result);    
    delete [] pivot;
    KALDI_MEMALIGN_FREE(work);

    if (result != 0) {
        if (result < 0)
            KALDI_ERR << "clapack_CZgetrf(): " << -result << "-th parameter had an illegal value";
        else
            KALDI_ERR << "Cannot invert: matrix is singular";
    }
}

template<typename Real>
bool CMatrixBase<Real>::IsHermitian(Real cutoff) {
    if (num_cols_ != num_rows_)
        return false;
    Real bad_sum = 0.0, good_sum = 0.0;
    for (MatrixIndexT i = 0; i < num_rows_; i++) {
        for (MatrixIndexT j = 0; j < i; j++) {
            Real a = (*this)(i, j, kReal), b = (*this)(i, j, kImag),
                 c = (*this)(j, i, kReal), d = (*this)(j, i, kImag);
            good_sum += (std::abs(a + c) * 0.5 + std::abs(b - d) * 0.5);
            bad_sum  += (std::abs(a - c) * 0.5 + std::abs(b + d) * 0.5);
        }
        good_sum += std::abs((*this)(i, i, kReal));
        bad_sum  += std::abs((*this)(i, i, kImag));
    }
    if (bad_sum > cutoff * good_sum)
        return false;
    return true;
}

// inline void clapack_CZheev(KaldiBlasInt *num_rows, void *eig_vecs, KaldiBlasInt *stride, float *eig_value,
//                            void *work, KaldiBlasInt *lwork, float *rwork, KaldiBlasInt *info)
template<typename Real>
void CMatrixBase<Real>::HEig(Vector<Real> *eig_value, CMatrix<Real> *eig_vector) {
    KALDI_ASSERT(IsHermitian());
    eig_vector->Resize(num_rows_, num_cols_);
    eig_value->Resize(num_rows_);
    KaldiBlasInt stride = (eig_vector->Stride() >> 1), num_rows = num_rows_, result; 
    
    eig_vector->CopyFromMat(*this);
    KaldiBlasInt lwork = std::max(1, 2 * num_rows - 1);
    CVector<Real> work(lwork);

    Real *rwork; void *temp;
    if ((rwork = static_cast<Real*>(KALDI_MEMALIGN(16, 
            sizeof(Real) * std::max(1, 3 * num_rows - 2), &temp))) == NULL)
        throw std::bad_alloc();

    clapack_CZheev(&num_rows, eig_vector->Data(), &stride, eig_value->Data(),
                    work.Data(), &lwork, rwork, &result);

    if (result != 0) {
        if (result < 0)
            KALDI_ERR << "clapack_CZheev(): " << -result << "-th parameter had an illegal value";
        else
            KALDI_ERR << "The algorithm failed to converge";
    }
    // NOTE: can add eig_vector->Herimite() to get same results as MATLAB
    // each row is a eigen vector
    // and for A.Eig(&V, D)
    // have A * V = V * D, see test-complex.cc
    // by default, eig_value is in ascend order.
}

// Implement for CMatrix

template<typename Real>
CMatrix<Real>::CMatrix(const CMatrixBase<Real> &M, 
                       MatrixTransposeType trans, 
                       ConjugateType conj) {
    if (trans == kNoTrans)
        Resize(M.num_rows_, M.num_cols_);
    else
        Resize(M.num_cols_, M.num_rows_);
    this->CopyFromMat(M, trans, conj);
}


template<typename Real>
CMatrix<Real>::CMatrix(const CMatrix<Real> &M) {
    Resize(M.num_rows_, M.num_cols_);
    this->CopyFromMat(M);
}

template<typename Real>
CMatrix<Real>::CMatrix(const MatrixBase<Real> &M, 
                       ComplexIndexType index) {
    Resize(M.NumRows(), M.NumCols());
    this->CopyFromMat(M, index);
}


template<typename Real>
CMatrix<Real>::CMatrix(const VectorBase<Real> &v) {
    MatrixIndexT dim = v.Dim();
    KALDI_ASSERT(dim != 0);
    Resize(dim, dim);
    for (MatrixIndexT i = 0; i < dim; i++) {
        (*this)(i, i, kReal) = v(i);
    }
}

template<typename Real>
void CMatrix<Real>::Destroy() {
    if (NULL != CMatrixBase<Real>::data_)
        KALDI_MEMALIGN_FREE( CMatrixBase<Real>::data_);
    CMatrixBase<Real>::data_ = NULL;
    CMatrixBase<Real>::num_rows_ = CMatrixBase<Real>::num_cols_ = CMatrixBase<Real>::stride_ = 0;
}


template<typename Real>
void CMatrix<Real>::Swap(CMatrix<Real> *other) {
    std::swap(this->data_, other->data_);
    std::swap(this->num_cols_, other->num_cols_);
    std::swap(this->num_rows_, other->num_rows_);
    std::swap(this->stride_, other->stride_);
}

template<typename Real>
void CMatrix<Real>::Transpose() {
    if (this->num_rows_ != this->num_cols_) {
        CMatrix<Real> tmp(*this, kTrans);
        Resize(this->num_cols_, this->num_rows_);
        this->CopyFromMat(tmp);
    } else {
        (static_cast<CMatrixBase<Real>&>(*this)).Transpose();
    }
}

template<typename Real>
void CMatrix<Real>::Hermite() {
    if (this->num_rows_ != this->num_cols_) {
        CMatrix<Real> tmp(*this, kTrans);
        Resize(this->num_cols_, this->num_rows_);
        this->CopyFromMat(tmp);
        this->Conjugate();
    } else {
        CMatrixBase<Real>::Hermite();
    }
}

template<typename Real>
void CMatrix<Real>::Init(const MatrixIndexT rows, const MatrixIndexT cols,
                         const MatrixStrideType stride_type) {
    if (rows * cols == 0) {
        KALDI_ASSERT(rows == 0 && cols == 0);
        this->num_rows_ = 0;
        this->num_cols_ = 0;
        this->stride_ = 0;
        this->data_ = NULL;
        return;
    }
    KALDI_ASSERT(rows > 0 && cols > 0);
    // In fact, we need 2 * cols space
    MatrixIndexT skip, stride, double_cols = cols * 2;
    size_t size;
    void *data;  // aligned memory block
    void *temp;  // memory block to be really freed

    // compute the size of skip and real cols
    skip = ((16 / sizeof(Real)) - double_cols % (16 / sizeof(Real)))
        % (16 / sizeof(Real));
    stride = double_cols + skip;
    size = static_cast<size_t>(rows) * static_cast<size_t>(stride) * sizeof(Real);

    // allocate the memory and set the right dimensions and parameters
    if (NULL != (data = KALDI_MEMALIGN(16, size, &temp))) {
        CMatrixBase<Real>::data_        = static_cast<Real *> (data);
        CMatrixBase<Real>::num_rows_    = rows;
        CMatrixBase<Real>::num_cols_    = cols;
        // NOTE: double_cols instead of cols
        CMatrixBase<Real>::stride_  = (stride_type == kDefaultStride ? stride : double_cols);
    } else {
        throw std::bad_alloc();
    }
}

template<typename Real>
void CMatrix<Real>::Resize(const MatrixIndexT rows,
                           const MatrixIndexT cols,
                           MatrixResizeType resize_type,
                           MatrixStrideType stride_type) {
    // the next block uses recursion to handle what we have to do if
    // resize_type == kCopyData.
    if (resize_type == kCopyData) {
        if (this->data_ == NULL || rows == 0) resize_type = kSetZero;  // nothing to copy.
        else if (rows == this->num_rows_ && cols == this->num_cols_ &&
	     (stride_type == kDefaultStride || this->stride_ == this->num_cols_ * 2)) { return; } // nothing to do.
        else {
            // set tmp to a matrix of the desired size; if new matrix
            // is bigger in some dimension, zero it.
            MatrixResizeType new_resize_type =
            (rows > this->num_rows_ || cols > this->num_cols_) ? kSetZero : kUndefined;
            CMatrix<Real> tmp(rows, cols, new_resize_type, stride_type);
            MatrixIndexT rows_min = std::min(rows, this->num_rows_),
                         cols_min = std::min(cols, this->num_cols_);
            tmp.Range(0, rows_min, 0, cols_min).CopyFromMat(this->Range(0, rows_min, 0, cols_min));
            tmp.Swap(this);
            // and now let tmp go out of scope, deleting what was in *this.
            return;
        }
    }
    // At this point, resize_type == kSetZero or kUndefined.

    if (CMatrixBase<Real>::data_ != NULL) {
        if (rows == CMatrixBase<Real>::num_rows_ && cols == CMatrixBase<Real>::num_cols_) {
        if (resize_type == kSetZero)
            this->SetZero();
            return;
        }
        else
            Destroy();
    }
    Init(rows, cols, stride_type);
    if (resize_type == kSetZero) 
        CMatrixBase<Real>::SetZero();
}



// Implement for SubCMatrix

template<typename Real>
SubCMatrix<Real>::SubCMatrix(const CMatrixBase<Real> &M,
                             const MatrixIndexT row_offset, const MatrixIndexT rows,
                             const MatrixIndexT col_offset, const MatrixIndexT cols) {
    if (rows == 0 || cols == 0) {
        KALDI_ASSERT(cols == 0 && rows == 0);
        this->data_ = NULL;
        this->num_cols_ = 0;
        this->num_rows_ = 0;
        this->stride_ = 0;
        return;
    }
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(row_offset) <
                 static_cast<UnsignedMatrixIndexT>(M.num_rows_) &&
                 static_cast<UnsignedMatrixIndexT>(col_offset) <
                 static_cast<UnsignedMatrixIndexT>(M.num_cols_) &&
                 static_cast<UnsignedMatrixIndexT>(rows) <=
                 static_cast<UnsignedMatrixIndexT>(M.num_rows_ - row_offset) &&
                 static_cast<UnsignedMatrixIndexT>(cols) <=
                 static_cast<UnsignedMatrixIndexT>(M.num_cols_ - col_offset));
    // point to the begining of window
    CMatrixBase<Real>::num_rows_ = rows;
    CMatrixBase<Real>::num_cols_ = cols;
    CMatrixBase<Real>::stride_   = M.Stride();
    // col_offset need x2
    CMatrixBase<Real>::data_     = M.Data_workaround() +
        static_cast<size_t>(col_offset * 2) +
        static_cast<size_t>(row_offset) * static_cast<size_t>(M.Stride());
}


template<typename Real>
SubCMatrix<Real>::SubCMatrix(Real *data,
                             MatrixIndexT num_rows,
                             MatrixIndexT num_cols,
                             MatrixIndexT stride):
    CMatrixBase<Real>(data, num_cols, num_rows, stride) { // caution: reversed order!
    if (data == NULL) {
        KALDI_ASSERT(num_rows * num_cols == 0);
        this->num_rows_ = 0;
        this->num_cols_ = 0;
        this->stride_ = 0;
    } else {
        // at least 2 x num_cols
        KALDI_ASSERT(this->stride_ >= this->num_cols_ * 2);
    }
}


template class CMatrix<float>;
template class CMatrix<double>;
template class CMatrixBase<float>;
template class CMatrixBase<double>;
template class SubCMatrix<float>;
template class SubCMatrix<double>;

}
