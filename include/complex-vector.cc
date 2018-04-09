// complex-vector.cc
// wujian@2018

#include "include/complex-vector.h"
#include "include/complex-matrix.h"

namespace kaldi {

// Implement of CVectorBase

template<typename Real>
void CVectorBase<Real>::SetZero() {
    std::memset(data_, 0, 2 * dim_ * sizeof(Real));
}

template<typename Real>
void CVectorBase<Real>::SetRandn() {
    kaldi::RandomState rstate;
    for (MatrixIndexT i = 0; i < dim_ * 2; i += 2) {
        kaldi::RandGauss2(data_ + i, data_ + i + 1, &rstate);
    }
}

template<typename Real>
std::complex<Real> CVectorBase<Real>::Sum() const {
    Real sr = 0.0, si = 0.0;
    for (MatrixIndexT i = 0; i < dim_; i++) { 
        sr += (*this)(i, kReal);
        si += (*this)(i, kImag); 
    }
    return std::complex<Real>(sr, si);
}

template<typename Real>
void CVectorBase<Real>::Add(Real cr, Real ci) {
    for (MatrixIndexT i = 0; i < dim_; i++) {
        (*this)(i, kReal) += cr;
        (*this)(i, kImag) += ci;
    }
}

template<typename Real>
void CVectorBase<Real>::Conjugate() {
    for (MatrixIndexT i = 0; i < dim_; i++) {
        (*this)(i, kImag) *= (-1.0);
    }
}

template<typename Real>
void CVectorBase<Real>::Adjust() {
    for (MatrixIndexT i = 0; i < dim_; i++) {
        (*this)(i, kReal) += (*this)(i, kImag);
        (*this)(i, kReal) /= 2;
        (*this)(i, kImag) -= (*this)(i, kReal);
    }
}

template<typename Real>
void CVectorBase<Real>::Scale(const Real alpha_r, const Real alpha_i) {
    Complex<Real> alpha(alpha_r, alpha_i);
    cblas_CZscal(dim_, &alpha, data_, 1);
}


template<typename Real>
void CVectorBase<Real>::MulElements(const CVectorBase<Real> &v) {
    KALDI_ASSERT(dim_ == v.Dim());
    for (int32 i = 0; i < dim_; i++) 
        ComplexMul(v(i, kReal), v(i, kImag), data_ + i * 2, data_ + i * 2 + 1);
}


template<typename Real>
void CVectorBase<Real>::DivElements(const CVectorBase<Real> &v) {
    KALDI_ASSERT(dim_ == v.Dim());
    for (int32 i = 0; i < dim_; i++) 
        ComplexDiv(v(i, kReal), v(i, kImag), data_ + i * 2, data_ + i * 2 + 1);
}

template<typename Real>
void CVectorBase<Real>::AddVec(Real alpha_r, Real alpha_i, CVectorBase<Real> &v) {
    KALDI_ASSERT(v.dim_ == dim_);
    KALDI_ASSERT(&v != this);
    Complex<Real> alpha(alpha_r, alpha_i);
    cblas_CZaxpy(dim_, &alpha, v.Data(), 1, data_, 1);
}

template<typename Real>
void CVectorBase<Real>::AddMatVec(const Real alpha_r, const Real alpha_i, 
                                  const CMatrixBase<Real> &M, const MatrixTransposeType trans, 
                                  const CVectorBase<Real> &v,
                                  const Real beta_r, const Real beta_i) {
    KALDI_ASSERT((trans == kNoTrans && M.NumCols() == v.dim_ && M.NumRows() == dim_)
                 || (trans == kTrans && M.NumRows() == v.dim_ && M.NumCols() == dim_));
    KALDI_ASSERT(&v != this);
    Complex<Real> alpha(alpha_r, alpha_i), beta(beta_r, beta_i);
    cblas_CZgemv(trans, M.NumRows(), M.NumCols(), &alpha, M.Data(), M.Stride(),
                v.Data(), 1, &beta, data_, 1);
    Adjust();
}

template<typename Real>
void CVectorBase<Real>::CopyFromVec(const CVectorBase<Real> &v) {
    KALDI_ASSERT(dim_ == v.Dim());
    if (data_ != v.data_) {
        std::memcpy(this->data_, v.data_, 2 * dim_ * sizeof(Real));
    }
}

template<typename Real>
void CVectorBase<Real>::CopyFromVec(const VectorBase<Real> &v, ComplexIndexType kIndex) {
    KALDI_ASSERT(dim_ == v.Dim());
    for (int32 i = 0; i < dim_; i++)
        (*this)(i, kIndex) = v(i);
}


template<typename Real>
void CVectorBase<Real>::CopyFromRealfft(const VectorBase<Real> &v) {
    KALDI_ASSERT((dim_ - 1) * 2 == v.Dim());
    for (MatrixIndexT i = 0; i < dim_; i++) {
        if (i == 0)
            (*this)(i, kReal) = v(0), (*this)(i, kImag) = 0;
        else if (i == dim_ - 1)
            (*this)(i, kReal) = v(1), (*this)(i, kImag) = 0;
        else
            (*this)(i, kReal) = v(i * 2), (*this)(i, kImag) = v(i * 2 + 1);
    }
}
// Implement of CVector

template<typename Real>
inline void CVector<Real>::Init(const MatrixIndexT dim) {
    KALDI_ASSERT(dim >= 0);
    if (dim == 0) {
        this->dim_ = 0;
        this->data_ = NULL;
        return;
    }
    MatrixIndexT size;
    void *data;
    void *free_data;

    // scale by 2
    size = 2 * dim * sizeof(Real);

    if ((data = KALDI_MEMALIGN(16, size, &free_data)) != NULL) {
        this->data_ = static_cast<Real*> (data);
        this->dim_ = dim;
    } else {
        throw std::bad_alloc();
    }
}

template<typename Real>
void CVector<Real>::Destroy() {
    if (this->data_ != NULL)
        KALDI_MEMALIGN_FREE(this->data_);
    this->data_ = NULL;
    this->dim_ = 0;
}

template<typename Real>
void CVector<Real>::Swap(CVector<Real> *other) {
    std::swap(this->data_, other->data_);
    std::swap(this->dim_, other->dim_);
}

template<typename Real>
void CVector<Real>::Resize(const MatrixIndexT dim, MatrixResizeType resize_type) {
    // the next block uses recursion to handle what we have to do if
    // resize_type == kCopyData.
    if (resize_type == kCopyData) {
        if (this->data_ == NULL || dim == 0) resize_type = kSetZero;  // nothing to copy.
        else if (this->dim_ == dim) { return; } // nothing to do.
        else {
            // set tmp to a vector of the desired size.
            CVector<Real> tmp(dim, kUndefined);
            if (dim > this->dim_) {
                memcpy(tmp.data_, this->data_, sizeof(Real)*this->dim_*2);
                memset(tmp.data_ + this->dim_ * 2, 0, sizeof(Real)*(dim-this->dim_) * 2);
            } else {
                memcpy(tmp.data_, this->data_, sizeof(Real)*dim*2);
            }
            tmp.Swap(this);
            // and now let tmp go out of scope, deleting what was in *this.
            return;
        }
    }
    // At this point, resize_type == kSetZero or kUndefined.
    if (this->data_ != NULL) {
        if (this->dim_ == dim) {
            if (resize_type == kSetZero) this->SetZero();
            return;
        } else {
            Destroy();
        }
    }
    Init(dim);
    if (resize_type == kSetZero) this->SetZero();
}

template class CVector<float>;
template class CVector<double>;
template class SubCVector<float>;
template class SubCVector<double>;
template class CVectorBase<float>;
template class CVectorBase<double>;

}
