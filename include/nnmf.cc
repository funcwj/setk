// nnmf.cc
// wujian@18.5.4

#include "include/nnmf.h"


namespace kaldi {

BaseFloat SparseNMF::Objf(const MatrixBase<BaseFloat> &V) {
    if (!SameDim(V, cost_))
        cost_.Resize(V.NumCols(), V.NumRows());
    cost_.SetZero();

    if (opts_.beta == 0) {
        Matrix<BaseFloat> x_div_y(Lambda_);
        x_div_y.DivElements(V); cost_.AddMat(1, x_div_y);
        x_div_y.ApplyLog(); cost_.AddMat(-1, x_div_y);
        cost_.Add(-1);
    } else if (opts_.beta == 0) {
        Matrix<BaseFloat> logx(Lambda_);
        logx.ApplyLog(); cost_.AddMat(1, logx);
        logx.CopyFromMat(V);
        logx.ApplyLog(); cost_.AddMat(-1, logx);
        cost_.MulElements(Lambda_);
        cost_.AddMat(1, V);
        cost_.AddMat(-1, Lambda_);
    } else if (opts_.beta == 0) {
        cost_.AddMat(1, Lambda_);
        cost_.AddMat(-1, V);
        cost_.ApplyPow(2);
    }
    // and sparsity
    return cost_.Sum() + H_.Sum() * opts_.sparsity;
}

bool SparseNMF::CheckNonNegative(const MatrixBase<BaseFloat> &M) {
    for (MatrixIndexT i = 0; i < M.NumRows(); i++)
        for (MatrixIndexT j = 0; j < M.NumCols(); j++)
            if (M(i, j) < 0)
                return false;
    return true;
}


BaseFloat SparseNMF::DoNMF(const MatrixBase<BaseFloat> &V,
                           Matrix<BaseFloat> *W, Matrix<BaseFloat> *H) {
    KALDI_ASSERT(V.NumRows() == W->NumRows() && V.NumCols() == H->NumCols());
    // Lazy init
    if (!SameDim(W_, *W))
        W_.Resize(W->NumRows(), W->NumCols()), Ws_.Resize(W->NumRows(), W->NumCols());
    if (!SameDim(H_, *H))
        H_.Resize(H->NumRows(), H->NumCols()), Hs_.Resize(H->NumRows(), H->NumCols());
    if (!SameDim(Lambda_, V))
        Lambda_.Resize(V.NumRows(), V.NumCols());

    // check non-negative
    if (!CheckNonNegative(*W))
        KALDI_ERR << "Matrix W has negative elements";
    if (!CheckNonNegative(*H))
        KALDI_ERR << "Matrix H has negative elements";

    // init values
    BaseFloat pre_cost, cur_cost;
    bool es = false;
    W_.CopyFromMat(*W), H_.CopyFromMat(*H);
    Lambda_.AddMatMat(1, W_, kNoTrans, H_, kNoTrans, 0);

    for (int32 iter = 0; iter < opts_.max_iter; iter++) {
        if (es)
            break;
        if (opts_.update_w)
            UpdateW(V);
        if (opts_.update_h)
            UpdateH(V);

        cur_cost = Objf(V);
        KALDI_LOG << "On iteration " << iter << ": objfunc " << cur_cost;

        if (iter >= 1) {
            if (std::abs(pre_cost - cur_cost) / pre_cost < opts_.conv_eps) {
                es = true;
                KALDI_LOG << "Convergence reached, aborting iteration";
            }
            pre_cost = cur_cost;
        }
    }
    if (opts_.update_w)
        W->CopyFromMat(W_);
    if (opts_.update_h)
        H->CopyFromMat(H_);
    return cur_cost;
}

void SparseNMF::UpdateH(const MatrixBase<BaseFloat> &V) {
    if (!SameDim(Lambda_, numerator))
        numerator.Resize(Lambda_.NumRows(), Lambda_.NumCols());
    if (!SameDim(Lambda_, denumerator_h))
        denumerator_h.Resize(Lambda_.NumRows(), Lambda_.NumCols());

    if (opts_.beta == 0) {
        Lambda_.ApplyPow(-1);
        denumerator_h.AddMatMat(1, W_, kTrans, Lambda_, kNoTrans, 0);
        denumerator_h.Add(opts_.sparsity);

        numerator.CopyFromMat(Lambda_);
        numerator.ApplyPow(2); 
        numerator.MulElements(V);

        Hs_.AddMatMat(1, W_, kTrans, numerator, kNoTrans, 0);
    } else if (opts_.beta == 1) {
        numerator.CopyFromMat(Lambda_);
        numerator.ApplyPow(-1); 
        numerator.MulElements(V);

        Lambda_.ApplyPow(0);
        denumerator_h.AddMatMat(1, W_, kTrans, Lambda_, kNoTrans, 0);
        denumerator_h.Add(opts_.sparsity);

        Hs_.AddMatMat(1, W_, kTrans, numerator, kNoTrans, 0);
    } else if (opts_.beta == 2) {
        denumerator_h.AddMatMat(1, W_, kTrans, Lambda_, kNoTrans, 0);
        denumerator_h.Add(opts_.sparsity);
        Hs_.AddMatMat(1, W_, kTrans, V, kNoTrans, 0);
    }

    Hs_.DivElements(denumerator_h);
    H_.MulElements(Hs_);
    // update Lambda
    Lambda_.AddMatMat(1, W_, kNoTrans, H_, kNoTrans, 0);
}

void SparseNMF::UpdateW(const MatrixBase<BaseFloat> &V) {
    if (!SameDim(Lambda_, numerator))
        numerator.Resize(Lambda_.NumRows(), Lambda_.NumCols());
    if (!SameDim(Lambda_, denumerator_w))
        denumerator_w.Resize(Lambda_.NumRows(), Lambda_.NumCols());

    if (opts_.beta == 0) {
        Lambda_.ApplyPow(-1);
        denumerator_w.AddMatMat(1, Lambda_, kNoTrans, H_, kTrans, 0);

        numerator.CopyFromMat(Lambda_);
        numerator.ApplyPow(2); 
        numerator.MulElements(V);

        Ws_.AddMatMat(1, numerator, kNoTrans, H_, kTrans, 0);

    } else if (opts_.beta == 1) {
        numerator.CopyFromMat(Lambda_);
        numerator.ApplyPow(-1); 
        numerator.MulElements(V);

        Lambda_.ApplyPow(0);
        denumerator_w.AddMatMat(1,  Lambda_, kNoTrans, H_, kTrans, 0);

        Ws_.AddMatMat(1, numerator, kNoTrans, H_, kTrans, 0);
    } else if (opts_.beta == 2) {
        denumerator_w.AddMatMat(1, Lambda_, kNoTrans, H_, kTrans, 0);
        Ws_.AddMatMat(1, V, kNoTrans, H_, kTrans, 0);
    }
    Ws_.DivElements(denumerator_w);
    W_.MulElements(Ws_);
    // update Lambda
    Lambda_.AddMatMat(1, W_, kNoTrans, H_, kNoTrans, 0);
}

}
