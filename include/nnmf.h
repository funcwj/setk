// nnmf.h
// wujian@18.5.4


#ifndef NNMF_H
#define NNMF_H

#include "matrix/matrix-lib.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"


namespace kaldi {


struct SparseNMFOptions {

    int32 max_iter;
    bool update_w, update_h;
    BaseFloat beta, sparsity, conv_eps;

    SparseNMFOptions(): beta(1), sparsity(0),
                        update_w(true), update_h(true),
                        max_iter(100), conv_eps(0) {}

    void Register(OptionsItf *opts) {
        opts->Register("beta", &beta,
                        "Value of beta in definition of beta-divergence,"
                        "beta = 0 yields the Itakura-Saito (IS) distance,"
                        "beta = 1 yields the generalized Kullback-Leibler(KL) divergence,"
                        "beta = 2 yields the Euclidean distance");
        opts->Register("update-dictionary", &update_w,
                        "Update value of matrix W or not, for supervised-NMF, matrix W"
                        "always will not be trained and only matrix H is updated");
        opts->Register("update-activations", &update_h,
                        "Upate value of matrix H or not");
        opts->Register("max-iter", &max_iter, 
                        "Maximum number of iterations to update W and H");
        opts->Register("conv-eps", &conv_eps,
                        "Predefined threshold for early stopping");
        opts->Register("sparsity", &sparsity,
                        "Weight for the L1 sparsity penalty on matrix W");

    }
};

// Implement of Sparse Non-negative Matrix Factor
// Reference:
//      Le Roux J, Weninger F J, Hershey J R. Sparse NMF–half-baked or well done?[J]. 
//      Mitsubishi Electric Research Labs (MERL), Cambridge, MA, USA, Tech. Rep., 
//      no. TR2015-023, 2015.

class SparseNMF {

public:
    SparseNMF(const SparseNMFOptions &opts): opts_(opts) {}

    BaseFloat Objf(const MatrixBase<BaseFloat> &V);

    BaseFloat DoNMF(const MatrixBase<BaseFloat> &V,
                    Matrix<BaseFloat> *W, Matrix<BaseFloat> *H);

private:

    SparseNMF &operator = (const SparseNMF &in);

    void UpdateH(const MatrixBase<BaseFloat> &V);

    void UpdateW(const MatrixBase<BaseFloat> &V);

    bool CheckNonNegative(const MatrixBase<BaseFloat> &M);

    SparseNMFOptions opts_;
    std::string objf_type;
    // some variable for NMF
    Matrix<BaseFloat> W_, H_, Ws_, Hs_, Lambda_, cost_;
    Matrix<BaseFloat> numerator, denumerator_w, denumerator_h;
};




}

#endif