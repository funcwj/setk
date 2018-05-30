// srp-phat.cc
// wujian@18.5.29

#include "include/srp-phat.h"


namespace kaldi {

void SrpPhatComputor::ComputeGccPhat(const CMatrixBase<BaseFloat> &L,
                                     const CMatrixBase<BaseFloat> &R,
                                     BaseFloat dist,
                                     CMatrixBase<BaseFloat> *gcc_phat) {
    BaseFloat max_tdoa = dist / opts_.sound_speed;
    BaseFloat step_tdoa = max_tdoa * 2 / (opts_.doa_resolution - 1);
    for (int32 i = 0; i < opts_.doa_resolution; i++)
        delay_axis_(i) = (max_tdoa - step_tdoa * i) * 2 * M_PI;

    idtft_coef_.SetZero();
    idtft_coef_.AddVecVec(1, frequency_axis_, delay_axis_);
    exp_idtft_coef_j_.Exp(idtft_coef_);

    CMatrix<BaseFloat> cor(L);
    cor.MulElements(R, kConj);
    cor.DivElements(L, kNoConj, true);
    cor.DivElements(R, kNoConj, true);
    // gcc_phat = gcc_phat + cor * coef
    gcc_phat->AddMatMat(1, 0, cor, kNoTrans, exp_idtft_coef_j_, kNoTrans, 1, 0);
}


void SrpPhatComputor::Compute(const CMatrixBase<BaseFloat> &stft, 
                              Matrix<BaseFloat> *spectrum) {
    std::vector<BaseFloat> &topo = opts_.array_topo; 
    int32 num_chs = topo.size();
    KALDI_ASSERT(num_chs >= 2);
    MatrixIndexT num_frames = stft.NumRows() / num_chs, num_bins = stft.NumCols();
    CMatrix<BaseFloat> coef(num_bins, delay_axis_.Dim());
    CMatrix<BaseFloat> srp_phat(num_frames, delay_axis_.Dim());
    spectrum->Resize(num_frames, delay_axis_.Dim());

    for (int32 i = 0; i < num_chs; i++) {
        for (int32 j = 0; j < num_chs; j++) {
            ComputeGccPhat(stft.RowRange(i * num_frames, num_frames),
                           stft.RowRange(j * num_frames, num_frames),
                           topo[i] - topo[j], &srp_phat);
        }
    }
    if (opts_.smooth_context)
        Smooth(&srp_phat);
    srp_phat.Part(spectrum, kReal);
}

void SrpPhatComputor::Smooth(CMatrix<BaseFloat> *spectrum) {
    int32 context = opts_.smooth_context;
    CMatrix<BaseFloat> smooth_spectrum(spectrum->NumRows(), spectrum->NumCols());
    for (int32 t = 0; t < spectrum->NumRows(); t++) {
        for (int32 c = -context; c <= context; c++) {
            int32 index = std::min(std::max(t + c, 0), spectrum->NumRows() - 1);
            SubCVector<BaseFloat> ctx(*spectrum, index);
            smooth_spectrum.Row(t).AddVec(1, 0, ctx);
        }
    }
    smooth_spectrum.Scale(1.0 / (2 * context + 1), 0);
    spectrum->CopyFromMat(smooth_spectrum);
}
    
} // kaldi
