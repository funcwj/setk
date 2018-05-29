// srp-phat.cc
// wujian@18.5.29

#include "include/srp-phat.h"


namespace kaldi {

void SrpPhatComputor::ComputeAugularSpectrum(const CMatrixBase<BaseFloat> &L,
                                             const CMatrixBase<BaseFloat> &R,
                                             CMatrixBase<BaseFloat> *spectrum) {
    KALDI_ASSERT(spectrum->NumCols() == L.NumCols() && spectrum->NumRows() == L.NumRows());
    KALDI_ASSERT(spectrum->NumCols() == R.NumCols() && spectrum->NumRows() == R.NumRows());

    int32 num_frames = L.NumRows(), num_bins = L.NumRows();
    CMatrix<BaseFloat> cor(L);
    cor.MulElements(R, kConj);
    cor.DivElements(L, kNoConj, true);
    cor.DivElements(R, kNoConj, true);
}

void SrpPhatComputor::Compute(const CMatrixBase<BaseFloat> &stft, 
                              CMatrix<BaseFloat> *spectrum) {
    
}

    
} // kaldi
