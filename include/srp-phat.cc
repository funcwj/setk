// include/srp-phat.cc
// wujian@18.5.29

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

#include "include/srp-phat.h"

namespace kaldi {

// GCC(x, y) = conj(GCC(y, x))
void SrpPhatComputor::ComputeGccPhat(const CMatrixBase<BaseFloat> &L,
                                     const CMatrixBase<BaseFloat> &R,
                                     BaseFloat dist,
                                     CMatrixBase<BaseFloat> *gcc_phat) {
  KALDI_ASSERT(dist > 0);
  BaseFloat max_tdoa = dist / opts_.sound_speed;
  BaseFloat inc_tdoa = max_tdoa * 2 / (opts_.samp_rate - 1);
  for (int32 i = 0; i < opts_.samp_rate; i++)
    if (opts_.samp_tdoa)
      delay_axis_(i) = (max_tdoa - inc_tdoa * i) * 2 * M_PI;
    else
      delay_axis_(i) =
          std::cos(i * M_PI / opts_.samp_rate) * max_tdoa * 2 * M_PI;

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
                              Matrix<BaseFloat> *spectra) {
  std::vector<BaseFloat> &topo = opts_.array_topo;
  int32 num_chs = topo.size();
  KALDI_ASSERT(num_chs >= 2);
  MatrixIndexT num_frames = stft.NumRows() / num_chs, num_bins = stft.NumCols();
  CMatrix<BaseFloat> coef(num_bins, delay_axis_.Dim());
  CMatrix<BaseFloat> srp_phat(num_frames, delay_axis_.Dim());
  spectra->Resize(num_frames, delay_axis_.Dim());

  // GCC(x, y) = conj(GCC(y, x))
  // GCC_PHAT(x, x) = I
  for (int32 i = 0; i < num_chs; i++) {
    for (int32 j = i + 1; j < num_chs; j++) {
      ComputeGccPhat(stft.RowRange(i * num_frames, num_frames),
                     stft.RowRange(j * num_frames, num_frames),
                     std::abs(topo[j] - topo[i]), &srp_phat);
    }
  }
  if (opts_.smooth_context) Smooth(&srp_phat);
  srp_phat.Part(spectra, kReal);
  spectra->ApplyFloor(0);
}

void SrpPhatComputor::Smooth(CMatrix<BaseFloat> *spectra) {
  int32 context = opts_.smooth_context;
  CMatrix<BaseFloat> smooth_spectra(spectra->NumRows(), spectra->NumCols());
  for (int32 t = 0; t < spectra->NumRows(); t++) {
    for (int32 c = -context; c <= context; c++) {
      int32 index = std::min(std::max(t + c, 0), spectra->NumRows() - 1);
      SubCVector<BaseFloat> ctx(*spectra, index);
      smooth_spectra.Row(t).AddVec(1, 0, ctx);
    }
  }
  smooth_spectra.Scale(1.0 / (2 * context + 1), 0);
  spectra->CopyFromMat(smooth_spectra);
}

}  // kaldi
