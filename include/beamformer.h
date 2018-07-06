// include/beamformer.h
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


#ifndef BEAMFORMER_H
#define BEAMFORMER_H

#include "include/complex-base.h"
#include "include/complex-vector.h"
#include "include/complex-matrix.h"

namespace kaldi {

// Cast CMatrix into Matrix, in Realfft format, to reconstruct speech
// The Realfft format is space efficient, so I refused to use CMatrix in stft.h
void CastIntoRealfft(const CMatrixBase<BaseFloat> &cstft,
                     Matrix<BaseFloat> *rstft);

// src_stft:    (num_frames, num_bins x num_channels) or
//              (num_frames x num_channels, num_bins)
// dst_stft:    (num_bins x num_frames, num_channels)
// Shape multiple complex stft from shape num_frames x [num_bins * num_channels]
// or [num_frames x num_channels] x num_bins into [num_bins * num_frames] x num_channels
// for convenience of psd estimate and beamforming
void TrimStft(const int32 num_bins, const int32 num_channels, 
              const CMatrixBase<BaseFloat> &src_stft,
              CMatrix<BaseFloat> *dst_stft);

//
// src_stft:    (num_bins x num_frames, num_channels)
// target_mask: (num_frames, num_bins)
// target_psd:  (num_bins x num_channels, num_channels)
//
void EstimatePsd(const CMatrixBase<BaseFloat> &src_stft, 
                 const MatrixBase<BaseFloat> &target_mask,
                 CMatrix<BaseFloat> *target_psd,
                 CMatrix<BaseFloat> *second_psd);

// target_psd:  (num_bins x num_channels, num_channels)
// steer_vector:(num_bins, num_channels)
// using maximum eigen vector as estimation of steer vector
void EstimateSteerVector(const CMatrixBase<BaseFloat> &target_psd,
                         CMatrix<BaseFloat> *steer_vector);


// target_psd:  (num_bins x num_channels, num_channels)
// steer_vector:(num_bins, num_channels)
// beam_weights:(num_bins, num_channels)
// NOTE mvdr:
// numerator = psd_inv * steer_vector
// denumerator = numerator * steer_vector^H
// weight    = numerator / denumerator
void ComputeMvdrBeamWeights(const CMatrixBase<BaseFloat> &noise_psd,
                            const CMatrixBase<BaseFloat> &steer_vector,
                            CMatrix<BaseFloat> *beam_weights);


// target_psd:  (num_bins x num_channels, num_channels)
// noise_psd:   (num_bins x num_channels, num_channels)
// beam_weights:(num_bins, num_channels)
void ComputeGevdBeamWeights(const CMatrixBase<BaseFloat> &target_psd,
                            const CMatrixBase<BaseFloat> &noise_psd,
                            CMatrix<BaseFloat> *beam_weights);


// src_stft:    (num_bins x num_frames, num_channels)
// weights:     (num_bins, num_channels)
// enh_stft:    (num_frames, num_bins)
// NOTE:
// To avoid Transpose, using AddMatMat instead of:
// enh_stft->Resize(num_bins, num_frames);
// for (int32 f = 0; f < num_bins; f++)
//      enh_stft->Row(f).AddMatVec(1, 0, src_stft.RowRange(f * num_frames, num_frames), kNoTrans, weights.Row(f), 0, 0);
// enh_stft->Transpose();
    
void Beamform(const CMatrixBase<BaseFloat> &src_stft, 
              const CMatrixBase<BaseFloat> &weights,
              CMatrix<BaseFloat> *enh_stft);

}

#endif
