// beamformer.h
// wujian@2018

#ifndef BEAMFORMER_H
#define BEAMFORMER_H

#include <vector>
#include "include/complex-base.h"
#include "include/complex-vector.h"
#include "include/complex-matrix.h"

namespace kaldi {


//
// src_stft: (num_bins x num_frames, num_channels)
// target_mask: (num_frames, num_bins)
// return target_psd: (num_bins x num_channels, num_channels)
//
void EstimatePsd(const CMatrixBase<BaseFloat> &src_stft, 
                 const MatrixBase<BaseFloat> &target_mask,
                 CMatrix<BaseFloat> *target_psd) {
    int32 num_channels = src_stft.NumCols(), num_frames = target_mask.NumRows(),
          num_bins = target_mask.NumCols();
    target_psd->Resize(num_bins * num_channels, num_channels);

    for (int32 f = 0; f < num_bins; f++) {
        BaseFloat mask_sum = 0.0, mask = 0.0;
        for (int32 t = 0; t < num_frames; t++) {
            SubCVector<BaseFloat> obs(src_stft, f * num_frames + t);
            mask = target_mask(t, f);
            (*target_psd).RowRange(f * num_channels, num_channels).AddVecVec(mask, 0, obs, obs, kConj);
            mask_sum += mask;
        }
        (*target_psd).RowRange(f * num_channels, num_channels).Scale(1.0 / mask_sum, 0);
    }
}

// target_psd: (num_bins x num_channels, num_channels)
// return steer_vector (num_bins, num_channels)
//
void EstimateSteerVector(const CMatrixBase<BaseFloat> &target_psd,
                         CMatrix<BaseFloat> *steer_vector) {
    int32 num_channels = target_psd.NumCols();
    KALDI_ASSERT(target_psd.NumRows() % num_channels == 0);
    int32 num_bins = target_psd.NumRows() / num_channels;
    steer_vector->Resize(num_bins, num_channels);
    
    CMatrix<BaseFloat> V; Vector<BaseFloat> D;
    for (int32 f = 0; f < num_bins; f++) {
        target_psd.RowRange(f * num_channels, num_channels).HEig(&D, &V);
        KALDI_VLOG(3) << "Computed eigen values:" << D;
        KALDI_VLOG(3) << "Computed eigen vectors(row):" << V;
        steer_vector->Row(f).CopyFromVec(V.Row(num_channels - 1)); 
    }
    steer_vector->Conjugate();
}


// target_psd: (num_bins x num_channels, num_channels)
// steer_vector (num_bins, num_channels)
// return beam_weights: (num_bins, num_channels)
void ComputeMvdrBeamWeights(const CMatrixBase<BaseFloat> &noise_psd,
                            const CMatrixBase<BaseFloat> &steer_vector,
                            CMatrix<BaseFloat> *beam_weights) {
    KALDI_ASSERT(noise_psd.NumCols() == steer_vector.NumCols());
    KALDI_ASSERT(noise_psd.NumRows() % steer_vector.NumCols() == 0);
    int32 num_bins = steer_vector.NumRows(), num_channels = steer_vector.NumCols();

    CMatrix<BaseFloat> psd_inv(num_channels, num_channels);
    beam_weights->Resize(num_bins, num_channels);
    for (int32 f = 0; f < num_bins; f++) {
        SubCVector<BaseFloat> numerator(*beam_weights, f), steer(steer_vector, f);
        psd_inv.CopyFromMat(noise_psd.RowRange(f * num_channels, num_channels));
        psd_inv.Invert();
        numerator.AddMatVec(1, 0, psd_inv, kNoTrans, steer, 0, 0); 
        std::complex<BaseFloat> s = VecVec(numerator, steer, kConj);
        numerator.Scale(std::real(s), std::imag(s));
    }
    beam_weights->Conjugate();
    // using beam_weights in Beamform
}

// src_stft: (num_bins x num_frames, num_channels)
// weights: (num_bins, num_channels), need to apply conjugate before function is called
// return enh_stft: (num_frames, num_bins)
void Beamform(const CMatrixBase<BaseFloat> &src_stft, 
              const CMatrixBase<BaseFloat> &weights,
              CMatrix<BaseFloat> *enh_stft) {
    KALDI_ASSERT(src_stft.NumCols() == weights.NumCols());
    KALDI_ASSERT(src_stft.NumRows() % weights.NumRows() == 0);
    int32 num_bins = weights.NumRows(), num_channels = weights.NumCols(),
          num_frames = src_stft.NumRows() / num_bins; 
    
    // NOTE:
    // To avoid Transpose, using AddMatMat instead of:
    // enh_stft->Resize(num_bins, num_frames);
    // for (int32 f = 0; f < num_bins; f++)
    //      enh_stft->Row(f).AddMatVec(1, 0, src_stft.RowRange(f * num_frames, num_frames), kNoTrans, weights.Row(f), 0, 0);
    // enh_stft->Transpose();
    
    enh_stft->Resize(num_frames, num_bins);
    for (int32 f = 0; f < num_bins; f++) {
        // enh_stft[f] = src_stft[f * t: f * t + t] * w^H
        enh_stft->ColRange(f, 1).AddMatMat(1, 0, src_stft.RowRange(f * num_frames, num_frames), 
                                           kNoTrans, weights.RowRange(f, 1), kTrans, 0, 0);
    }
}

}

#endif
