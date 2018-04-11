// beamformer.h
// wujian@2018

#ifndef BEAMFORMER_H
#define BEAMFORMER_H

#include "include/complex-base.h"
#include "include/complex-vector.h"
#include "include/complex-matrix.h"

namespace kaldi {

void CastIntoRealfft(const CMatrix<BaseFloat> &cstft,
                     Matrix<BaseFloat> *rstft) {
    int32 num_rows = cstft.NumRows(), num_cols = (cstft.NumCols() - 1) * 2;
    rstft->Resize(num_rows, num_cols);
    for (int32 r = 0; r < num_rows; r++) {
        for (int32 c = 0; c < cstft.NumCols(); c++) {
            if (c == 0)
                (*rstft)(r, 0) = cstft(r, c, kReal);
            else if (c == cstft.NumCols() - 1)
                (*rstft)(r, 1) = cstft(r, c, kReal);
            else
                (*rstft)(r, c * 2) = cstft(r, c, kReal), (*rstft)(r, c * 2 + 1) = cstft(r, c, kImag);
        }
    }
}

// src_stft:    (num_frames, num_bins x num_channels)
// dst_stft:    (num_bins x num_frames, num_channels)
// Shape multiple complex stft from shape num_frames x [num_bins * num_channels]
// into [num_bins * num_frames] x num_channels
// for convenience of psd estimate and beamforming
void ReshapeMultipleStft(const int32 num_bins, const int32 num_channels, 
                         const CMatrix<BaseFloat> &src_stft,
                         CMatrix<BaseFloat> *dst_stft) {
    KALDI_ASSERT(num_channels * num_bins == src_stft.NumCols());
    int32 num_frames = src_stft.NumRows();
    dst_stft->Resize(num_bins * num_frames, num_channels);
    
    for (int32 c = 0; c < num_channels; c++) {
        for (int32 f = 0; f < num_bins; f++) {
            dst_stft->Range(f * num_frames, num_frames, c, 1)
                .CopyFromMat(src_stft.ColRange(num_bins * c + f, 1));
        }
    }
}

//
// src_stft:    (num_bins x num_frames, num_channels)
// target_mask: (num_frames, num_bins)
// target_psd:  (num_bins x num_channels, num_channels)
//
void EstimatePsd(const CMatrixBase<BaseFloat> &src_stft, 
                 const MatrixBase<BaseFloat> &target_mask,
                 CMatrix<BaseFloat> *target_psd,
                 CMatrix<BaseFloat> *second_psd) {
    int32 num_channels = src_stft.NumCols(), num_frames = target_mask.NumRows(),
          num_bins = target_mask.NumCols();
    KALDI_ASSERT(target_psd);
    target_psd->Resize(num_bins * num_channels, num_channels);
    if (second_psd)
        second_psd->Resize(num_bins * num_channels, num_channels);

    for (int32 f = 0; f < num_bins; f++) {
        BaseFloat mask_sum = 0.0, mask = 0.0;
        for (int32 t = 0; t < num_frames; t++) {
            SubCVector<BaseFloat> obs(src_stft, f * num_frames + t);
            mask = target_mask(t, f);
            target_psd->RowRange(f * num_channels, num_channels).AddVecVec(mask, 0, obs, obs, kConj);
            if (second_psd)
                second_psd->RowRange(f * num_channels, num_channels).AddVecVec(1 - mask, 0, obs, obs, kConj);
            mask_sum += mask;
        }
        target_psd->RowRange(f * num_channels, num_channels).Scale(1.0 / mask_sum, 0);
        if (second_psd)
            second_psd->RowRange(f * num_channels, num_channels).Scale(1.0 / (num_frames - mask_sum), 0);
    }
}

// target_psd:  (num_bins x num_channels, num_channels)
// steer_vector:(num_bins, num_channels)
// using maximum eigen vector as estimation of steer vector
void EstimateSteerVector(const CMatrixBase<BaseFloat> &target_psd,
                         CMatrix<BaseFloat> *steer_vector) {
    int32 num_channels = target_psd.NumCols();
    KALDI_ASSERT(target_psd.NumRows() % num_channels == 0);
    int32 num_bins = target_psd.NumRows() / num_channels;
    steer_vector->Resize(num_bins, num_channels);
    
    CMatrix<BaseFloat> V(num_channels, num_channels); Vector<BaseFloat> D(num_channels);
    for (int32 f = 0; f < num_bins; f++) {
        target_psd.RowRange(f * num_channels, num_channels).HEig(&D, &V);
        KALDI_VLOG(3) << "Compute eigen-dcomposition for matrix: " << target_psd.RowRange(f * num_channels, num_channels);
        KALDI_VLOG(3) << "Computed eigen values:" << D;
        KALDI_VLOG(3) << "Computed eigen vectors(row-major):" << V;
        // steer_vector->Row(f).CopyFromVec(V.Row(num_channels - 1), kConj); 
        steer_vector->Row(f).CopyFromVec(V.Row(num_channels - 1), kConj); 
    }
}


// target_psd:  (num_bins x num_channels, num_channels)
// steer_vector:(num_bins, num_channels)
// beam_weights:(num_bins, num_channels)
// note mvdr:
// numerator = psd_inv * steer_vector
// denumerator = numerator * steer_vector^H
// weight    = numerator / denumerator
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
        KALDI_VLOG(3) << "Noise power spectrum matrix: " << psd_inv;
        KALDI_VLOG(3) << "Using steer vector: " << steer;
        psd_inv.Invert(); // may be singular, using diag loading to avoid
        numerator.AddMatVec(1, 0, psd_inv, kNoTrans, steer, 0, 0); 
        KALDI_VLOG(3) << "R^{-1} * d: " << numerator;
        std::complex<BaseFloat> s = std::complex<BaseFloat>(1.0, 0) / VecVec(numerator, steer, kConj);
        KALDI_VLOG(3) << "1 / (d^H * R^{-1} * d): " << "(" << std::real(s) 
                      << (std::imag(s) >= 0 ? "+": "") << std::imag(s) << ")" << std::endl;
        numerator.Scale(std::real(s), std::imag(s));
        KALDI_VLOG(3) << "R^{-1} * d / (d^H * R^{-1} * d): " << numerator;
    }
    beam_weights->Conjugate();
    // using beam_weights in Beamform
}


// target_psd:  (num_bins x num_channels, num_channels)
// noise_psd:  (num_bins x num_channels, num_channels)
// beam_weights:(num_bins, num_channels)
void ComputeGevdBeamWeights(const CMatrixBase<BaseFloat> &target_psd,
                            const CMatrixBase<BaseFloat> &noise_psd,
                            CMatrix<BaseFloat> *beam_weights) {
    KALDI_ASSERT(target_psd.NumCols() == noise_psd.NumCols() && target_psd.NumRows() == noise_psd.NumRows()); 
    KALDI_ASSERT(target_psd.NumRows() % target_psd.NumRows() == 0);
    int32 num_channels = target_psd.NumCols(), num_bins = target_psd.NumRows() / target_psd.NumCols();

    beam_weights->Resize(num_bins, num_channels);
    CMatrix<BaseFloat> V(num_channels, num_channels); Vector<BaseFloat> D(num_channels);
    for (int32 f = 0; f < num_bins; f++) {
        SubCMatrix<BaseFloat> B(noise_psd, f * num_channels, num_channels, 0, num_channels);
        target_psd.RowRange(f * num_channels, num_channels).HGeneralizedEig(&B, &D, &V);     
        beam_weights->Row(f).CopyFromVec(V.Row(num_channels - 1), kConj);
    }
}


// src_stft:    (num_bins x num_frames, num_channels)
// weights:     (num_bins, num_channels), need to apply conjugate before calling this function
// enh_stft:    (num_frames, num_bins)
// note:
// To avoid Transpose, using AddMatMat instead of:
// enh_stft->Resize(num_bins, num_frames);
// for (int32 f = 0; f < num_bins; f++)
//      enh_stft->Row(f).AddMatVec(1, 0, src_stft.RowRange(f * num_frames, num_frames), kNoTrans, weights.Row(f), 0, 0);
// enh_stft->Transpose();
    
void Beamform(const CMatrixBase<BaseFloat> &src_stft, 
              const CMatrixBase<BaseFloat> &weights,
              CMatrix<BaseFloat> *enh_stft) {
    KALDI_ASSERT(src_stft.NumCols() == weights.NumCols());
    KALDI_ASSERT(src_stft.NumRows() % weights.NumRows() == 0);
    int32 num_bins = weights.NumRows(), num_channels = weights.NumCols(),
          num_frames = src_stft.NumRows() / num_bins; 
    
    enh_stft->Resize(num_frames, num_bins);
    // enh_stft[f] = src_stft[f * t: f * t + t] * w^H
    for (int32 f = 0; f < num_bins; f++) {
        enh_stft->ColRange(f, 1).AddMatMat(1, 0, src_stft.RowRange(f * num_frames, num_frames), 
                                           kNoTrans, weights.RowRange(f, 1), kTrans, 0, 0);
    }
}

}

#endif
