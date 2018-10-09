// include/stft.cc
// wujian@18.2.12

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


#include "include/stft.h"

namespace kaldi {

// support multi-channel input
// wave:    (num_channels, num_samples)
// stft:    (num_channels x num_frames, num_bins)
void ShortTimeFTComputer::ShortTimeFT(const MatrixBase<BaseFloat> &wave, Matrix<BaseFloat> *stft) {
    KALDI_ASSERT(analysis_window_.Dim() == frame_length_);

    int32 num_samples = wave.NumCols(), num_channels = wave.NumRows();
    // affected by center
    int32 num_frames  = NumFrames(num_samples);
    
    stft->Resize(num_frames * num_channels, opts_.PaddingLength(), kSetZero);
    
    // new copy of wave, cause may modify origin matrix
    int32 pad_samples = opts_.center ? opts_.PaddingLength() >> 1: 0;
    Matrix<BaseFloat> copy_mat(num_channels, num_samples + pad_samples * 2);
    copy_mat.ColRange(pad_samples, num_samples).CopyFromMat(wave);

    if (opts_.normalize_input)
        copy_mat.Scale(1.0 / int16_max);

    int32 ibeg, iend;
    for (int32 c = 0; c < num_channels; c++) {
        // channel c
        SubVector<BaseFloat> samples(copy_mat, c);

        if (opts_.enable_scale) {
            BaseFloat samp_norm = samples.Norm(float_inf);
            samples.Scale(int16_max / samp_norm);
        }

        for (int32 i = 0; i < num_frames; i++) {
            SubVector<BaseFloat> spectra(*stft, c * num_frames + i);
            ibeg = i * frame_shift_;
            iend = ibeg + frame_length_ <= num_samples ? ibeg + frame_length_: num_samples;  
            spectra.Range(0, iend - ibeg).CopyFromVec(samples.Range(ibeg, iend - ibeg)); 
            spectra.Range(0, frame_length_).MulElements(analysis_window_);
            srfft_->Compute(spectra.Data(), true);
        }
    }
}
    
void ShortTimeFTComputer::ComputeSpectrogram(MatrixBase<BaseFloat> &stft, 
                                             Matrix<BaseFloat> *spectra) {
    int32 window_size = stft.NumCols(), num_frames = stft.NumRows();
    // index range(0, num_bins - 1)
    int32 num_bins = (window_size >> 1) + 1;

    spectra->Resize(num_frames, num_bins);
    for (int32 t = 0; t < num_frames; t++) {
        (*spectra)(t, 0) = stft(t, 0) * stft(t, 0);
        (*spectra)(t, num_bins - 1) = stft(t, 1) * stft(t, 1);
        for (int32 f = 1; f < num_bins - 1; f++) {
            BaseFloat r = stft(t, f * 2), i = stft(t, f * 2 + 1);
            (*spectra)(t, f) = r * r + i * i;
        }
    }
    if (!opts_.apply_pow)
        spectra->ApplyPow(0.5);
    if (opts_.apply_log) {
        // to avoid nan
        spectra->ApplyFloor(std::numeric_limits<BaseFloat>::epsilon());
        spectra->ApplyLog();
    }   
}


void ShortTimeFTComputer::ComputePhaseAngle(MatrixBase<BaseFloat> &stft, Matrix<BaseFloat> *angle) {
    int32 window_size = stft.NumCols(), num_frames = stft.NumRows();
    // index range(0, num_bins - 1)
    int32 num_bins = (window_size >> 1) + 1;
    angle->Resize(num_frames, num_bins);
    // processing angle(i, j)
    for (int32 t = 0; t < num_frames; t++) {
        (*angle)(t, 0) = atan2(0, stft(t, 0));
        (*angle)(t, num_bins - 1) = atan2(0, stft(t, 1));
        for (int32 f = 1; f < num_bins - 1; f++) {
            BaseFloat r = stft(t, f * 2), i = stft(t, f * 2 + 1);
            (*angle)(t, f) = atan2(i, r);
        }
    }
}

void ShortTimeFTComputer::Compute(const MatrixBase<BaseFloat> &wave, Matrix<BaseFloat> *stft, 
                                  Matrix<BaseFloat> *spectra, Matrix<BaseFloat> *angle) {
    KALDI_ASSERT(analysis_window_.Dim() == frame_length_);
    
    Matrix<BaseFloat> stft_cache;
    ShortTimeFT(wave, &stft_cache);
    
    if (spectra) {
        ComputeSpectrogram(stft_cache, spectra); 
    }
    if (angle) {
        ComputePhaseAngle(stft_cache, angle);
    }
    // copy back to stft
    if (stft) {
        stft->Swap(&stft_cache);
    }
} 

void ShortTimeFTComputer::Polar(MatrixBase<BaseFloat> &spectra, MatrixBase<BaseFloat> &angle, 
                                Matrix<BaseFloat> *stft) {
    KALDI_ASSERT(spectra.NumCols() == angle.NumCols() && spectra.NumRows() == angle.NumRows());
    int32 num_frames = spectra.NumRows(), num_bins = spectra.NumCols();
    int32 window_size = (num_bins - 1) * 2;
    stft->Resize(num_frames, window_size);
    
    if (opts_.apply_log)
        spectra.ApplyExp();
    if (opts_.apply_pow)
        spectra.ApplyPow(0.5);

    BaseFloat theta = 0;
    for (int32 t = 0; t < num_frames; t++) {
        (*stft)(t, 0) = spectra(t, 0);
        (*stft)(t, 1) = -spectra(t, num_bins - 1);
        for (int32 f = 1; f < num_bins - 1; f++) {
            theta = angle(t, f);
            (*stft)(t, f * 2) = cos(theta) * spectra(t, f);
            (*stft)(t, f * 2 + 1) = sin(theta) * spectra(t, f);
        }
    }
}

void ShortTimeFTComputer::InverseShortTimeFT(MatrixBase<BaseFloat> &stft, Matrix<BaseFloat> *wave, 
                                             BaseFloat range) {
    int32 num_frames = stft.NumRows();
    // should be longer than original
    int32 num_samples = NumSamples(num_frames); 
    wave->Resize(1, num_samples);
    
    SubVector<BaseFloat> samples(*wave, 0);
    Vector<BaseFloat> seg(frame_length_);

    for (int32 i = 0; i < num_frames; i++) {
        SubVector<BaseFloat> spectra(stft, i);
        // iRealFFT
        srfft_->Compute(spectra.Data(), false);
        spectra.Scale(1.0 / frame_length_);

        seg.CopyFromVec(spectra.Range(0, frame_length_));
        // NOTE: synthetic window should be orthogonalized with analysis window
        seg.MulElements(synthesis_window_);
        samples.Range(i * frame_shift_, frame_length_).AddVec(1, seg);
    }

    // cutoff padding zeros
    if (opts_.center) {
        int32 pad_samples = opts_.PaddingLength() >> 1;
        Matrix<BaseFloat> cutoff(1, num_samples - pad_samples * 2);
        cutoff.CopyFromMat(wave->ColRange(pad_samples, num_samples - pad_samples * 2));
        wave->Swap(&cutoff);
    }

    SubVector<BaseFloat> cutoff_samples(*wave, 0);
    // If range < 0, left what it is
    if (range >= 0) {
        BaseFloat samp_norm = cutoff_samples.Norm(float_inf);
        // by default, normalize to int16 to avoid cutoff when writing wave to disk
        if (range == 0)
            range = int16_max;
        // range < 0, do not normalize it.
        if (range >= 0) {
            cutoff_samples.Scale(range / samp_norm);
            KALDI_VLOG(3) << "Rescale samples(" << range << "/" << samp_norm << ")";
        }
    }        
}

void ShortTimeFTComputer::CacheAnalysisWindow(const ShortTimeFTOptions &opts) {
    int32 window_size = opts.frame_length;
    analysis_window_.Resize(window_size);
    double a = M_2PI / (window_size - 1);
    for (int32 i = 0; i < window_size; i++) {
        double d = static_cast<double>(i);
        // numpy's coeff is 0.42
        if (opts.window == "blackman") {
            analysis_window_(i) = 0.42 - 0.5 * cos(a * d) + 0.08 * cos(2 * a * d);
        } else if (opts.window == "hamming") {
            analysis_window_(i) = 0.54 - 0.46 * cos(a * d);
        } else if (opts.window == "hanning") {
            analysis_window_(i) = 0.50 - 0.50 * cos(a * d);          
        } else if (opts.window == "rectangular") {
            analysis_window_(i) = 1.0;
        } else {
            KALDI_ERR << "Unknown analysis window type " << opts.window;
        }
    }
    synthesis_window_.Resize(window_size);
    synthesis_window_.CopyFromVec(analysis_window_);
}

void ShortTimeFTComputer::CacheSynthesisWindow(const ShortTimeFTOptions &opts) {
    int32 window_size = opts_.frame_length;

    Vector<BaseFloat> analysis_window_square(analysis_window_), denominator(window_size);
    analysis_window_square.ApplyPow(2);

    int32 width = static_cast<int32>((window_size - 1) / opts_.frame_shift), s;

    for (int32 i = -width; i <= width; i++) {
        s = i * opts_.frame_shift;
        if (s < 0) {
            // [0: end - s] += [-s: end]
            denominator.Range(0, window_size + s).AddVec(1, analysis_window_square.Range(-s, window_size + s));
        } else{
            // [s: end] += [0: end - s]
            denominator.Range(s, window_size - s).AddVec(1, analysis_window_square.Range(0, window_size - s));
        }
    }
    // synthesis_window_.Resize(window_size);
    // synthesis_window_.CopyFromVec(analysis_window_);
    synthesis_window_.DivElements(denominator);
}


}

