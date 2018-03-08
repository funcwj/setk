// stft.cc
// wujian@18.2.12
// 
//
#include "include/stft.h"

namespace kaldi {

void ShortTimeFTComputer::ShortTimeFT(const MatrixBase<BaseFloat> &wave, Matrix<BaseFloat> *stft) {
    KALDI_ASSERT(wave.NumRows() == 1);
    KALDI_ASSERT(window_.Dim() == frame_length_);

    int32 num_samples = wave.NumCols();
    int32 num_frames  = NumFrames(num_samples);
    
    stft->Resize(num_frames, opts_.PaddingLength(), kSetZero);
    
    // new copy of wave, cause my modify origin matrix
    Matrix<BaseFloat> copy_mat(wave);
    SubVector<BaseFloat> samples(copy_mat, 0);

    if (opts_.normalize_input)
        samples.Scale(1.0 / int16_max);

    int32 ibeg, iend;
    for (int32 i = 0; i < num_frames; i++) {
        SubVector<BaseFloat> specs(*stft, i);
        ibeg = i * frame_shift_;
        iend = ibeg + frame_length_ <= num_samples ? ibeg + frame_length_: num_samples;  
        specs.Range(0, iend - ibeg).CopyFromVec(samples.Range(ibeg, iend - ibeg)); 
        specs.Range(0, frame_length_).MulElements(window_);
        srfft_->Compute(specs.Data(), true);
    } 
}
    
void ShortTimeFTComputer::ComputeSpectrum(MatrixBase<BaseFloat> &stft, 
                                          Matrix<BaseFloat> *spectrum) {
    int32 window_size = stft.NumCols(), num_frames = stft.NumRows();
    // index range(0, num_bins - 1)
    int32 num_bins = (window_size >> 1) + 1;

    spectrum->Resize(num_frames, num_bins);
    for (int32 t = 0; t < num_frames; t++) {
        (*spectrum)(t, 0) = stft(t, 0) * stft(t, 0);
        (*spectrum)(t, num_bins - 1) = stft(t, 1) * stft(t, 1);
        for (int32 f = 1; f < num_bins - 1; f++) {
            BaseFloat r = stft(t, f * 2), i = stft(t, f * 2 + 1);
            (*spectrum)(t, f) = r * r + i * i;
        }
    }
    if (!opts_.apply_pow)
        spectrum->ApplyPow(0.5);
    if (opts_.apply_log) {
        // to avoid nan
        spectrum->ApplyFloor(std::numeric_limits<BaseFloat>::epsilon());
        spectrum->ApplyLog();
    }   
}


void ShortTimeFTComputer::ComputeArg(MatrixBase<BaseFloat> &stft, Matrix<BaseFloat> *arg) {
    int32 window_size = stft.NumCols(), num_frames = stft.NumRows();
    // index range(0, num_bins - 1)
    int32 num_bins = (window_size >> 1) + 1;
    arg->Resize(num_frames, num_bins);
    // processing arg(i, j)
    for (int32 t = 0; t < num_frames; t++) {
        (*arg)(t, 0) = atan2(0, stft(t, 0));
        (*arg)(t, num_bins - 1) = atan2(0, stft(t, 1));
        for (int32 f = 1; f < num_bins - 1; f++) {
            BaseFloat r = stft(t, f * 2), i = stft(t, f * 2 + 1);
            (*arg)(t, f) = atan2(i, r);
        }
    }
}

void ShortTimeFTComputer::Compute(const MatrixBase<BaseFloat> &wave, Matrix<BaseFloat> *stft, 
                                  Matrix<BaseFloat> *spectrum, Matrix<BaseFloat> *arg) {
    // support one channel
    KALDI_ASSERT(wave.NumRows() == 1);
    KALDI_ASSERT(window_.Dim() == frame_length_);

    int32 num_samples = wave.NumCols();
    int32 num_frames  = NumFrames(num_samples);
    
    Matrix<BaseFloat> stft_cache(num_frames, opts_.PaddingLength());
    ShortTimeFT(wave, &stft_cache);
    
    if (spectrum) {
        ComputeSpectrum(stft_cache, spectrum); 
    }
    if (arg) {
        ComputeArg(stft_cache, arg);
    }
    // copy back to stft
    if (stft) {
        stft->Swap(&stft_cache);
    }
} 

void ShortTimeFTComputer::Polar(MatrixBase<BaseFloat> &spectrum, MatrixBase<BaseFloat> &arg, 
                                             Matrix<BaseFloat> *stft) {
    KALDI_ASSERT(spectrum.NumCols() == arg.NumCols() && spectrum.NumRows() == arg.NumRows());
    int32 num_frames = spectrum.NumRows(), num_bins = spectrum.NumCols();
    int32 window_size = (num_bins - 1) * 2;
    stft->Resize(num_frames, window_size);
    
    if (opts_.apply_log)
        spectrum.ApplyExp();
    if (opts_.apply_pow)
        spectrum.ApplyPow(0.5);

    for (int32 t = 0; t < num_frames; t++) {
        (*stft)(t, 0) = spectrum(t, 0);
        (*stft)(t, 1) = -spectrum(t, num_bins - 1);
        for (int32 f = 1; f < num_bins - 1; f++) {
           BaseFloat theta = arg(t, f);
           (*stft)(t, f * 2) = cos(theta) * spectrum(t, f);
           (*stft)(t, f * 2 + 1) = sin(theta) * spectrum(t, f);
        }
    }
}

void ShortTimeFTComputer::InverseShortTimeFT(MatrixBase<BaseFloat> &stft, Matrix<BaseFloat> *wave, 
                                             BaseFloat range) {
    int32 num_frames = stft.NumRows();
    int32 num_samples = NumSamples(num_frames); 
    wave->Resize(1, num_samples);
    
    SubVector<BaseFloat> samples(*wave, 0);
    Vector<BaseFloat> seg(frame_length_);

    for (int32 i = 0; i < num_frames; i++) {
        SubVector<BaseFloat> specs(stft, i);
        // iRealFFT
        srfft_->Compute(specs.Data(), false);
        specs.Scale(1.0 / frame_length_);

        seg.CopyFromVec(specs.Range(0, frame_length_));
        seg.MulElements(window_);
        samples.Range(i * frame_shift_, frame_length_).AddVec(1, seg);
    }

    BaseFloat samp_norm = samples.Norm(float_inf);
    if (range == 0) 
        range = int16_max;

    samples.Scale(range / samp_norm);
    KALDI_VLOG(3) << "Rescale samples(" << range << "/" << samp_norm << ")";
}

void ShortTimeFTComputer::CacheWindow(const ShortTimeFTOptions &opts) {
    int32 frame_length = opts.frame_length;
    window_.Resize(frame_length);
    double a = M_2PI / (frame_length - 1);
    for (int32 i = 0; i < frame_length; i++) {
        double d = static_cast<double>(i);
        // numpy's coeff is 0.42
        if (opts.window == "blackman") {
            window_(i) = 0.42 - 0.5 * cos(a * d) + 0.08 * cos(2 * a * d);
        } else if (opts.window == "hamming") {
            window_(i) = 0.54 - 0.46 * cos(a * d);
        } else if (opts.window == "hanning") {
            window_(i) = 0.50 - 0.50 * cos(a * d);          
        } else if (opts.window == "rectangular") {
            window_(i) = 1.0;
        } else {
            KALDI_ERR << "Unknown window type " << opts.window;
        }
    }
}

}

