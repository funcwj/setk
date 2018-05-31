// srp-phat.h
// wujian@18.5.29

#ifndef SRP_PHAT_H
#define SRP_PHAT_H

#include <iterator>
#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "include/complex-base.h"
#include "include/complex-vector.h"
#include "include/complex-matrix.h"



namespace kaldi {

struct SrpPhatOptions {

    BaseFloat sound_speed;
    int32 samp_rate, smooth_context;
    // For linear microphone arrays
    std::string topo_descriptor;
    std::vector<BaseFloat> array_topo;
    bool samp_doa = false, samp_tdoa = true;

    SrpPhatOptions(): sound_speed(340.4), 
        samp_rate(180), smooth_context(0),
        topo_descriptor("") {} 

    void Register(OptionsItf *opts) {
        opts->Register("sound-speed", &sound_speed, "Speed of sound(or other kinds of wave)");
        opts->Register("samp-rate", &samp_rate, 
                    "Sample rate of DoA/TDoA, depending on options --samp-doa and --samp-tdoa");
        opts->Register("samp-doa", &samp_doa, "Sample doa instead of tdoa on y-axis");
        opts->Register("samp-tdoa", &samp_tdoa, "Sample tdoa instead of doa on y-axis, by default using it");
        opts->Register("smooth-context", &smooth_context, "Context of frames used for spectra smoothing");
        opts->Register("topo-descriptor", &topo_descriptor, 
                    "Description of microarray's topology, now only support linear array."
                    " egs: --topo-descriptor=0,0.3,0.6,0.9 described a ULA with element spacing equals 0.3");
    }

    void ComputeDerived() {
        if (samp_tdoa && samp_doa) 
            KALDI_ERR << "Options --samp-tdoa conflicts with --samp-doa";
        KALDI_ASSERT(topo_descriptor != "");
        KALDI_ASSERT(SplitStringToFloats(topo_descriptor, ",", false, &array_topo));
        KALDI_ASSERT(array_topo.size() >= 2);
        KALDI_ASSERT(samp_rate);
        std::ostringstream oss;
        std::copy(array_topo.begin(), array_topo.end(), std::ostream_iterator<BaseFloat>(oss, " "));
        KALDI_VLOG(1) << "Parse topo_descriptor(" << topo_descriptor << ") to " << oss.str();
    }
};

class SrpPhatComputor {

public:
    SrpPhatComputor(const SrpPhatOptions &opts, 
                    BaseFloat freq, int32 num_bins): 
        samp_frequency_(freq), opts_(opts) {
            opts_.ComputeDerived();
            frequency_axis_.Resize(num_bins);
            delay_axis_.Resize(opts_.samp_rate);
            idtft_coef_.Resize(num_bins, opts_.samp_rate);
            exp_idtft_coef_j_.Resize(num_bins, opts_.samp_rate);
            for (int32 f = 0; f < num_bins; f++) 
                frequency_axis_(f) = f * samp_frequency_ / ((num_bins - 1) * 2);
        }

    void Compute(const CMatrixBase<BaseFloat> &stft, 
                 Matrix<BaseFloat> *spectra);

    int32 NumChannels() { return opts_.array_topo.size(); }

private:
    SrpPhatOptions opts_;
    // sample frequency of wave
    BaseFloat samp_frequency_;

    Vector<BaseFloat> frequency_axis_, delay_axis_;
    // note that exp_idtft_coef_j_ = exp(idtft_coef_)
    Matrix<BaseFloat> idtft_coef_;
    CMatrix<BaseFloat> exp_idtft_coef_j_;
    
    // This function implements GCC-PHAT algorithm. For MATLAB:
    // >> R = L .* conj(R) ./ (abs(L) .* abs(R));
    // >> frequency = linspace(0, fs / 2, num_bins);
    // >> augular = R * (exp(frequency' * tau * 2j * pi));
    void ComputeGccPhat(const CMatrixBase<BaseFloat> &L,
                        const CMatrixBase<BaseFloat> &R,
                        BaseFloat dist,
                        CMatrixBase<BaseFloat> *spectra);

    void Smooth(CMatrix<BaseFloat> *spectra);
};


}

#endif
