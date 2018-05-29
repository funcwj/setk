// srp-phat.h
// wujian@18.5.29

#include "include/stft.h"
#include "include/complex-base.h"
#include "include/complex-vector.h"
#include "include/complex-matrix.h"



namespace kaldi {

struct SrpPhatOptions {

    BaseFloat sound_speed, doa_resolution;
    int32 smooth_context;
    std::string topo_descriptor;
    std::vector<BaseFloat> array_topo;

    SrpComputeOptions(): sound_speed(340.4), 
        doa_resolution(180), smooth_context(0),
        topo_descriptor("") {} 

    void Register(OptionsItf *opts) {
        opts->Register("sound-speed", &sound_speed, "Speed of sound(or other kinds of wave)");
        opts->Register("doa-resolution", &doa_resolution, 
                    "The sample rate of DoA, for linear microarray, we sampled from 0 to \pi.");
        opts->Register("smooth-context", &smooth_context, "Context of frames used for spectrum smoothing");
        opts->Register("topo-descriptor", &topo_descriptor, 
                    "Description of microarray's topology, now only support linear array."
                    "Egs --topo-descriptor=0,0.3,0.6,0.9 described a ULA with element spacing equals 0.3");
        ComputeDerived();
    }

    void ComputeDerived() {
        KALDI_ASSERT(topo_descriptor != "");
        SplitStringToFloats(&topo_descriptor, ",", false, &array_topo);
        KALDI_ASSERT(array_topo.size() >= 2);
        KALDI_ASSERT(doa_resolution);
    }
};

class SrpPhatComputor {

public:
    SrpPhatComputor(const SrpPhatOptions &opts, 
                    BaseFloat freq, int32 num_bins): 
        samp_frequency_(freq), opts_(opts) {
            frequency_axis_.Resize(num_bins);
            delay_axis_.Resize(opts_.doa_resolution);
            for (int32 f = 0; f < num_bins; i++) 
                frequency_axis_[f] = f * samp_frequency_ / ((num_bins - 1) * 2);
        }

    void Compute(const CMatrixBase<BaseFloat> &stft, 
                 CMatrix<BaseFloat> *spectrum);

private:
    SrpComputeOptions opts_;
    // sample frequency of wave
    BaseFloat samp_frequency_;
    // linspace(0, fs / 2, num_bins)
    Matrix<BaseFloat> frequency_axis_;

    Matrix<BaseFloat> delay_axis_;

    // This function implements GCC-PHAT algorithm. For MATLAB:
    // >> R = L .* conj(R) ./ (abs(L) .* abs(R));
    // >> frequency = linspace(0, fs / 2, num_bins);
    // >> augular = R * (exp(frequency' * tau * 2j * pi));
    void ComputeAugularSpectrum(const CMatrixBase<BaseFloat> &L,
                                const CMatrixBase<BaseFloat> &R,
                                CMatrixBase<BaseFloat> *spectrum);
};


}