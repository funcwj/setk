// test_stft.cc
// wujian@18.2.12

#include "matrix/matrix-lib.h"
#include "base/kaldi-common.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"
#include "include/stft.h"

using namespace kaldi;
    
BaseFloat float_inf = static_cast<BaseFloat>(std::numeric_limits<BaseFloat>::infinity());

void test_stft() {

    bool binary;
    Input wave_in("orig.wav", &binary);        
    WaveData wave_orig;
    wave_orig.Read(wave_in.Stream());
    
    // configs
    ShortTimeFTOptions opts;
    opts.frame_length = 1024;
    opts.normalize_input = false;


    ShortTimeFTComputer stft_computer(opts);

    Matrix<BaseFloat> specs;
    stft_computer.ShortTimeFT(wave_orig.Data(), &specs);

    Matrix<BaseFloat> recon;
    stft_computer.InverseShortTimeFT(specs, &recon);

    Output ko("copy.wav", binary, false);
    WaveData wave_copy(16000, recon);
    wave_copy.Write(ko.Stream());
    // std:: cout << vec << std::endl;
}

void test_realfft() {
    int32 dim = 16;

    Vector<BaseFloat> vec(dim);
    vec.SetRandn();
    std::cout << vec;
    RealFft(&vec, true);
    std::cout << vec;
    RealFft(&vec, false);
    vec.Scale(1.0 / dim);
    std::cout << vec;
}

void test_istft() {

    bool binary;
    Input wave_in("orig.wav", &binary);        
    WaveData wave_orig;
    wave_orig.Read(wave_in.Stream());
    
    // configs
    ShortTimeFTOptions opts;
    opts.frame_length = 1024;
    opts.frame_shift  = 256;
    opts.normalize_input = false;
    opts.apply_log    = true;
    opts.power        = true;


    ShortTimeFTComputer stft_computer(opts);

    Matrix<BaseFloat> stft_orig, specs, arg;
    stft_computer.Compute(wave_orig.Data(), &stft_orig, &specs, &arg);
    BaseFloat range = wave_orig.Data().LargestAbsElem();
    /*
    stft_computer.ShortTimeFT(wave_orig.Data(), &stft_orig);
    stft_computer.ComputeSpectrum(stft_orig, &specs);
    stft_computer.ComputeArg(stft_orig, &arg);
    */

    Matrix<BaseFloat> stft_recon;
    stft_computer.Polar(specs, arg, &stft_recon);  

    std::cout << stft_orig.Row(10);
    std::cout << stft_recon.Row(10);
    
    Matrix<BaseFloat> recon;
    stft_computer.InverseShortTimeFT(stft_recon, &recon, range);

    Output ko("copy.wav", binary, false);
    WaveData wave_copy(16000, recon);
    wave_copy.Write(ko.Stream());
    // std:: cout << vec << std::endl;
}

int main() {
    // test_istft();
    test_stft();
    return 0;
}
