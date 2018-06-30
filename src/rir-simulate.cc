//
// rir-simulate.cc
// wujian@2018.6.26
//

#include "feat/wave-reader.h"
#include "include/rir-generator.h"


int main(int argc, char const *argv[]) {
    try {
        using namespace kaldi;

        const char* usage = 
            "Computes the response of an acoustic source to one or more microphones "
            "in a reverberant room using the image method.\n"
            "Reference: https://github.com/ehabets/RIR-Generator\n"
            "\n"
            "Usage: rir-simulate [options] <wav-wspecifier>\n"
            "See also: wav-reverberate\n";

        ParseOptions po(usage);

        bool report = false, normalize = false;
        po.Register("report", &report, "If true, output RirGenerator's statistics");
        po.Register("normalize", &normalize, "If true, normalize output room impluse response");


        RirGeneratorOptions generator_opts;
        generator_opts.Register(&po);

        po.Read(argc, argv);
        
        if (po.NumArgs() != 1) {
            po.PrintUsage();
            exit(1);
        }

        RirGenerator generator(generator_opts);
        Matrix<BaseFloat> rir;
        BaseFloat int16_max = static_cast<BaseFloat>(std::numeric_limits<int16>::max());

        generator.GenerateRir(&rir);

        if (normalize) {
            rir.Scale(1.0 / rir.LargestAbsElem());
        }
        rir.Scale(int16_max);

        if (report)
            std::cout << generator.Report();

        std::string target_rir = po.GetArg(1);
        Output ko(target_rir, true, false);
        WaveData rir_simu(generator.Frequency(), rir);
        rir_simu.Write(ko.Stream());

    
    } catch(const std::exception& e) {
        std::cerr << e.what();
        return -1;
    }
    return 0; 
}

