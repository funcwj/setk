//
// compute-stft-stats
// wujian@2018
//

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "include/stft.h"


using namespace kaldi;

void ComputeSTFTStats(ShortTimeFTComputer &stft_computer, 
                      const MatrixBase<BaseFloat> &wave_data,
                      std::string &output,
                      Matrix<BaseFloat> *feature) {
    if (output == "stft")
        stft_computer.Compute(wave_data, feature, NULL, NULL);
    if (output == "spectrum")
        stft_computer.Compute(wave_data, NULL, feature, NULL);
    if (output == "arg")
        stft_computer.Compute(wave_data, NULL, NULL, feature);
}

int main(int argc, char *argv[]) {
    try{
        const char *usage = 
            "Compute short-time fourier transform statictis(arg, spectrum, stft) of waveform"
            "(using for speech enhancement)\n"
            "Usage:  compute-stft-stats [options...] <wav-rspecifier> <feats-wspecifier>\n"
            "   or:  compute-stft-stats [options...] <wav-rxfilename> <feats-wxfilename>\n";

        ParseOptions po(usage);
        ShortTimeFTOptions stft_options;

        std::string output = "spectrum";
        bool wx_binary = false;

        po.Register("output", &output, "Type(\"stft\"|\"arg\"|\"spectrum\") of stft for output");
        po.Register("binary", &wx_binary, "Write in binary mode (only relevant if output is a wxfilename)");

        stft_options.Register(&po);

        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }

        if (output != "spectrum" && output != "arg" && output != "stft")
            KALDI_ERR << "Unknown arguments for --output: " << output;

        std::string wave_in = po.GetArg(1), stft_out = po.GetArg(2);
        
        bool in_is_rspecifier = (ClassifyRspecifier(wave_in, NULL, NULL) != kNoRspecifier),
        out_is_wspecifier = (ClassifyWspecifier(stft_out, NULL, NULL, NULL) != kNoWspecifier);

        if (in_is_rspecifier != out_is_wspecifier)
            KALDI_ERR << "Cannot mix archives with regular files";

        ShortTimeFTComputer stft_computer(stft_options);

        if (in_is_rspecifier) {
            SequentialTableReader<WaveHolder> wave_reader(wave_in);

            BaseFloatMatrixWriter kaldi_writer;
            if (!kaldi_writer.Open(stft_out)) {
                KALDI_ERR << "Could not initialize output with wspecifier " << stft_out;
            }
            
            int num_utts = 0;
            for (; !wave_reader.Done(); wave_reader.Next()) {
                std::string utt_key = wave_reader.Key();
                const WaveData &wave_data = wave_reader.Value();
                
                if (wave_data.Data().NumRows() != 1) 
                    KALDI_WARN << utt_key << ": MULTI-CHANNEL!";

                Matrix<BaseFloat> feature;
                ComputeSTFTStats(stft_computer, wave_data.Data(), output, &feature);

                kaldi_writer.Write(utt_key, feature);

                num_utts += 1;
                if (num_utts % 100 == 0)
                    KALDI_LOG << "Processed " << num_utts << " utterances";
                KALDI_VLOG(2) << "Processed features for key " << utt_key;
            }
            KALDI_LOG << "Done " << num_utts << " utterances";
            return num_utts == 0 ? 1: 0;
        } else {
            bool binary;
            Input ki(wave_in, &binary);
            WaveData wave_input;
            wave_input.Read(ki.Stream());
            if (wave_input.Data().NumRows() != 1) 
                    KALDI_WARN << "MULTI-CHANNEL input!";
            Matrix<BaseFloat> feature;
            ComputeSTFTStats(stft_computer, wave_input.Data(), output, &feature);
            WriteKaldiObject(feature, stft_out, wx_binary);
            KALDI_LOG << "Done processed " << wave_in;
        }

    } catch(const std::exception& e) {
        std::cerr << e.what();
        return -1;
    }
    return 0; 
}
