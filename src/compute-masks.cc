//
// compute-masks
// wujian@2018
//

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "include/stft.h"

using namespace kaldi;

void ComputeMasks(ShortTimeFTComputer &stft_computer, 
                      const MatrixBase<BaseFloat> &noise_data,
                      const MatrixBase<BaseFloat> &clean_data,
                      std::string &type,
                      Matrix<BaseFloat> *mask) {

    Matrix<BaseFloat> numerator, denumerator;
    stft_computer.Compute(clean_data, NULL, &numerator, NULL);
    stft_computer.Compute(noise_data, NULL, &denumerator, NULL);

    if (type == "irm" || type == "wiener") {
        denumerator.AddMat(1, numerator);
        numerator.DivElements(denumerator);
    }

    // just a simple version
    if (type == "ibm") {
        numerator.AddMat(-1, denumerator);
        numerator.Heaviside(numerator);
    }
    mask->Swap(&numerator);
}

int main(int argc, char *argv[]) {
    try{
        const char *usage = 
            "Compute clean mask(using for speech enhancement)\n"
            "Usage:  compute-masks [options...] <noise-rspecifier> <clean-rspecifier> <mask-wspecifier>\n"
            "   or:  compute-masks [options...] <noise-rxfilename> <clean-rxfilename> <mask-wxfilename>\n"
            "\n"
            "By default, this command compute clean masks, to compute noise part, swap noise-* and clean-*\n";

        ParseOptions po(usage);
        ShortTimeFTOptions stft_options;

        bool wx_binary = false;
        std::string mask_type = "irm";

        po.Register("mask", &mask_type, "Type(\"irm\"|\"ibm\"|\"wiener\") of masks for output");
        po.Register("binary", &wx_binary, "Write in binary mode (only relevant if output is a wxfilename)");

        stft_options.Register(&po);

        po.Read(argc, argv);

        if (po.NumArgs() != 3) {
            po.PrintUsage();
            exit(1);
        }

        if (mask_type != "irm" && mask_type != "ibm" && mask_type != "wiener")
            KALDI_ERR << "Unknown arguments for --mask: " << mask_type;

        std::string noise_in = po.GetArg(1), clean_in = po.GetArg(2), mask_out = po.GetArg(3);
        
        bool noise_is_rspecifier = (ClassifyRspecifier(noise_in, NULL, NULL) != kNoRspecifier),
             clean_is_rspecifier = (ClassifyRspecifier(clean_in, NULL, NULL) != kNoRspecifier),
             mask_is_wspecifier = (ClassifyWspecifier(mask_out, NULL, NULL, NULL) != kNoWspecifier);

        if (noise_is_rspecifier != mask_is_wspecifier)
            KALDI_ERR << "Cannot mix archives with regular files";

        if (noise_is_rspecifier != clean_is_rspecifier)
            KALDI_ERR << "Configure with noise/clean must keep same";

        // config power
        if (mask_type == "wiener")
            stft_options.apply_pow = true;
        // compute mask do not need to apply log 
        stft_options.apply_log = false;
        ShortTimeFTComputer stft_computer(stft_options);

        if (noise_is_rspecifier) {
            SequentialTableReader<WaveHolder> noise_reader(noise_in);
            RandomAccessTableReader<WaveHolder> clean_reader(clean_in);

            BaseFloatMatrixWriter kaldi_writer;
            if (!kaldi_writer.Open(mask_out))
                KALDI_ERR << "Could not initialize output with wspecifier " << mask_out;
            
            int num_utts = 0, num_no_tgt_utts = 0, num_done = 0;
            for (; !noise_reader.Done(); noise_reader.Next()) {
                std::string utt_key = noise_reader.Key();
                num_utts += 1;

                if (!clean_reader.HasKey(utt_key)) {
                    KALDI_WARN << utt_key << ", missing clean";
                    num_no_tgt_utts++;
                    continue;
                }

                const WaveData &noise_data = noise_reader.Value(), 
                      &clean_data = clean_reader.Value(utt_key);
                    
                Matrix<BaseFloat> mask;
                ComputeMasks(stft_computer, noise_data.Data(), clean_data.Data(), mask_type, &mask);

                kaldi_writer.Write(utt_key, mask);
                num_done++;

                if (num_done % 100 == 0)
                    KALDI_LOG << "Processed " << num_utts << " utterances";
                KALDI_VLOG(2) << "Compute mask" << "(" << mask_type << ") for key " << utt_key;
            }
            KALDI_LOG << "Done " << num_utts << " utterances out of " << num_done
                      << ", " << num_no_tgt_utts << " missing targets";
        } else {
            bool binary;
            Input kn(noise_in, &binary);
            Input kt(clean_in, &binary);

            WaveData noise_data, clean_data;
            noise_data.Read(kn.Stream());
            clean_data.Read(kt.Stream());

            Matrix<BaseFloat> mask;
            ComputeMasks(stft_computer, noise_data.Data(), clean_data.Data(), mask_type, &mask);
            WriteKaldiObject(mask, mask_out, wx_binary);
            KALDI_LOG << "Done processed " << noise_in;
        }

        return 1;

    } catch(const std::exception& e) {
        std::cerr << e.what();
        return -1;
    }
    return 0; 
}
