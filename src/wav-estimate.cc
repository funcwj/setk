//
// wav-estimate.cc
// wujian@2018
//

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "include/stft.h"

using namespace kaldi;

void EstimateSpeech(ShortTimeFTComputer &stft_computer,
                    const MatrixBase<BaseFloat> &refer_data,
                    MatrixBase<BaseFloat> &spectrum, 
                    Matrix<BaseFloat>  *target_speech,
                    bool track_volumn) {
    Matrix<BaseFloat> refer_phase, target_stft;
    stft_computer.Compute(refer_data, NULL, NULL, &refer_phase);
    stft_computer.Polar(spectrum, refer_phase, &target_stft);   
    if (track_volumn) {
        BaseFloat range = refer_data.LargestAbsElem();
        stft_computer.InverseShortTimeFT(target_stft, target_speech, range);
    } else {
        stft_computer.InverseShortTimeFT(target_stft, target_speech);
    }
}

int main(int argc, char *argv[]) {
    try{
        const char *usage = 
            "Estimate speech from magnitude spectrum and reference(noisy) wave\n"
            "Usage:  wav-estimate [options...] <spectrum-rspecifier> <refer-wav-rspecifier> <target-wav-wspecifier>\n"
            "   or:  wav-estimate [options...] <spectrum-rxfilename> <refer-wav-rxfilename> <target-wav-wxfilename>\n";

        ParseOptions po(usage);
        ShortTimeFTOptions stft_options;

        bool track_volumn = true;
        po.Register("track-volumn", &track_volumn, "If true, keep targets' volumn same as orginal wave files");

        stft_options.Register(&po);

        po.Read(argc, argv);

        if (po.NumArgs() != 3) {
            po.PrintUsage();
            exit(1);
        }

        std::string spectrum_in = po.GetArg(1), refer_in = po.GetArg(2), target_out = po.GetArg(3);
        
        bool spectrum_is_rspecifier = (ClassifyRspecifier(spectrum_in, NULL, NULL) != kNoRspecifier),
             refer_is_rspecifier = (ClassifyRspecifier(refer_in, NULL, NULL) != kNoRspecifier),
             target_is_wspecifier = (ClassifyWspecifier(target_out, NULL, NULL, NULL) != kNoWspecifier);

        if (spectrum_is_rspecifier != target_is_wspecifier)
            KALDI_ERR << "Cannot mix archives with regular files";

        if (spectrum_is_rspecifier != refer_is_rspecifier)
            KALDI_ERR << "Configure with noisy file and target mask must keep same";
        
        ShortTimeFTComputer stft_computer(stft_options);

        if (spectrum_is_rspecifier) {
            SequentialBaseFloatMatrixReader spectrum_reader(spectrum_in);
            RandomAccessTableReader<WaveHolder> refer_reader(refer_in);
            TableWriter<WaveHolder> wav_writer(target_out);

            int num_utts = 0, num_no_tgt_utts = 0, num_done = 0;
            for (; !spectrum_reader.Done(); spectrum_reader.Next()) {
                std::string utt_key = spectrum_reader.Key();
                num_utts += 1;

                if (!refer_reader.HasKey(utt_key)) {
                    KALDI_WARN << utt_key << ", missing target masks";
                    num_no_tgt_utts++;
                    continue;
                }

                Matrix<BaseFloat> &spectrum = spectrum_reader.Value();
                const WaveData &refer_data = refer_reader.Value(utt_key);
                BaseFloat target_freq = refer_data.SampFreq();

                Matrix<BaseFloat> target_speech;
                EstimateSpeech(stft_computer, refer_data.Data(), spectrum, &target_speech, track_volumn);

                WaveData target_data(target_freq, target_speech);
                wav_writer.Write(utt_key, target_data);
                num_done++;

                if (num_done % 100 == 0)
                    KALDI_LOG << "Processed " << num_utts << " utterances";
                KALDI_VLOG(2) << "Estimate target for utterance " << utt_key;
            }
            KALDI_LOG << "Done " << num_utts << " utterances out of " << num_done
                      << ", " << num_no_tgt_utts << " missing targets masks";
        } else {
            Matrix<BaseFloat> spectrum;
            ReadKaldiObject(spectrum_in, &spectrum);

            bool binary;
            Input ki(refer_in, &binary);
            WaveData refer_data;
            refer_data.Read(ki.Stream());
            BaseFloat target_freq = refer_data.SampFreq();

            Matrix<BaseFloat> target_speech;
            EstimateSpeech(stft_computer, refer_data.Data(), spectrum, &target_speech, track_volumn);

            Output ko(target_out, binary, false);
            WaveData target_data(target_freq, target_speech);
            target_data.Write(ko.Stream());

            KALDI_LOG << "Done processed " << spectrum_in;
        }

        return 1;

    } catch(const std::exception& e) {
        std::cerr << e.what();
        return -1;
    }
    return 0; 
}
