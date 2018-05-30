//
// wav-seperate.cc
// wujian@2018
//

#include "include/stft.h"

using namespace kaldi;

void SeperateSpeech(ShortTimeFTComputer &stft_computer,
                    const MatrixBase<BaseFloat> &noisy_data, 
                    const MatrixBase<BaseFloat> &target_mask, 
                    Matrix<BaseFloat>  *target_speech, 
                    bool track_volumn) {
    Matrix<BaseFloat> specs, args;
    stft_computer.Compute(noisy_data, NULL, &specs, &args);
    KALDI_ASSERT(SameDim(specs, target_mask));
    // here need raw spectrum(means no power & log) to make sure mask work properly
    specs.MulElements(target_mask);
    
    Matrix<BaseFloat> target_stft; 
    stft_computer.Polar(specs, args, &target_stft);   
    if (track_volumn) {
        BaseFloat range = noisy_data.LargestAbsElem();
        stft_computer.InverseShortTimeFT(target_stft, target_speech, range);
    } else {
        stft_computer.InverseShortTimeFT(target_stft, target_speech);
    }
}

int main(int argc, char *argv[]) {
    try{
        const char *usage = 
            "Seperate target component of wave file based on TF mask approach\n"
            "Usage:  wav-seperate [options...] <wav-rspecifier> <mask-rspecifier> <target-wav-wspecifier>\n"
            "   or:  wav-seperate [options...] <wav-rxfilename> <mask-rxfilename> <target-wav-wxfilename>\n";

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

        std::string noisy_in = po.GetArg(1), mask_in = po.GetArg(2), target_out = po.GetArg(3);
        
        bool noisy_is_rspecifier = (ClassifyRspecifier(noisy_in, NULL, NULL) != kNoRspecifier),
             mask_is_rspecifier = (ClassifyRspecifier(mask_in, NULL, NULL) != kNoRspecifier),
             target_is_wspecifier = (ClassifyWspecifier(target_out, NULL, NULL, NULL) != kNoWspecifier);

        if (noisy_is_rspecifier != target_is_wspecifier)
            KALDI_ERR << "Cannot mix archives with regular files";

        if (noisy_is_rspecifier != mask_is_rspecifier)
            KALDI_ERR << "Configure with noisy file and target mask must keep same";
        
        // when reconstruct waveform, just need magnitude spectrum
        stft_options.apply_log = false;
        stft_options.apply_pow = false;

        ShortTimeFTComputer stft_computer(stft_options);

        if (noisy_is_rspecifier) {
            SequentialTableReader<WaveHolder> noisy_reader(noisy_in);
            RandomAccessBaseFloatMatrixReader mask_reader(mask_in);
            TableWriter<WaveHolder> wav_writer(target_out);

            int num_utts = 0, num_no_tgt_utts = 0, num_done = 0;
            for (; !noisy_reader.Done(); noisy_reader.Next()) {
                std::string utt_key = noisy_reader.Key();
                num_utts++;

                if (!mask_reader.HasKey(utt_key)) {
                    KALDI_WARN << utt_key << ", missing target masks";
                    num_no_tgt_utts++;
                    continue;
                }

                const WaveData &noisy_data = noisy_reader.Value();
                BaseFloat target_freq = noisy_data.SampFreq();
                KALDI_ASSERT(noisy_data.Data().NumRows() == 1);

                const Matrix<BaseFloat> &target_mask = mask_reader.Value(utt_key);
                Matrix<BaseFloat> target_speech;
                SeperateSpeech(stft_computer, noisy_data.Data(), target_mask, &target_speech, track_volumn);

                WaveData target_data(target_freq, target_speech);
                wav_writer.Write(utt_key, target_data);
                num_done++;

                if (num_done % 100 == 0)
                    KALDI_LOG << "Processed " << num_utts << " utterances";
                KALDI_VLOG(2) << "Seperate target for utterance " << utt_key;
            }
            KALDI_LOG << "Done " << num_done << " utterances out of " << num_utts
                      << ", " << num_no_tgt_utts << " missing targets masks";
            return num_done == 0 ? 1: 0;

        } else {
            bool binary;
            Input ki(noisy_in, &binary);
            Matrix<BaseFloat> target_mask;
            ReadKaldiObject(mask_in, &target_mask);

            WaveData noisy_data;
            noisy_data.Read(ki.Stream());
            BaseFloat target_freq = noisy_data.SampFreq();
            KALDI_ASSERT(noisy_data.Data().NumRows() == 1);

            Matrix<BaseFloat> target_speech;
            SeperateSpeech(stft_computer, noisy_data.Data(), target_mask, &target_speech, track_volumn);

            Output ko(target_out, binary, false);
            WaveData target_data(target_freq, target_speech);
            target_data.Write(ko.Stream());

            KALDI_LOG << "Done processed " << noisy_in;
        }


    } catch(const std::exception& e) {
        std::cerr << e.what();
        return -1;
    }
    return 0; 
}
