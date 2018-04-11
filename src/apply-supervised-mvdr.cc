//
// apply-supervised-mvdr.cc
// wujian@2018
//

#include "feat/wave-reader.h"
#include "include/stft.h"
#include "include/beamformer.h"

using namespace kaldi;

int main(int argc, char *argv[]) {
    try{
        const char *usage = 
            "Do minimum variance distortionless response (MVDR) beamformer, depending on TF mask\n"
            "\n"
            "Usage: apply-supervised-mvdr [options...] <mask-rspecifier> <ch1-rspecifier> ... <target-wav-wspecifier>\n";

        ParseOptions po(usage);
        ShortTimeFTOptions stft_options;

        bool track_volumn = true, normalize_input = true;
        std::string window = "hamming";
        BaseFloat frame_shift = 256, frame_length = 1024;

        po.Register("track-volumn", &track_volumn, "If true, keep targets' volumn same as orginal wave files");
        po.Register("frame-shift", &frame_shift, "Frame shift in number of samples");
        po.Register("frame-length", &frame_length, "Frame length in number of samples");
        po.Register("window", &window, "Type of window(\"hamming\"|\"hanning\"|\"blackman\"|\"rectangular\")");
        po.Register("normalize-input", &normalize_input, "Scale samples into range [-1, 1], like MATLAB or librosa");

        po.Read(argc, argv);

        int32 num_args = po.NumArgs();

        if (num_args <= 3) {
            po.PrintUsage();
            exit(1);
        }

        std::string mask_rspecifier = po.GetArg(1), enhan_wspecifier = po.GetArg(num_args);
        int32 num_channels = num_args - 2;
    
        // Construct wave reader.
        std::vector<RandomAccessTableReader<WaveHolder> > wav_reader(num_channels);
        for (int32 i = 2; i < num_args; i++) {
            std::string cur_ch = po.GetArg(i);
            if (ClassifyRspecifier(cur_ch, NULL, NULL) == kNoRspecifier)
                KALDI_ERR << cur_ch << " is not a rspecifier";
            KALDI_ASSERT(wav_reader[i - 2].Open(cur_ch));
        }
    
        // config stft options
        stft_options.window = window;
        stft_options.normalize_input = normalize_input;
        stft_options.frame_shift = frame_shift;
        stft_options.frame_length = frame_length;

        ShortTimeFTComputer stft_computer(stft_options);

        SequentialBaseFloatMatrixReader mask_reader(mask_rspecifier);
        TableWriter<WaveHolder> wav_writer(enhan_wspecifier);

        int32 num_done = 0, num_miss = 0, num_utts = 0;

        for (; !mask_reader.Done(); mask_reader.Next()) {
            std::string utt_key = mask_reader.Key();
            const Matrix<BaseFloat> &target_mask = mask_reader.Value();

            // init num_channels
            // mstft: cache for realfft of each channel
            std::vector<Matrix<BaseFloat> > mstft(num_channels);
            std::vector<BaseFloat> mfreq(num_channels);
            BaseFloat range = 0.0;
            num_utts++;

            int32 cur_ch = 0;
            for (int32 c = 0; c < num_channels; c++) {
                if (wav_reader[c].HasKey(utt_key)) {
                    const WaveData &wave_data = wav_reader[c].Value(utt_key);
                    const Matrix<BaseFloat> &wave_samp = wave_data.Data(); 
                    if (track_volumn)
                        range += wave_samp.LargestAbsElem();
                    mfreq[cur_ch] = wave_data.SampFreq();
                    stft_computer.Compute(wave_samp, &mstft[cur_ch], NULL, NULL);
                    cur_ch++;
                }
            }
            KALDI_VLOG(2) << "Processing " << cur_ch << " channels for " << utt_key;
            // do not process if num_channels <= 1
            if (cur_ch <= 1) {
                num_miss++;
                continue;
            }
            
            // check dimentions
            // mstft[..].NumCols() == frame_length
            int32 num_frames = mstft[0].NumRows(), num_bins = mstft[0].NumCols() / 2 + 1;
            BaseFloat target_freq = mfreq[0];

            bool problem = false;
            for (int32 c = 1; c < cur_ch; c++) {
                if (mstft[c].NumCols() != (num_bins - 1) * 2 || mstft[c].NumRows() != num_frames) {
                    KALDI_WARN << "There is obvious length difference between"
                        << "multiple channels, please check, skip for " << utt_key;
                    problem = true;
                    break;
                }
                if (target_freq != mfreq[c]) {
                    KALDI_WARN << "Sample frequency may be difference between"
                        << "multiple channels, please check, skip for " << utt_key;
                    problem = true;
                    break;
                }
            }
            if (problem) {
                num_miss++;
                continue;
            }
            
            // target_mask is a real matrix
            if (target_mask.NumRows() != num_frames || target_mask.NumCols() != num_bins) {
                KALDI_WARN << "Utterance " << utt_key << ": The shape of target mask is different from stft"
                           << " (" << target_mask.NumRows() << " x " << target_mask.NumCols() << ") vs"
                           << " (" << num_frames << " x " << num_bins << ")";
                num_miss++;
                continue;
            }
            
            CMatrix<BaseFloat> stft_reshape(num_frames, num_bins * cur_ch), src_stft;
            for (int32 c = 0; c < cur_ch; c++) {
                stft_reshape.ColRange(c * num_bins, num_bins).CopyFromRealfft(mstft[c]);
            }
            ReshapeMultipleStft(num_bins, cur_ch, stft_reshape, &src_stft); 

            CMatrix<BaseFloat> noise_psd, target_psd, steer_vector, beam_weights, enh_stft; 

            EstimatePsd(src_stft, target_mask, &target_psd, &noise_psd);
            EstimateSteerVector(target_psd, &steer_vector);
            ComputeMvdrBeamWeights(noise_psd, steer_vector, &beam_weights);
            Beamform(src_stft, beam_weights, &enh_stft);
            
            Matrix<BaseFloat> rstft, enhan_speech;
            CastIntoRealfft(enh_stft, &rstft);
            stft_computer.InverseShortTimeFT(rstft, &enhan_speech, range / cur_ch);

            WaveData target_data(target_freq, enhan_speech);
            wav_writer.Write(utt_key, target_data);
            num_done++;

            if (num_done % 100 == 0)
                KALDI_LOG << "Processed " << num_utts << " utterances.";
            KALDI_VLOG(2) << "Do mvdr beamforming for utterance-id " << utt_key << " done.";
        }

        KALDI_LOG << "Done " << num_done << " utterances out of " << num_utts
                  << ", " << num_miss << " missing cause of some problems.";

        return num_done == 0 ? 1: 0;

    } catch(const std::exception& e) {
        std::cerr << e.what();
        return -1;
    }
    return 0; 
}
