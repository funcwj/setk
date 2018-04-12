//
// apply-supervised-max-snr.cc
// wujian@2018
//

#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "include/stft.h"
#include "include/beamformer.h"

using namespace kaldi;

void ParseInputRspecifier(std::string &input_rspecifier, 
                          std::vector<std::string> *rspecifiers) {
    size_t found = input_rspecifier.find_first_of(":", 0);
    if (found == std::string::npos)
        KALDI_ERR << "Wrong input-rspecifier format: " << input_rspecifier;
    const std::string &decorator = input_rspecifier.substr(0, found);

    std::vector<std::string> tmp;
    SplitStringToVector(input_rspecifier.substr(found + 1), ",", false, &tmp);
    for (std::string &s: tmp)
        rspecifiers->push_back(decorator + ":" + s);
}

int main(int argc, char *argv[]) {
    try{
        const char *usage = 
            "Do max-snr (using generalized eigenvector decomposition method) beamformer, depending on TF mask\n"
            "\n"
            "Usage: apply-supervised-max-snr [options...] <mask-rspecifier> <input-rspecifier> <target-wav-wspecifier>\n"
            "\n"
            "e.g.:\n"
            " apply-supervised-max-snr --config=mask.conf scp:mask.scp scp:CH1.scp,CH2.scp,CH3.scp scp:dst.scp\n";

        ParseOptions po(usage);
        ShortTimeFTOptions stft_options;

        bool track_volumn = true, normalize_input = true;
        std::string window = "hamming";
        BaseFloat frame_shift = 256, frame_length = 1024;
        int32 update_periods = 0, minimum_update_periods = 20;

        po.Register("frame-shift", &frame_shift, "Frame shift in number of samples");
        po.Register("frame-length", &frame_length, "Frame length in number of samples");
        po.Register("window", &window, "Type of window(\"hamming\"|\"hanning\"|\"blackman\"|\"rectangular\")");
        po.Register("track-volumn", &track_volumn, 
                    "If true, using average volumn of input channels as target's");
        po.Register("normalize-input", &normalize_input, 
                    "Scale samples into float in range [-1, 1], like MATLAB or librosa");
        po.Register("update-periods", &update_periods, 
                    "Number of frames to use for estimating psd of noise or target, "
                    "if zero, do beamforming offline");

        po.Read(argc, argv);

        int32 num_args = po.NumArgs();

        if (num_args != 3) {
            po.PrintUsage();
            exit(1);
        }

        KALDI_ASSERT(update_periods >= 0);
        if (update_periods < minimum_update_periods && update_periods > 0) {
            KALDI_WARN << "Value of update_periods may be too small, ignore it";
        }

        std::string mask_rspecifier = po.GetArg(1), input_rspecifier = po.GetArg(2),
                    enhan_wspecifier = po.GetArg(3);
        
        std::vector<std::string> rspecifiers;
        ParseInputRspecifier(input_rspecifier, &rspecifiers);

        int32 num_channels = rspecifiers.size();
    
        // Construct wave reader.
        std::vector<RandomAccessTableReader<WaveHolder> > wav_reader(num_channels);
        for (int32 c = 0; c < num_channels; c++) {
            std::string &cur_ch = rspecifiers[c];
            if (ClassifyRspecifier(cur_ch, NULL, NULL) == kNoRspecifier)
                KALDI_ERR << cur_ch << " is not a rspecifier";
            KALDI_ASSERT(wav_reader[c].Open(cur_ch));
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

            CMatrix<BaseFloat> noise_psd, target_psd, beam_weights, enh_stft; 
            int32 num_segments = (num_frames - minimum_update_periods) / update_periods + 1;

            if (update_periods >= minimum_update_periods && num_segments > 1) {
                KALDI_VLOG(1) << "Do max-snr beamforming, update power spectrum matrix estimation per " << update_periods << " frames";
                int32 duration = 0, start_from = 0;
                CMatrix<BaseFloat> enh_stft_segment;
                enh_stft.Resize(num_frames, num_bins);
                for (int32 i = 0; i < num_segments; i++) {
                    start_from = i * update_periods;
                    duration = (i == num_segments - 1 ? num_frames - start_from: update_periods); 
                    ReshapeMultipleStft(num_bins, cur_ch, stft_reshape.RowRange(start_from, duration), &src_stft); 
                    EstimatePsd(src_stft, target_mask.RowRange(start_from, duration), &target_psd, &noise_psd);
                    ComputeGevdBeamWeights(target_psd, noise_psd, &beam_weights);
                    Beamform(src_stft, beam_weights, &enh_stft_segment);
                    enh_stft.RowRange(start_from, duration).CopyFromMat(enh_stft_segment);
                }

            } else {
                KALDI_VLOG(1) << "Do max-snr beamforming offline";
                ReshapeMultipleStft(num_bins, cur_ch, stft_reshape, &src_stft); 
                EstimatePsd(src_stft, target_mask, &target_psd, &noise_psd);
                ComputeGevdBeamWeights(target_psd, noise_psd, &beam_weights);
                Beamform(src_stft, beam_weights, &enh_stft);
            }

            Matrix<BaseFloat> rstft, enhan_speech;
            CastIntoRealfft(enh_stft, &rstft);
            stft_computer.InverseShortTimeFT(rstft, &enhan_speech, range / cur_ch);

            WaveData target_data(target_freq, enhan_speech);
            wav_writer.Write(utt_key, target_data);
            num_done++;

            if (num_done % 100 == 0)
                KALDI_LOG << "Processed " << num_utts << " utterances.";
            KALDI_VLOG(2) << "Do max-snr beamforming for utterance-id " << utt_key << " done.";
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
