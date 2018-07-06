// src/compute-srp-phat.cc
// wujian@2018.5.29

// Copyright 2018 Jian Wu

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "include/srp-phat.h"
#include "include/stft.h"

using namespace kaldi;

void PostNormalize(Matrix<BaseFloat> *srp_phat, bool norm_srp, 
                   bool norm_time, bool norm_tdoa) {
    if (norm_srp) {
        BaseFloat max_abs = srp_phat->LargestAbsElem();
        srp_phat->Scale(1.0 / max_abs);
        return;
    }
    BaseFloat inf_norm;
    BaseFloat float_inf = static_cast<BaseFloat>(std::numeric_limits<BaseFloat>::infinity());

    if (norm_time) {
        for (int32 t = 0; t < srp_phat->NumRows(); t++) {
            inf_norm = srp_phat->Row(t).Norm(float_inf);
            srp_phat->Row(t).Scale(1.0 / inf_norm);
        }
    }
    if (norm_tdoa) {
        for (int32 d = 0; d < srp_phat->NumCols(); d++) {
            inf_norm = srp_phat->ColRange(d, 1).LargestAbsElem();
            srp_phat->ColRange(d, 1).Scale(1.0 / inf_norm);
        }
    }
}

int main(int argc, char *argv[]) {
    try {
        const char *usage = "Compute angular spectrum using SRP-PHAT methods.\n"
            "This command now only support linear microarray configures. see --topo-descriptor for details. And output\n"
            "angular spectrum, Y-axis represents tdoa(not DoA) and X-axis represents times\n"
            "\n"
            "Usage: compute-srp-phat [options...] <wav-rspecifier> <srp-wspecifier>\n"
            "   or: compute-srp-phat [options...] <wav-rxfilename> <srp-wxfilename>\n"
            "egs: compute-srp-phat scp:egs.scp ark,scp:srp_phat.ark,srp_phat.scp\n"
            "\n"
            "NOTE: This command needs multi-channel wave as input. Using \'sox -M\' to merge wave files if needed."
            "\n";


        ParseOptions po(usage);

        ShortTimeFTOptions stft_options;
        SrpPhatOptions srp_options;

        BaseFloat samp_frequency = 16000;
        bool binary = true, norm_srp = false;
        bool norm_time_axis = false, norm_tdoa_axis = false;

        po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");
        po.Register("samp-frequency", &samp_frequency, "Waveform data sample frequency (must match the waveform file, if specified there)");
        po.Register("srp-normalize", &norm_srp, "Normalize values when output srp-phat angular spectrum");
        po.Register("normalize-time-axis", &norm_time_axis, "Normalize values along time axis");
        po.Register("normalize-tdoa-axis", &norm_tdoa_axis, "Normalize values along tdoa axix");

        stft_options.Register(&po);
        srp_options.Register(&po);
        
        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }

        std::string chs_in = po.GetArg(1), srp_out = po.GetArg(2);

        bool in_is_rspecifier = (ClassifyRspecifier(chs_in, NULL, NULL) != kNoRspecifier),
             out_is_wspecifier = (ClassifyWspecifier(srp_out, NULL, NULL, NULL) != kNoWspecifier);

        if (in_is_rspecifier != out_is_wspecifier)
            KALDI_ERR << "Cannot mix archives with regular files";

        int32 num_bins = stft_options.PaddingLength() / 2 + 1;

        ShortTimeFTComputer stft_computer(stft_options);
        SrpPhatComputor srp_computor(srp_options, samp_frequency, num_bins);

        int32 config_num_chs = srp_computor.NumChannels();

        if (in_is_rspecifier) {
            
            SequentialTableReader<WaveHolder> wave_reader(chs_in);

            BaseFloatMatrixWriter kaldi_writer;
            if (!kaldi_writer.Open(srp_out))
                KALDI_ERR << "Could not initialize output with wspecifier " << srp_out;

            int num_utts = 0;
            for (; !wave_reader.Done(); wave_reader.Next()) {
                std::string utt_key = wave_reader.Key();
                const WaveData &wave_data = wave_reader.Value();

                Matrix<BaseFloat> rstft, srp_phat;
                // only compute stft in realfft format
                const Matrix<BaseFloat> &ch_data = wave_data.Data();
                if (ch_data.NumRows() != config_num_chs) {
                    KALDI_ERR << "num_channels config from topo-descriptor is " << config_num_chs 
                              << " while " << ch_data.NumRows() << " channels in utterance " << utt_key;
                }
                stft_computer.Compute(wave_data.Data(), &rstft, NULL, NULL);

                CMatrix<BaseFloat> cstft(rstft.NumRows(), num_bins);
                cstft.CopyFromRealfft(rstft);
                srp_computor.Compute(cstft, &srp_phat);
                
                // post-process
                PostNormalize(&srp_phat, norm_srp, norm_time_axis, norm_tdoa_axis);

                kaldi_writer.Write(utt_key, srp_phat);

                num_utts += 1;
                if (num_utts % 100 == 0)
                    KALDI_LOG << "Processed " << num_utts << " utterances";
                KALDI_VLOG(2) << "Processed features for key " << utt_key;

            }
            KALDI_LOG << "Done " << num_utts << " utterances";
            return num_utts == 0 ? 1: 0;

        } else {
            bool read_binary;
            Input wave_in(chs_in, &read_binary);        
            WaveData wave;
            wave.Read(wave_in.Stream());
            
            KALDI_ASSERT(config_num_chs == wave.Data().NumRows());

            Matrix<BaseFloat> rstft, srp_phat;
            // only compute stft in realfft format
            stft_computer.Compute(wave.Data(), &rstft, NULL, NULL);

            CMatrix<BaseFloat> cstft(rstft.NumRows(), num_bins);
            cstft.CopyFromRealfft(rstft);
            srp_computor.Compute(cstft, &srp_phat);

            // post-process
            PostNormalize(&srp_phat, norm_srp, norm_time_axis, norm_tdoa_axis);

            WriteKaldiObject(srp_phat, srp_out, binary); 
            KALDI_LOG << "Done processed " << chs_in;
            return 0;
        }
    
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
