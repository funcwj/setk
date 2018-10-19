// src/apply-fixed-beamformer.cc
// wujian@2018.5.29
// Test pass.

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

#include "include/stft.h"
#include "include/beamformer.h"

using namespace kaldi;

BaseFloat DoBeamforming(ShortTimeFTComputer &stft_computer,
                        const Matrix<BaseFloat> &data,
                        const CMatrix<BaseFloat> &weight, int32 num_bins,
                        int32 num_chs, Matrix<BaseFloat> *enh_rstft) {
  Matrix<BaseFloat> rstft;
  stft_computer.Compute(data, &rstft, NULL, NULL);

  KALDI_ASSERT(rstft.NumCols() == (num_bins - 1) * 2);
  int32 num_frames = rstft.NumRows() / num_chs;

  CMatrix<BaseFloat> cstft(num_frames * num_chs, num_bins), src_stft, enh_cstft;
  cstft.CopyFromRealfft(rstft);

  BaseFloat range = 0;
  for (int32 c = 0; c < num_chs; c++)
    range += data.RowRange(c, 1).LargestAbsElem();

  TrimStft(num_bins, num_chs, cstft, &src_stft);
  Beamform(src_stft, weight, &enh_cstft);
  CastIntoRealfft(enh_cstft, enh_rstft);
  return range;
}

int main(int argc, char *argv[]) {
  try {
    const char *usage =
        "Apply fixed beamformer on input wave files. To use this command, \n"
        "you need to pre-design/compute beam weights according to array's "
        "topology and other prior infomation, such as DoA\n"
        "It's designed for DS(delay and sum) or superdirective beamformer\n"
        "\n"
        "Usage: apply-fixed-beamformer [options...] <wav-rspecifier> "
        "<complex-mat-rxfilename> <wav-wspecifier>\n"
        "or   : apply-fixed-beamformer [options...] <wav-rxfilename> "
        "<complex-mat-rxfilename> <wav-wxfilename>\n"
        "e.g:\n"
        "   apply-fixed-beamformer 4ch.wav weight.cmat enhan.wav\n";

    ParseOptions po(usage);
    ShortTimeFTOptions stft_options;

    bool track_volumn = true, normalize_output = false;

    po.Register("track-volumn", &track_volumn,
                "If true, set target's volumn as average of input channels'");
    po.Register("normalize-output", &normalize_output,
                "If true, normalize enhanced samples when write files");

    stft_options.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    if (track_volumn && normalize_output)
      KALDI_ERR << "Options --track-volumn conflict with --normalize-output, "
                << "setting one of them true, or both false";

    std::string chs_in = po.GetArg(1), enhan_out = po.GetArg(3);

    bool in_is_rspecifier =
             (ClassifyRspecifier(chs_in, NULL, NULL) != kNoRspecifier),
         out_is_wspecifier =
             (ClassifyWspecifier(enhan_out, NULL, NULL, NULL) != kNoWspecifier);

    if (in_is_rspecifier != out_is_wspecifier)
      KALDI_ERR << "Cannot mix archives with regular files";

    CMatrix<BaseFloat> beam_weight;
    ReadKaldiObject(po.GetArg(2), &beam_weight);
    int32 num_bins = beam_weight.NumRows(), num_chs = beam_weight.NumCols();
    ShortTimeFTComputer stft_computer(stft_options);

    if (in_is_rspecifier) {
      SequentialTableReader<WaveHolder> wave_reader(chs_in);
      TableWriter<WaveHolder> wav_writer(enhan_out);

      int num_utts = 0;
      for (; !wave_reader.Done(); wave_reader.Next()) {
        std::string utt_key = wave_reader.Key();
        const WaveData &wave_data = wave_reader.Value();
        BaseFloat target_freq = wave_data.SampFreq();

        if (wave_data.Data().NumRows() != num_chs)
          KALDI_ERR << "Input weight designed for " << num_chs
                    << " channels, but utterance " << utt_key << " has "
                    << wave_data.Data().NumRows() << " channels";

        Matrix<BaseFloat> enh_rstft, enhan_speech;
        BaseFloat range =
            DoBeamforming(stft_computer, wave_data.Data(), beam_weight,
                          num_bins, num_chs, &enh_rstft);

        if (track_volumn) {
          range = range / num_chs - 1;
        } else if (normalize_output) {
          range = 0;
        } else {
          range = -1;  // keep what it is
        }
        stft_computer.InverseShortTimeFT(enh_rstft, &enhan_speech, range);

        WaveData enhan_wavedata(target_freq, enhan_speech);
        wav_writer.Write(utt_key, enhan_wavedata);

        num_utts += 1;
        if (num_utts % 100 == 0)
          KALDI_LOG << "Processed " << num_utts << " utterances";
        KALDI_VLOG(2) << "Processed features for key " << utt_key;
      }
      KALDI_LOG << "Done " << num_utts << " utterances";
      return num_utts == 0 ? 1 : 0;

    } else {
      bool binary;
      Input wave_in(chs_in, &binary);
      WaveData wave_data;
      wave_data.Read(wave_in.Stream());

      KALDI_ASSERT(num_chs == wave_data.Data().NumRows());
      Matrix<BaseFloat> enh_rstft, enhan_speech;
      BaseFloat range =
          DoBeamforming(stft_computer, wave_data.Data(), beam_weight, num_bins,
                        num_chs, &enh_rstft);
      if (track_volumn) {
        range = range / num_chs - 1;
      } else if (normalize_output) {
        range = 0;
      } else {
        range = -1;
      }
      stft_computer.InverseShortTimeFT(enh_rstft, &enhan_speech, range);

      WaveData enhan_wavedata(wave_data.SampFreq(), enhan_speech);
      Output ko(enhan_out, binary, false);
      enhan_wavedata.Write(ko.Stream());

      KALDI_LOG << "Done " << chs_in;
    }

  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
  return 0;
}