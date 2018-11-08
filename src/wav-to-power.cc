// src/wav-to-power.cc

// Copyright 2018  Jian Wu

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

#include "base/kaldi-common.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Read wav files and output an archive consisting of a single float:\n"
        "the power of each one in dB.\n"
        "Usage:  wav-to-power [options...] <wav-rspecifier> "
        "<power-wspecifier>\n"
        "E.g.: wav-to-power scp:wav.scp ark,t:-\n"
        "See also: wav-copy extract-segments feat-to-len\n"
        "Currently this program may output a lot of harmless warnings "
        "regarding\n"
        "nonzero exit status of pipes\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1), power_wspecifier = po.GetArg(2);

    double sum_power = 0.0,
           min_power = std::numeric_limits<BaseFloat>::infinity(),
           max_power = 0;
    int32 num_done = 0;

    BaseFloatWriter power_writer(power_wspecifier);
    SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
    for (; !wav_reader.Done(); wav_reader.Next()) {
      std::string key = wav_reader.Key();
      const WaveData &wave_data = wav_reader.Value();
      const Matrix<BaseFloat> &data = wave_data.Data();

      BaseFloat power = VecVec(data.Row(0), data.Row(0)) / data.Row(0).Dim();
      BaseFloat power_db = 10 * log10(power);

      power_writer.Write(key, power_db);

      sum_power += power_db;
      min_power = std::min<double>(min_power, power_db);
      max_power = std::max<double>(max_power, power_db);
      num_done++;
    }

    KALDI_LOG << "Printed power for " << num_done << " audio files.";
    if (num_done > 0) {
      KALDI_LOG << "Mean power was " << (sum_power / num_done)
                << ", min and max power were " << min_power << ", "
                << max_power;
    }
    return (num_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
