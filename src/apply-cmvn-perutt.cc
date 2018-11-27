// featbin/apply-cmvn.cc

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

#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"
#include "transform/cmvn.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Apply cepstral mean and (optionally) variance normalization for each "
        "utterance\n"
        "Usage: apply-cmvn-perutt [options] <feats-rspecifier> "
        "<feats-wspecifier>\n"
        "e.g.: copy-matrix --apply-log scp:data/train/feats.scp ark:- | "
        "apply-cmvn-perutt ark:- ark:-\n"
        "See also: apply-cmvn\n";

    ParseOptions po(usage);
    bool norm_vars = false;
    bool norm_means = true;
    std::string skip_dims_str, cmvn_rxfilename = "";

    po.Register("norm-vars", &norm_vars, "If true, normalize variances.");
    po.Register("norm-means", &norm_means, "If true, normalize means.");
    po.Register("gcmvn", &cmvn_rxfilename,
                "If assigned, do global mean and variance normalization.");
    po.Register(
        "skip-dims", &skip_dims_str,
        "Dimensions for which to skip "
        "normalization: colon-separated list of integers, e.g. 13:14:15)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    if (norm_vars && !norm_means)
      KALDI_ERR << "You cannot normalize the variance but not the mean.";

    std::string feat_rspecifier = po.GetArg(1);
    std::string feat_wspecifier = po.GetArg(2);

    if (!norm_means) {
      // CMVN is a no-op, we're not doing anything.  Just echo the input
      // don't even uncompress, if it was a CompressedMatrix.
      SequentialGeneralMatrixReader reader(feat_rspecifier);
      GeneralMatrixWriter writer(feat_wspecifier);
      kaldi::int32 num_done = 0;
      for (; !reader.Done(); reader.Next()) {
        writer.Write(reader.Key(), reader.Value());
        num_done++;
      }
      KALDI_LOG << "Copied " << num_done << " utterances.";
      return (num_done != 0 ? 0 : 1);
    }

    std::vector<int32> skip_dims;  // optionally use "fake"
                                   // (zero-mean/unit-variance) stats for some
                                   // dims to disable normalization.
    if (!SplitStringToIntegers(skip_dims_str, ":", false, &skip_dims)) {
      KALDI_ERR << "Bad --skip-dims option (should be colon-separated list of "
                << "integers)";
    }

    kaldi::int32 num_done = 0, num_err = 0;

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    if (cmvn_rxfilename == "") {
      for (; !feat_reader.Done(); feat_reader.Next()) {
        std::string utt = feat_reader.Key();
        Matrix<BaseFloat> feat(feat_reader.Value());
        if (norm_means) {
          Matrix<double> cmvn_stats;
          InitCmvnStats(feat.NumCols(), &cmvn_stats);
          AccCmvnStats(feat, NULL, &cmvn_stats);
          if (!skip_dims.empty()) FakeStatsForSomeDims(skip_dims, &cmvn_stats);

          ApplyCmvn(cmvn_stats, norm_vars, &feat);
          feat_writer.Write(utt, feat);
        } else {
          feat_writer.Write(utt, feat);
        }
        num_done++;
      }
    } else {
      bool binary;
      Input ki(cmvn_rxfilename, &binary);
      Matrix<double> cmvn_stats;
      cmvn_stats.Read(ki.Stream(), binary);
      if (!skip_dims.empty()) FakeStatsForSomeDims(skip_dims, &cmvn_stats);

      for (; !feat_reader.Done(); feat_reader.Next()) {
        std::string utt = feat_reader.Key();
        Matrix<BaseFloat> feat(feat_reader.Value());
        if (norm_means) {
          ApplyCmvn(cmvn_stats, norm_vars, &feat);
        }
        feat_writer.Write(utt, feat);
        num_done++;
      }
    }
    if (norm_vars)
      KALDI_LOG << "Applied cepstral mean and variance normalization to "
                << num_done << " utterances, errors on " << num_err;
    else
      KALDI_LOG << "Applied cepstral mean normalization to " << num_done
                << " utterances, errors on " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
