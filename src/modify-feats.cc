// src/modify-feats.cc

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

#include <algorithm>
#include <iterator>
#include <sstream>
#include <utility>

#include "base/kaldi-common.h"
#include "matrix/kaldi-matrix.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;

    const char *usage =
        "Do some common modification on input features.\n"
        "Usage: modify-feats --operator=average <in-rspecifier> "
        "<out-wspecifier>\n"
        "See also copy-feats, copy-matrix\n";

    ParseOptions po(usage);

    bool output_vector = false;
    std::string op = "average";
    int32 n = -1;

    po.Register("output-vector", &output_vector,
                "If true, output in vector instead of matrix if possible, egs, "
                "when \"--operator=average\"");
    po.Register("n", &n, "Value of index when operator == \'index\'");
    po.Register("operator", &op,
                "Operation on input features: "
                "\"average\"|\"sum\"|\"sample\"|\"index\"");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    if (op != "average" && op != "sum" && op != "sample" && op != "index") {
      KALDI_ERR << "Unknown operator: " << op;
    }

    // srand seed
    if (op == "sample") srand(time(0));

    string rspecifier = po.GetArg(1);
    string wspecifier = po.GetArg(2);

    // set up input (we'll need that to validate the selected indices)
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);

    // pre-allocated
    Matrix<BaseFloat> op_mat(1, kaldi_reader.Value().NumCols());
    Vector<BaseFloat> op_vec(kaldi_reader.Value().NumCols());
    // set up output
    BaseFloatMatrixWriter mat_writer;
    BaseFloatVectorWriter vec_writer;

    if (!output_vector) {
      if (!mat_writer.Open(wspecifier))
        KALDI_ERR << "Error in opening wspecifier: " << wspecifier;
    } else {
      if (!vec_writer.Open(wspecifier))
        KALDI_ERR << "Error in opening wspecifier: " << wspecifier;
    }
    // process all keys
    for (; !kaldi_reader.Done(); kaldi_reader.Next()) {
      Matrix<BaseFloat> &mat = kaldi_reader.Value();

      if (op == "sample") {
        // generated time index
        int32 time_index = RandInt(0, mat.NumRows() - 1);
        op_mat.Row(0).CopyFromVec(mat.Row(time_index));
        KALDI_VLOG(2) << "Random choose time index as " << time_index
                      << " for matrix " << kaldi_reader.Key();
      } else if (op == "index") {
        int32 num_rows = mat.NumRows();
        int32 index = (n >= 0 ? n : num_rows + n);
        if (index >= num_rows || index < 0)
          KALDI_ERR << "Index out of range(" << n << " vs " << num_rows << ")";
        op_mat.Row(0).CopyFromVec(mat.Row(index));
      } else {
        op_mat.Row(0).AddRowSumMat(1.0, mat, 0.0);
        if (op == "average") op_mat.Scale(1.0 / mat.NumRows());
      }
      if (output_vector) {
        op_vec.CopyFromVec(op_mat.Row(0));
        vec_writer.Write(kaldi_reader.Key(), op_vec);
      } else {
        mat_writer.Write(kaldi_reader.Key(), op_mat);
      }
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
