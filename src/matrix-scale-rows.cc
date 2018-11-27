// src/matrix-scale-rows.cc

// Copyright 2018  (author: Jian Wu)

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
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Scale the rows of an input table of matrices and output the "
        "corresponding table of matrices\n"
        "\n"
        "Usage: matrix-scale-rows [options] <vector-rspecifier> "
        "<matrix-rspecifier> <matrix-wspecifier>\n"
        "e.g.: matrix-scale-rows ark:- scp:post.scp ark:weight_post.ark\n"
        "See also: matrix-sum, vector-sum\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string vector_rspecifier = po.GetArg(1);
    std::string matrix_rspecifier = po.GetArg(2);
    std::string matrix_wspecifier = po.GetArg(3);

    SequentialBaseFloatVectorReader vec_reader(vector_rspecifier);
    RandomAccessBaseFloatMatrixReader mat_reader(matrix_rspecifier);
    BaseFloatMatrixWriter mat_writer(matrix_wspecifier);

    int32 num_done = 0, num_matrix = 0;

    for (; !vec_reader.Done(); vec_reader.Next()) {
      std::string key = vec_reader.Key();
      num_matrix++;

      if (!mat_reader.HasKey(key)) continue;

      Matrix<BaseFloat> mat(mat_reader.Value(key));
      const Vector<BaseFloat> &scale = vec_reader.Value();

      int32 vector_dim = scale.Dim(), num_rows = mat.NumRows();
      int32 num_scale_rows = vector_dim;
      if (vector_dim != num_rows) {
        num_scale_rows = std::min(vector_dim, num_rows);
        KALDI_VLOG(1) << "vector dim = " << vector_dim
                      << ", matrix num_rows = " << num_rows << ", scale first "
                      << num_scale_rows << " rows";
      }

      mat.RowRange(0, num_scale_rows)
          .MulRowsVec(scale.Range(0, num_scale_rows));
      mat_writer.Write(key, mat);
      num_done++;
    }

    KALDI_LOG << "Scaled " << num_done << " matrices, " << num_matrix
              << " matrix in total.";

    return (num_done != 0 ? 0 : 1);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
