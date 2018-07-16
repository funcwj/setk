// src/matrix-scale-elements.cc
// wujian@2018

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
// See the Apache 2 License for the spec

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;

        const char *usage =
            "Compute hadamard product of matrix\n"
            "\n"
            "Usage: matrix-scale-elements [options] <matrix-rspecifier> <matrix-rspecifier> <matrix-wspecifier>\n"
            "e.g.: matrix-scale-elements scp:masks.scp scp:weights.scp ark,scp:fixed_masks.ark,fixed_masks.scp\n";
        
        ParseOptions po(usage);

        BaseFloat power = 1;
        po.Register("apply-pow", &power, "Apply power after hadamard product");

        po.Read(argc, argv);

        if (po.NumArgs() != 3) {
            po.PrintUsage();
            exit(1);
        }
        std::string input_rspecifier = po.GetArg(1);
        std::string scale_rspecifier = po.GetArg(2);
        std::string matrix_wspecifier = po.GetArg(3);

        SequentialBaseFloatMatrixReader input_reader(input_rspecifier);
        RandomAccessBaseFloatMatrixReader scale_reader(scale_rspecifier);
        BaseFloatMatrixWriter mat_writer(matrix_wspecifier);

        int32 num_done = 0, num_matrix = 0;

        for (; !input_reader.Done(); input_reader.Next()) {
            std::string key = input_reader.Key();
            num_matrix++;

            if (!scale_reader.HasKey(key))
                continue;

            Matrix<BaseFloat> scale(scale_reader.Value(key));
            const Matrix<BaseFloat> &input = input_reader.Value();
            scale.MulElements(input);
            scale.ApplyPow(power);

            mat_writer.Write(key, scale);
            num_done++;
        }

        KALDI_LOG << "Scaled " << num_done << " matrices, "
                  << num_matrix << " matrix in total.";

        return (num_done != 0 ? 0 : 1);
    }
    catch (const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
