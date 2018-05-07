// test_snmf.cc

#include "include/snmf.h"

using namespace kaldi;

void TestNnmf() {
    SparseNMFOptions opts;
    opts.max_iter = 500;
    opts.conv_eps = 1e-3;
    opts.sparsity = 0;
    int32 f = 256, t = 100, d = 64;
    for (int32 c = 0; c < 10; c++) {
        for (int32 i = 0; i < 3; i++) {
            opts.beta = i;
            Matrix<BaseFloat> V(f, t);
            V.SetRandUniform();
            Matrix<BaseFloat> W(f, d), H(d, t);
            W.SetRandUniform(), H.SetRandUniform();
            SparseNMF snmf(opts);
            snmf.DoSparseNMF(V, &W, &H);
        }
    }
}


int main() {
    TestNnmf();
    return 0;
}
