// test_nnmf.cc

#include "include/nnmf.h"

using namespace kaldi;

void TestNnmf() {
    SparseNMFOptions opts;
    int32 f = 256, t = 100, d = 64;
    Matrix<BaseFloat> V(f, t);
    V.SetRandUniform();
    Matrix<BaseFloat> W(f, d), H(d, t);
    W.SetRandUniform(), H.SetRandUniform();
    SparseNMF snmf(opts);
    snmf.DoNMF(V, &W, &H);

}


int main() {
    TestNnmf();
    return 0;
}