// srp-phat.cc
// wujian@18.5.29
//

#include "include/srp-phat.h"
#include "include/stft.h"

using namespace kaldi;

void TestSrpPhat() {
  SrpPhatOptions srp_opts;
  srp_opts.topo_descriptor = "0,0.037,0.113,0.226";
  srp_opts.smooth_context = 0;

  ShortTimeFTOptions stft_opts;
  stft_opts.window = "hanning";

  SrpPhatComputor srp_computor(srp_opts, 16000, 513);

  bool binary;
  Input wave_in("srp_test.wav", &binary);
  WaveData wave;
  wave.Read(wave_in.Stream());

  Matrix<BaseFloat> stft, srp_phat;
  ShortTimeFTComputer stft_computer(stft_opts);
  stft_computer.Compute(wave.Data(), &stft, NULL, NULL);

  CMatrix<BaseFloat> cstft(stft.NumRows(), stft.NumCols() / 2 + 1);
  cstft.CopyFromRealfft(stft);
  srp_computor.Compute(cstft, &srp_phat);

  Output ko("srp_test.mat", true);
  srp_phat.Write(ko.Stream(), true);
  // KALDI_LOG << srp_phat;
}

void TestMathOpts() {
  CMatrix<BaseFloat> A(4, 6), B(4, 6);
  A.SetRandn();
  B.SetRandn();
  CMatrix<BaseFloat> R(A);
  KALDI_LOG << "A = " << A;
  KALDI_LOG << "B = " << B;
  R.MulElements(B, kConj);
  KALDI_LOG << "A .* B = " << R;
  R.DivElements(A, kNoConj, true);
  KALDI_LOG << "A .* B / |A| = " << R;
  R.DivElements(B, kNoConj, true);
  KALDI_LOG << "A .* B / (|A| .* |B|) = " << R;
}
int main() {
  TestMathOpts();
  TestSrpPhat();
  return 0;
}
