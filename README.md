## SETK: Speech Enhancement Tools based on Kaldi

This is some speech enhancement tools implemented based on kaldi. I use them for quick experiment.

Updating...

### Finished

* Compute kinds of masks(ibm, irm etc)
* Compute (phase angle/power&magnitude spectrum/complex STFT results) of input wave
* Seperate target component from input wave according to input masks
* Estimate wave from enhanced spectrum and reference wave
* Complex matrix/vector class
* MVDR(minimum variance distortionless response) beamformer(depend on estimated target mask)
* GEV(generalized eigenvector decomposition)/max-SNR beamformer(depend on estimated target mask)

### Install
Compile pass on Mac OSX and RedHat. I haven't try them on Ubuntu yet.

Patch `matrix/matrix-common.h` in Kaldi
```c++
typedef enum {
    kTrans          = 112,  // CblasTrans
    kNoTrans        = 111,  // CblasNoTrans
    kConjTrans      = 113,  // CblasConjTrans
    kConjNoTrans    = 114   // CblasConjNoTrans
} MatrixTransposeType;
```

Then run
```shell
mkdir build
cd build
export KALDI_ROOT=/kaldi/root/dir
# if on UNIX, need compile kaldi with openblas
export OPENBLAS_ROOT=/openblas/root/dir
cmake ..
make
```

### Unit Test

* Single channel speech enhancement experiment on CHiME4 dt05_simu(CH5)

| Training Data |    Model    | PESQ(noisy/enhan) |
| :-----------: | :---------: | :---------------: |
|      CH5      |     IRM     |     2.18/2.65     |
|    CH[1-6]    |     IRM     |     2.18/2.70     |
|      CH5      |   LSP-MAP   |     2.18/2.49     |
|      CH5      |     CM      |     2.18/2.58     |
|      CH5      |   MVN-CM    |     2.18/2.54     |
|      CH5      | MVN-LSP-MAP |     2.18/2.57     |
