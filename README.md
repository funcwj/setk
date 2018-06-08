## SETK: Speech Enhancement Tools based on Kaldi

This is some speech enhancement tools implemented based on kaldi. I use them for quick experiment.

Updating...

### Finished

* Compute kinds of masks(ibm, irm etc)
* Compute (phase angle/power&magnitude spectrum/complex STFT results) of input wave
* Seperate target component from input wave according to input masks
* Estimate wave from enhanced spectrum and reference wave
* Complex matrix/vector class
* MVDR/max-SNR beamformer(depend on T-F mask)
* Fixed beamformer
* Compute angular spectrogram based on SRP-PHAT 

### Install
Compile pass on Mac OSX, Ubuntu and RedHat. I haven't try them on Ubuntu yet.

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

