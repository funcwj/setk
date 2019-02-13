## SETK: Speech Enhancement Tools integrated with Kaldi

Here are some speech enhancement tools implemented integrated with kaldi. I use them for quick validation.

### Finished based on Kaldi

* Compute kinds of masks (ibm, irm etc)
* Compute (phase angle/power&magnitude spectrogram/complex STFT results) of input wave
* Seperate target component from input wave according to input masks
* Estimate wave from enhanced spectrogram and reference wave
* Complex matrix/vector class
* MVDR/max-SNR beamformer (depend on T-F mask, may not very stable, need further debug)
* Fixed beamformer
* Compute angular spectrogram based on SRP-PHAT
* RIR generator (reference from [RIR-Generator](https://github.com/ehabets/RIR-Generator))

***Now I mainly work on [sptk](scripts) package, development based on kaldi is stoped.***

### Compile
Compile pass on Mac OSX, Ubuntu and RedHat.

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
make -j
```

