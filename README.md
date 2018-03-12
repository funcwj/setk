## Notes

This is some speech enhancement tools implemented based on kaldi.

Updating...

### Finished

* Compute kinds of masks(ibm, irm etc)
* Compute (phase angle/power&magnitude spectrum/complex STFT results) of input wave
* Seperate target component from input wave according to input masks
* Estimate wave from enhanced spectrum and reference wave

### Install
```shell
mkdir build
export KALDI_ROOT=/kaldi/root/dir
cmake ..
make
```

### Unit Test

* experiment on CHiME4 dt05_simu(CH5)

| Training Data |    Model    | PESQ(noisy/enhan) |
| :-----------: | :---------: | :---------------: |
|      CH5      |     IRM     |     2.18/2.65     |
|    CH[1-6]    |     IRM     |     2.18/2.70     |
|      CH5      |   LSP-MAP   |     2.18/2.49     |
|      CH5      |     CM      |     2.18/2.58     |
|      CH5      |   MVN-CM    |     2.18/2.54     |
|      CH5      | MVN-LSP-MAP |     2.18/2.57     |