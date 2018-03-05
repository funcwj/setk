### Notes

This is some speech enhancement tools implemented based on kaldi.

Update...

#### Finished

* Compute kinds of masks(ibm, irm etc)
* Compute (phase angle/power&magnitude spectrum/complex STFT results) of input wave
* Seperate target component from input wave according to input masks
* Estimate wave from enhanced spectrum and reference wave

#### Unit Test

|    Data    | Mask | PESQ(noisy/enhan) |
| :--------: | :--: | :---------------: |
| CHiME4 dev | IRM  |  2.16744/2.64598  |