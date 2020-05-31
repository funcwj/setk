## SETK: Speech Enhancement Tools integrated with Kaldi

Here are some speech enhancement/separation tools integrated with [Kaldi](https://github.com/kaldi-asr/kaldi). I use them for front-end's data processing.

### Finished based on Kaldi

* Compute kinds of masks (ibm, irm etc)
* Compute (phase angle/power&magnitude spectrogram/complex STFT results) of input wave
* Seperate target component using input masks
* Wave reconstruction from enhanced spectral features and reference phase
* Complex matrix/vector class
* MVDR/GEVD beamformer (depend on T-F mask, may not very stable, need further debugging)
* Fixed beamformer
* Compute angular spectrogram based on SRP-PHAT
* RIR generator (reference from [RIR-Generator](https://github.com/ehabets/RIR-Generator))

***Now I mainly work on [sptk](scripts) package, development based on kaldi is stopped.***

### Python ([scripts/sptk](scripts)) Extention

* Supervised (mask-based) adaptive beamformer (GEVD/MVDR/PWWF)
* Data convertion among MATLAB, Numpy and Kaldi
* Data visualization (TF-mask, spatial/spectral features, beam pattern...)
* Unified data and IO handlers for Kaldi's scripts, archives, wave, spectrogram, numpy's ndarray...
* Unsupervised mask estimation (CGMM/CACGMM)
* Spatial/Spectral feature computation
* DS (delay and sum) beamformer, SD (supper-directive) beamformer
* AuxIVA, GWPE, FB (Fixed Beamformer)
* Mask computation (iam, irm, ibm, psm, crm)
* RIR simulation (1D/2D arrays)
* Single channel speech separation (TF spectral masking)
* Si-SDR/SDR/WER evaluation
* Pywebrtc vad wrapper
* Mask-based source localization
* ...

### Compile

Compile [Kaldi](https://github.com/kaldi-asr/kaldi) with `--shared` flags and patch `matrix/matrix-common.h`
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
export KALDI_ROOT=/path/to/kaldi/root
export OPENFST_ROOT=/path/to/openfst/root
# if on UNIX, need compile kaldi with openblas
export OPENBLAS_ROOT=/path/to/openblas/root
cmake ..
make -j
```

### Document

* [Adaptive Beamformer](doc/adaptive_beamformer)
* [Fixed Beamformer](doc/fixed_beamformer)
* [Sound Source Localization](doc/ssl)
* [Spectral Feature](doc/spectral_feature)
* [Spatial Feature](doc/spatial_feature)
* [VAD](doc/vad)
* [Noise Suppression](doc/ns)
* [Steer Vector](doc/steer_vector)
* [Room Impluse Response](doc/rir)
* [Spatial Clustering](doc/spatial_clustering)
* [WPE](doc/wpe)
* [Time-frequency Mask](doc/tf_mask)

