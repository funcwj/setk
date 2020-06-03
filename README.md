## SETK: Speech Enhancement Tools integrated with Kaldi

Here are some speech enhancement/separation tools integrated with [Kaldi](https://github.com/kaldi-asr/kaldi). I use them for front-end's data processing.

### Python Scripts

* Supervised (mask-based) adaptive beamformer (GEVD/MVDR/MCWF...)
* Data convertion among MATLAB, Numpy and Kaldi
* Data visualization (TF-mask, spatial/spectral features, beam pattern...)
* Unified data and IO handlers for Kaldi's scripts, archives, wave and numpy's ndarray...
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
* Noise suppression
* ...

Please check out the following instruction for usage of the scripts.

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
* [Format Transform](doc/format_transform)
* [Data Simulation](doc/data_simu)

### Kaldi Commands

* Compute time-frequency masks (ibm, irm etc)
* Compute phase & magnitude spectrogram & complex STFT
* Seperate target component using input masks
* Wave reconstruction from enhanced spectral features
* Complex matrix/vector class
* MVDR/GEVD beamformer (depend on T-F mask, not very stable)
* Fixed beamformer
* Compute angular spectrogram based on SRP-PHAT
* RIR generator (reference from [RIR-Generator](https://github.com/ehabets/RIR-Generator))

To build the sources, you need to compile [Kaldi](https://github.com/kaldi-asr/kaldi) with `--shared` flags and patch `matrix/matrix-common.h` first
```c++
typedef enum {
    kTrans          = 112,  // CblasTrans
    kNoTrans        = 111,  // CblasNoTrans
    kConjTrans      = 113,  // CblasConjTrans
    kConjNoTrans    = 114   // CblasConjNoTrans
} MatrixTransposeType;
```

Then run
```bash
mkdir build
cd build
export KALDI_ROOT=/path/to/kaldi/root
export OPENFST_ROOT=/path/to/openfst/root
# if on UNIX, need compile kaldi with openblas
export OPENBLAS_ROOT=/path/to/openblas/root
cmake ..
make -j
```

***Now I mainly work on [sptk](scripts) package, development based on kaldi is stopped.***

