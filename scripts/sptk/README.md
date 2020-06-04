Python scripts (work with python 3.6+) for speech enhancement/separation integrated with kaldi, which could be used independently.

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
* Noise suppression
* Data simulation
* ...