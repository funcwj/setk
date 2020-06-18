## Adaptive Beamformer

Implementation of the mask-based adaptive beamformer (MVDR, GEVD, MCWF).

### Cmd options

See `./scripts/sptk/apply_adaptive_beamformer.py -h`.

### Usage

```bash
echo "egs asset/egs.wav" > wav.scp
# estimate TF-masks
../../scripts/sptk/estimate_cgmm_masks.py \
    --frame-len 512 \
    --frame-hop 256 \
    --num-iters 20 \
    wav.scp mask
# visualize and check
../../scripts/sptk/visualize_tf_matrix.py \
    --input dir \
    --cmap binary \
    --frame-hop 256 \
    mask
# mvdr
../../scripts/sptk/apply_adaptive_beamformer.py \
    --frame-len 512 \
    --frame-hop 256 \
    --mask-format numpy \
    --beamformer mvdr \
    wav.scp mask.scp mvdr
# gevd
../../scripts/sptk/apply_adaptive_beamformer.py \
    --frame-len 512 \
    --frame-hop 256 \
    --mask-format numpy \
    --beamformer gevd \
    wav.scp mask.scp gevd
# gevd-ban
../../scripts/sptk/apply_adaptive_beamformer.py \
    --frame-len 512 \
    --frame-hop 256 \
    --mask-format numpy \
    --beamformer gevd \
    --ban true
    wav.scp mask.scp gevd-ban
# pmwf-0
../../scripts/sptk/apply_adaptive_beamformer.py \
    --frame-len 512 \
    --frame-hop 256 \
    --mask-format numpy \
    --beamformer pmwf-0 \
    wav.scp mask.scp pmwf-0
# pmwf-0-eig
../../scripts/sptk/apply_adaptive_beamformer.py \
    --frame-len 512 \
    --frame-hop 256 \
    --mask-format numpy \
    --beamformer pmwf-0 \
    --rank1-appro eig \
    wav.scp mask.scp pmwf-0
# pmwf-0-gev
../../scripts/sptk/apply_adaptive_beamformer.py \
    --frame-len 512 \
    --frame-hop 256 \
    --mask-format numpy \
    --beamformer pmwf-0 \
    --rank1-appro gev \
    wav.scp mask.scp pmwf-0-gev
```

### Reference

1. J. Heymann, L. Drude, R. Haeb-Umbach. Neural Network Based Spectral Mask Estimation for Acoustic Beamforming[C]. 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2016:196–200.
2. Erdogan H, Hershey J R, Watanabe S, et al. Improved MVDR Beamforming Using Single-Channel Mask Prediction Networks[C]//Interspeech. 2016: 1981-1985.
3. Souden M, Benesty J, Affes S. On optimal frequency-domain multichannel linear filtering for noise reduction[J]. IEEE Transactions on audio, speech, and language processing, 2010, 18(2): 260-276.
4. E. Warsitz, R. Haeb-Umbach. Blind Acoustic Beamforming Based on Generalized Eigenvalue Decomposition[J]. IEEE Transactions on audio, speech, and language processing, 2007, 15(5):1529–1539.
5. Ziteng Wang, Emmanuel Vincent, Romain Serizel, and Yonghong Yan, “Rank-1 Constrained Multichannel Wiener Filter for Speech Recognition in Noisy Environments,” Jul 2017.