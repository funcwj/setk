## Spatial Features

Computation of spatial features, e.g., IPD (inter-channel difference), DF (directional/angle feature).

### Cmd options

See `./scripts/sptk/compute_ipd_and_linear_srp.py -h` for IPD and `./scripts/sptk/compute_df_on_{geometry,mask}.py` for DF.

### Usage

1. IPD
    ```bash
    # compute cosIPD between channel 0,2 0,4 and 2,4
    echo "egs asset/egs.wav" | ../../scripts/sptk/compute_ipd_and_linear_srp.py 
        --frame-len 512 
        --frame-hop 256 \
        --ipd.pair "0,2;0,4;2,4" \
        --type ipd \
        --ipd.cos true \
        - feats.ark
    # visualize and check
    ../../scripts/sptk/visualize_tf_matrix.py \
        --input ark \
        --split 3 \
        --cmap jet \
        --frame-hop 256 \
        feats.ark
    ```

2. DF

    We provide two methods to compute directional features, one is based on given steer vector and another one is based on TF-masks. The following shows the usage of `./scripts/sptk/compute_df_on_mask.py`.

    ```bash
    # estimate TF-mask of the source speaker
    echo "egs asset/egs.wav" | ../../scripts/sptk/estimate_cgmm_masks.py \
        --frame-len 512 \
        --num-iters 20 \
        - mask
    # compute DF
    echo "egs mask/egs.npy" > mask.scp
    df_pair="0,1;0,2;0,3;0,4;1,2;1,3;1,4;2,3;2,4;3,4"
    echo "egs asset/egs.wav" | ../../scripts/sptk/compute_df_on_mask.py \
        --frame-len 512 \
        --mask-format numpy \
        --df-pair $df_pair \
        - mask.scp feats.ark
    # visualize and check
    ../../scripts/sptk/visualize_tf_matrix.py \
        --input ark \
        --cmap jet \
        --frame-hop 256 \
        feats.ark 
    ```

### Reference

1. Z. Chen, X. Xiao, T. Yoshioka, et al. Multi-channel Overlapped Speech Recognition with Location Guided Speech Extraction Network[C]. 2018 IEEE Spoken Language Technology Workshop (SLT), 2018:558–565.
2. Z.-Q. Wang, D. Wang. Integrating Spectral and Spatial Features for Multi-channel Speaker Separation.[C]. Interspeech, 2018:2718–2722.
3. Z.-Q. Wang, D. Wang. All-neural Multi-channel Speech Enhancement.[C]. Interspeech, 2018:3234–3238.