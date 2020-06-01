## Spectral Feature

Computation of the spectrogam & mel-filter bank features.

### Cmd options

See `./scripts/sptk/compute_{spectrogram,fbank}.py -h`

### Usage

1. Spectrogram
    ```bash
    # 257-dim log spectrogram
    echo "egs asset/egs.wav" | ../../scripts/sptk/compute_spectrogram.py \
        --frame-len 400 \
        --frame-hop 256 \
        --round-power-of-two true \
        --center true \
        --apply-log true \
        - feats.ark
    # visualize and check
    ../../scripts/sptk/visualize_tf_matrix.py \
        --input ark --cmap jet --frame-hop 256 \
        feats.ark
    ```

2. Fbank
    ```bash
    # 80-dim log fbank
    echo "egs asset/egs.wav" | ../../scripts/sptk/compute_fbank.py \
        --frame-len 400 \
        --frame-hop 256 \
        --round-power-of-two true \
        --center true \
        --apply-log true \
        --num-bins 80 \
        - feats.ark
    # visualize and check
    ../../scripts/sptk/visualize_tf_matrix.py \
        --input ark --cmap jet --frame-hop 256 \
        feats.ark
    ```