## Time-frequency Mask

Computation of the time-frequency mask (PSM, IRM, IBM, IAM, ...) as the neural network training labels.

### Cmd options

See `./scripts/sptk/compute_mask.py -h`

### Usage

1. IBM & IRM computation
    ```bash
    # prepare scp
    echo "egs asset/clean.wav" > clean.scp
    echo "egs asset/noisy.wav" > noisy.scp
    # computation
    ../../scripts/sptk/compute_mask.py --mask irm clean.scp noisy.scp irm.ark
    # visualize and check
    ../../scripts/sptk/visualize_tf_matrix.py --input ark irm.ark --cmap jet --cache-dir irm
    ```

2. PSM & IAM (FFT-mask or SMM) computation
    ```bash
    # add cutoff as they are unbounded
    ../../scripts/sptk/compute_mask.py --mask psm --cutoff 2 clean.scp noisy.scp psm.ark
    # visualize and check
    ../../scripts/sptk/visualize_tf_matrix.py --input ark psm.ark --cmap jet --cache-dir psm
    ```

3. Restore audio using TF-masks
    ```bash
    # psm as example
    ../../scripts/sptk/compute_mask.py --mask psm --cutoff 2 --scp mask.scp clean.scp noisy.scp mask.ark
    # do TF masking (using noisy phase)
    ../../scripts/sptk/wav_separate.py --mask-format kaldi noisy.scp mask.scp enh
    ```
    The enhancement output is under directory `enh`. See `../../scripts/sptk/wav_separate.py -h` for more command options.