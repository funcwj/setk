## WPE

Weighted Prediction Error for speech dereverberation.

### Cmd options

See `./scripts/sptk/apply_wpe.py -h`

### Usage

```bash
echo "egs asset/egs.wav" | ../../scripts/sptk/apply_wpe.py \
    --frame-len 512 \
    --frame-hop 128 \
    - wpe
```

To use nara-wpe:
```bash
echo "egs asset/egs.wav" | ../../scripts/sptk/apply_wpe.py \
    --frame-len 512 \
    --frame-hop 128 \
    --nara-wpe true \
    - wpe
```

To run WPD, using command
```bash
echo "egs asset/egs.wav" |  ../../scripts/sptk/apply_wpd.py \
    --frame-len 512 \
    --taps 10 --delay 3 --context 1 \
    --wpd-iters 2 --cgmm-iters 10 \
    --update-alpha false \
    --dump-mask true - wpd
```
which will generate TF-masks and dereverbrated & enhanced audio simultaneously.

### Reference

1. [nara_wpe](https://github.com/fgnt/nara_wpe)
2. Yoshioka, Takuya, and Tomohiro Nakatani. "Generalization of multi-channel linear prediction methods for blind MIMO impulse response shortening." IEEE Transactions on Audio, Speech, and Language Processing 20.10 (2012): 2707-2720.
3.  Nakatani, Tomohiro, and Keisuke Kinoshita. "A unified convolutional beamformer for simultaneous denoising and dereverberation." IEEE Signal Processing Letters 26.6 (2019): 903-907.
4.  Boeddeker, Christoph, et al. "Jointly optimal dereverberation and beamforming." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.
5.  Nakatani, Tomohiro, and Keisuke Kinoshita. "Maximum likelihood convolutional beamformer for simultaneous denoising and dereverberation." 2019 27th European Signal Processing Conference (EUSIPCO). IEEE, 2019.