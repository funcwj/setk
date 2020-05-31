## WPE

Weighted Prediction Error for speech dereverberation.

### Cmd options

See `./scripts/sptk/apply_gwpe.py -h`

### Usage

```bash
echo "egs asset/egs.wav" | ../../scripts/sptk/apply_gwpe.py --frame-hop 128  - wpe
```

To use nara-wpe:
```bash
echo "egs asset/egs.wav" | ../../scripts/sptk/apply_gwpe.py --frame-hop 128  --nara-wpe true - wpe
```

The dereverbrated output of the `asset/egs.wav` is in `asset/dereverb_egs.wav`.

### Reference

1. [nara_wpe](https://github.com/fgnt/nara_wpe)

2. Yoshioka, Takuya, and Tomohiro Nakatani. "Generalization of multi-channel linear prediction methods for blind MIMO impulse response shortening." IEEE Transactions on Audio, Speech, and Language Processing 20.10 (2012): 2707-2720.