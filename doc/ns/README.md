## Noise Suppression

MCRA based noise suppression

### Cmd options

See `./scripts/sptk/apply_ns.py -h`

### Usage

```bash
echo "egs asset/egs.wav" | ../../scripts/sptk/apply_ns.py --output wave - ns
```

The `--output` option controls the output type, audio or TF masks (also named TF gain). Note that this command is hard coded using iMCRA method.

### Reference

1. Cohen I, Berdugo B. Speech enhancement for non-stationary noise environments[J]. Signal processing, 2001, 81(11): 2403-2418.

2. Cohen I. Noise spectrum estimation in adverse environments: Improved minima controlled recursive averaging[J]. IEEE Transactions on speech and audio processing, 2003, 11(5): 466-475.