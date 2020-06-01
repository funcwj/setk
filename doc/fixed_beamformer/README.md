## Fixed Beamformer

DS (delay and sum) beamformer, SD (supper-directive) beamformer and other fixed beamformers.

### Cmd options

See `./scripts/sptk/apply_{ds,sd,fixed}_beamformer.py -h`

### Usage

```bash
# Get steer vector
../../scripts/sptk/compute_steer_vector.py \
    --num-doas 360 \
    --num-bins 257 \
    --geometry circular \
    --circular-radius 0.05 \
    --circular-around 4 \
    4mic_sv.npy
# SSL (got 100 degree)
echo "egs asset/egs.wav" | ../../scripts/sptk/do_ssl.py \
    --frame-len 512 \
    --frame-hop 256 \
    --backend srp \
    --srp-pair "0,2;1,3" \
    --doa-range 0,360 \
    --output degree \
    - 4mic_sv.npy doa.scp
# DS beamformer
echo "egs asset/egs.wav" | ../../scripts/sptk/apply_ds_beamformer.py \
    --frame-len 512 \
    --frame-hop 256 \
    --geometry circular \
    --circular-around 4 \
    --circular-radius 0.05 \
    --utt2doa doa.scp \
    --sr 16000 - ds
# SD beamformer
echo "egs asset/egs.wav" | ../../scripts/sptk/apply_sd_beamformer.py \
    --frame-len 512 \
    --frame-hop 256 \
    --geometry circular \
    --circular-around 4 \
    --circular-radius 0.05 \
    --utt2doa doa.scp \
    --sr 16000 - sd
```

To use other fixed beamformers, pre-design the filter coefficients on each direction (in shape of `num_directions x num_bins x num_mics`) and run `./scripts/sptk/apply_fixed_beamformer.py`.