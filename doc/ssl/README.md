## SSL (Sound Source Localization)

SSL implementation.

### Cmd options

See `./scripts/sptk/do_ssl.py -h` for SSL and `./scripts/sptk/compute_ipd_and_linear_srp.py`, `./scripts/sptk/compute_circular_srp.py` for SRP angular spectrum computation.

### Usage

```bash
echo "egs asset/egs.wav" > wav.scp
# srp matrices
../../scripts/sptk/compute_circular_srp.py \
    --frame-len 512 \
    --frame-hop 256 \
    --n 16 \
    --d 0.1 \
    --diag-pair "0,8;1,9;2,10;3,11;4,12;5,13;6,14;7,15" \
    --num-doa 361 \
    wav.scp srp.ark
# visualize and check (found peak around 60 degree)
../../scripts/sptk/visualize_angular_spectrum.py --frame-hop 16 srp.ark
# compute steer vector
../../scripts/sptk/compute_steer_vector.py \
    --num-doas 360 \
    --num-bins 267 \
    --sr 16000 \
    --geometry circular \
    --circular-radius 0.05 \
    --circular-around 16 \
    16mic_sv.npy
# run srp-based SSL (got 59 degree)
../../scripts/sptk/do_ssl.py \
    --frame-len 512 \
    --frame-hop 256 \
    --backend srp \
    --doa-range 0,360 \
    --output degree \
    --srp-pair "0,8;1,9;2,10;3,11;4,12;5,13;6,14;7,15" 
    wav.scp 16mic_sv.npy doa.scp
# run SSL using ml backend (also got 59 degree)
../../scripts/sptk/do_ssl.py \
    --frame-len 512 \
    --frame-hop 256 \
    --backend ml \
    --doa-range 0,360 \
    --output degree \
    wav.scp 16mic_sv.npy doa.scp
```