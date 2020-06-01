## Steer Vector

Computation of the steer vector matrices (used in SSL and directional features). The matrice is in shape of `num_directions x num_mics x num_bins`.

### Cmd options

See `./scripts/sptk/compute_steer_vector.py -h`.

### Usage

1. Linear array
    ```bash
    # DoA in [0, 180]
    # output shape: 181 x 6 x 257
    # 0 deg    *   *   *   *   *   * 180 deg
    ../../scripts/sptk/compute_steer_vector.py \
        --sr 16000 \
        --num-doas 181 \
        --num-bins 257 \
        --geometry linear \
        --linear-topo "0,0.01,0.02,0.03,0.04,0.05" \
        asset/1d_6mic_sv.npy
    ```

2. Circular array
    ```bash
    # DoA in [0, 360)
    # output shape: 360 x 4 x 257
    #              *
    #   180 deg *     * 0 deg
    #              *   
    ../../scripts/sptk/compute_steer_vector.py \
        --sr 16000 \
        --num-doas 360 \
        --num-bins 257 \
        --geometry circular \
        --circular-radius 0.05 \
        --circular-around 4 \
        --circular-center false \
        asset/2d_4mic_sv.npy
    ```

3. Beam pattern
    ```bash
    ../../scripts/sptk/visualize_beampattern.py --doa-range 360 \
        asset/beam_v1.npy asset/2d_4mic_sv.npy 
    ```