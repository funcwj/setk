## RIR simulation

Generation of the room impluse response (RIR) using image method. Now three optional backends are available:

1. rir-simulate (see setk/src/rir-simulate.cc)
2. [pyrirgen](https://github.com/Marvin182/rir-generator)
3. [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR)

### Cmd options

See `./scripts/sptk/rir_generate_1d.py -h` or `./scripts/sptk/rir_generate_2d.py -h`. Using `--gpu true` to set [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) as backend.

### Usage

The following commands will generate `Room{1,2}-{1..25}.wav, rir.json, Room{1,2}.jpg` under directory `rir_egs`. See examples at [asset](asset).

1. 1D (linear) arrays
    ```bash
    dump_dir=rir_egs
    num_room=2
    num_rirs=25
    # CPU version is slow, use --gpu true or run.pl to make parallelization
    ../../scripts/sptk/rir_generate_1d.py \
        --num-rirs $num_rirs \
        --dump-dir $dump_dir \
        --array-height "1.2,1.8" \
        --array-topo "0,0.05,0.1,0.15" \
        --room-dim "4,7;4,7;2,3" \
        --rt60 "0.2,0.5" \
        --array-relx "0.4,0.6" \
        --array-rely "0.1,0.2" \
        --speaker-height "1,2" \
        --source-distance "1.5,3" \
        --rir-dur 0.5 \
        --vertical-oriented false \
        --dump-cfg true \
        --gpu false \
        $num_room
    ```

2. 2D (circular) arrays
    ```bash
    dump_dir=rir_egs
    num_room=2
    num_rirs=25
    
    ../../scripts/sptk/rir_generate_2d.py \
        --num-rirs $num_rirs \
        --dump-dir $dump_dir \
        --array-height "1.2,1.8" \
        --array-topo "0,0.05;0.05,0;0,-0.05;-0.05,0" \
        --room-dim "4,7;4,7;2,3" \
        --rt60 "0.2,0.5" \
        --array-relx "0.4,0.6" \
        --array-rely "0.1,0.2" \
        --speaker-height "1,2" \
        --source-distance "1.5,3" \
        --rir-dur 0.5 \
        --dump-cfg true \
        --gpu false \
        $num_room
    ```