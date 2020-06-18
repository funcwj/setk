## Data Simulation

Add reverberation only, noise only or mix speakers with noises in close-talk & far-field scenarios.

### Cmd options

See `./scripts/sptk/wav_simulate.py -h`

### Usage

1. Add reverberation
    ```bash
    sox asset/4ch-rir1.wav asset/4ch-rir1-ch2.wav remix 2
    # spk1_reverb_{1,2}.wav are same
    ../../scripts/sptk/wav_simulate.py \
        --src-spk asset/spk1.wav \
        --src-rir asset/4ch-rir1-ch2.wav \
        spk1_reverb_1.wav
    ../../scripts/sptk/wav_simulate.py \
        --src-spk asset/spk1.wav \
        --src-rir asset/4ch-rir1.wav \
        --dump-channel 1 \
        spk1_reverb_2.wav 
    ```

2. Add noise
    ```bash
    # close-talk + noise
    ../../scripts/sptk/wav_simulate.py \
        --src-spk asset/spk1.wav \
        --point-noise asset/noise.wav \
        --point-noise-snr 5 \
        spk1_noisy1.wav
    # far-field + pointsource noise
    ../../scripts/sptk/wav_simulate.py \
        --src-spk asset/spk1.wav \
        --src-rir asset/4ch-rir1.wav \
        --point-noise asset/noise.wav \
        --point-noise-snr 5 \
        --point-noise-rir asset/4ch-rir3.wav \
        spk1_noisy2.wav
    # far-field + pointsource noise + isotropic noise
    ../../scripts/sptk/wav_simulate.py \
        --src-spk asset/spk1.wav \
        --src-rir asset/4ch-rir1.wav \
        --point-noise asset/noise.wav \
        --point-noise-snr 5 \
        --point-noise-rir asset/4ch-rir3.wav \
        --isotropic-noise-snr 8 \
        --isotropic-noise asset/iso.wav \
        --isotropic-noise-offset 16000 \
        spk1_noisy3.wav
    ```

3. Mix speakers
    ```bash
    # close-talk (no noise)
    ../../scripts/sptk/wav_simulate.py \
        --src-spk asset/spk1.wav,asset/spk2.wav \
        --src-begin=32000,0 \
        --src-sdr=3 \
        2spk_mix1.wav
    # close-talk (noise)
    ../../scripts/sptk/wav_simulate.py \
        --src-spk asset/spk1.wav,asset/spk2.wav \
        --src-begin=32000,0 \
        --src-sdr=3 \
        --point-noise asset/noise.wav \
        --point-noise-snr 5 \
        2spk_mix2.wav
    # far-field
    ../../scripts/sptk/wav_simulate.py \
        --src-spk asset/spk1.wav,asset/spk2.wav \
        --src-rir asset/4ch-rir1.wav,asset/4ch-rir2.wav \
        --src-begin=32000,0 \
        --src-sdr=3 \
        --point-noise asset/noise.wav \
        --point-noise-snr 5 \
        --point-noise-rir asset/4ch-rir3.wav \
        --isotropic-noise-snr 8 \
        --isotropic-noise asset/iso.wav \
        --isotropic-noise-offset 16000 \
        --dump-ref-dir ref \
        2spk_mix1.wav 
    ```

