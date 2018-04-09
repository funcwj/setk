#!/bin/bash

# For CHiME4 mask & spectrum prepare
# wujian@2018

set -eu
seperated_data=/gruntdata/jianwu.wj/data/audio/16kHz/seperated
multichan_data=/gruntdata/jianwu.wj/data/audio/16kHz/isolated

data_dir=data
nj=40

for x in train dev test; do mkdir -p $data_dir/$x; done 

# ch5: tr_ch5/dt_ch5
# prepare clean/noise parts of simulated data
for x in noise clean; do
    find $seperated_data/dt_ch5 -name "*_$x.wav" | \
        awk -F '/' '{split($NF, a, ".");print a[1]"_"substr(a[2], 0, 3)"\t"$0}' | \
        sort -k1 > $data_dir/dev/$x.scp
    find $seperated_data/tr05_*_simu -name "*_$x.wav" | \
        awk -F '/' '{split($NF, a, ".");print a[1]"_"substr(a[2], 0, 3)"\t"$0}' | \
        sort -k1  > $data_dir/train/$x.scp
done

# prepare origin noisy wave
find $multichan_data/tr05_*_simu -name "*.wav" | \
    awk -F '/' '{split($NF, a, ".");print a[1]"_"a[2]"\t"$0}' | \
    sort -k1 > $data_dir/train/wav.scp

# using ch5 as dev
find $multichan_data/dt05_*_simu -name "*.CH5.wav" | \
    awk -F '/' '{split($NF, a, ".");print a[1]"_"a[2]"\t"$0}' | \
    sort -k1 > $data_dir/dev/wav.scp

find $multichan_data/et05_*_real -name "*.CH5.wav" | \
    awk -F '/' '{split($NF, a, ".");print a[1]"_"a[2]"\t"$0}' | \
    sort -k1 > $data_dir/test/wav.scp

# compute stft and target masks
for x in train dev test; do
    awk '{print $1"\t"$1}' $data_dir/$x/wav.scp > $data_dir/$x/utt2spk
    cat $data_dir/$x/utt2spk | ./utils/utt2spk_to_spk2utt.pl > $data_dir/$x/spk2utt
    ./scripts/compute_stft_stats.sh --stats-type spectrum --nj $nj \
        $data_dir/$x exp/compute_stft_stats stft/$x
    ./steps/compute_cmvn_stats.sh $data_dir/$x exp/make_cmvn stft/$x
    if [ $x != "test" ]; then
        ./scripts/compute_masks.sh --mask irm --nj $nj --noise false \
            $data_dir/$x exp/compute_masks mask/$x
    fi
done

echo "$0: prepare masks & features done!"
