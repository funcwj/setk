#!/bin/bash

# For CHiME4 (log)-spectrum prepare
# wujian@2018

set -eu
seperated_data=/gruntdata/jianwu.wj/data/audio/16kHz/seperated
multichan_data=/gruntdata/jianwu.wj/data/audio/16kHz/isolated

data_dir=data/spectrum
nj=30


for x in noisy clean; do mkdir -p $data_dir/$x; done 

find $seperated_data/tr05_*_simu -name "*.CH5_clean.wav" | \
    awk -F '/' '{split($NF, a, ".");print a[1]"_"substr(a[2], 0, 3)"\t"$0}' | \
    sort -k1  > $data_dir/clean/wav.scp

find $multichan_data/tr05_*_simu -name "*.CH5.wav" | \
    awk -F '/' '{split($NF, a, ".");print a[1]"_"a[2]"\t"$0}' | \
    sort -k1 > $data_dir/noisy/wav.scp

for x in clean noisy; do
    awk '{print $1"\t"$1}' $data_dir/$x/wav.scp > $data_dir/$x/utt2spk
    cat $data_dir/$x/utt2spk | ./utils/utt2spk_to_spk2utt.pl > $data_dir/$x/spk2utt
    ./scripts/compute_stft_stats.sh --stft-config conf/stft.conf --stats-type spectrum --nj $nj \
        $data_dir/$x exp/compute_stft_stats stft/$x
    ./steps/compute_cmvn_stats.sh $data_dir/$x exp/make_cmvn stft/$x
done

echo "$0: prepare spectrum done!"
