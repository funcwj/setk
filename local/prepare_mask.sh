#!/bin/bash
# wujian@2018

set -ue

nj=40

. ./path.sh
. ./utils/parse_options.sh || exit 1


[ $# -ne 2 ] && echo "format error: $0 <wav-dir> <data-dir>" && exit 1


wav_dir=$(cd $1; pwd)

clean_dir=$wav_dir/target 
noisy_dir=$wav_dir/noisy
noise_dir=$wav_dir/noise

for f in $clean_dir $noisy_dir $noise_dir; do [ ! -d $f ] && echo "**error**: $f not exists!" && exit 1; done

mask_dir=$2 && mkdir -p $mask_dir && mask_dir=$(cd $mask_dir; pwd)

find `cd $noisy_dir; pwd` -name "*.wav" | awk -F '/' '{print $NF"\t"$0}' | \
    sed 's:\.wav::' | sort -k1 > $mask_dir/wav.scp
find `cd $clean_dir; pwd` -name "*.wav" | awk -F '/' '{print $NF"\t"$0}' | \
    sed 's:\.wav::' | sort -k1 > $mask_dir/clean.scp
find `cd $noise_dir; pwd` -name "*.wav" | awk -F '/' '{print $NF"\t"$0}' | \
    sed 's:\.wav::' | sort -k1 > $mask_dir/noise.scp

# utt-level cmvn
awk '{print $1"\t"$1}' $mask_dir/wav.scp > $mask_dir/utt2spk
awk '{print $1"\t"$1}' $mask_dir/wav.scp > $mask_dir/spk2utt

feats_dir=`basename $mask_dir`
./steps/compute_masks.sh --nj $nj $mask_dir exp/compute_mask mask/$feats_dir
# or use
# ./scripts/compute_oracle_mask.sh --nj $nj $mask_dir exp/compute_mask mask/$feats_dir

echo "$0: done."
