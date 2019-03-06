#!/usr/bin/env bash

set -eu

function prepare_wav {
    find $1 -name "*.wav" | awk -F '/' '{print $NF"\t"$0}' \
      | sed "s:\.wav::" | sort -k1
}

function prepare_perutt {
  [ ! -f $1/wav.scp ] && echo "missing $1/wav.scp" && exit 1
  awk '{print $1" "$1}' $1/wav.scp > $1/utt2spk
  awk '{print $1" "$1}' $1/wav.scp > $1/spk2utt
}

nj=40
index="0,3\;1,4\;2,5\;0,1\;2,3\;4,5"

. ./utils/parse_options.sh

[ $# -ne 2 ] && echo "Script format error: $0 <simu-dir> <data-dir>" && exit 1

simu_dir=$1
data_dir=$2

for dir in train dev test; do
  [ -d $data_dir/$dir ] && echo "clean $dir..." && rm -rf $data_dir/$dir
  [ ! -d $simu_dir/$dir/mix ] && echo "$0: missing dir $simu_dir/$dir/mix" && exit 1
  # init data dir
  mkdir -p $data_dir/$dir
  prepare_wav $simu_dir/$dir/mix > $data_dir/$dir/wav.scp
  prepare_perutt $data_dir/$dir
  
  ./scripts/compute_spatial_feats.sh \
    --nj $nj \
    --feats ipd \
    --stft-conf ./conf/16k.stft.conf \
    --ipd-index $index \
    --ipd-cos true \
    --compress false \
    $data_dir/$dir \
    ./exp/make_ipd/$dir \
    ./ipd/$dir
done