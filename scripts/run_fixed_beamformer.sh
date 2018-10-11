#!/usr/bin/env bash

# wujian@2018

set -eu

nj=40
cmd="run.pl"
stft_conf=conf/stft.conf
weight_key="weights"

echo "$0 $@"

. ./path.sh
. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "Script format error: $0 <wav-scp> <weight-mat> <enhan-dir>" && exit 1

wav_scp=$1
weight=$2
enhan_dir=$3

exp_dir=./exp/fixed_beamformer && mkdir -p $exp_dir

wav_split_scp="" && for n in $(seq $nj); do wav_split_scp="$wav_split_scp $exp_dir/wav.$n.scp"; done
./utils/split_scp.pl $wav_scp $wav_split_scp

stft_opts=$(cat $stft_conf | xargs)
beamformer_opts="$stft_opts --weight-key $weight_key"

mkdir -p $enhan_dir
$cmd JOB=1:$nj $exp_dir/log/run_beamformer.JOB.log \
  ./scripts/sptk/apply_fix_beamformer.py \
  $beamformer_opts \
  $exp_dir/wav.JOB.scp \
  $weight $enhan_dir

echo "$0: Run fixed beamformer done!"

