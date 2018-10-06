#!/usr/bin/env bash

# wujian@2018


set -eu

cmd="run.pl"
nj=40
epochs=20
window="hann"
frame_length=1024
frame_shift=256

. ./path.sh || exit 1
. ./utils/parse_options.sh || exit 1

[ $# -ne 2 ] && echo "Script format error: $0 <wav-scp> <dst-dir>" && exit 1

wav_scp=$1
dst_dir=$2

[ ! -d $dst_dir ] && mkdir -p $dst_dir

exp_dir=exp/auxiva && mkdir -p $exp_dir

split_wav_scp=""
for n in $(seq $nj); do split_wav_scp="$split_wav_scp $exp_dir/wav.$n.scp"; done

./utils/split_scp.pl $wav_scp $split_wav_scp || exit 1

$cmd JOB=1:$nj $exp_dir/log/run_auxiva.JOB.log \
  ./scripts/sptk/apply_auxiva.py \
  --num-epochs $epochs \
  --window $window \
  --frame-length $frame_length \
  --frame-shift $frame_shift \
  $exp_dir/wav.JOB.scp \
  $dst_dir


echo "$0: do auxiva for $wav_scp done"
