#!/usr/bin/env bash

# wujian@2018

set -eu

nj=40
cmd="run.pl"
stft_conf=conf/stft.conf

echo "$0 $@"

function usage {
  echo "Options:"
  echo "  --nj          <nj>                  # number of jobs to run parallel, (default=40)"
  echo "  --cmd         <run.pl|queue.pl>     # how to run jobs, (default=run.pl)"
  echo "  --stft-conf   <stft-conf>           # stft configurations files, (default=conf/stft.conf)"
}

. ./path.sh
. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "Script format error: $0 <wav-scp> <weight-mat> <enhan-dir>" && usage && exit 1

wav_scp=$1
weight=$2
enhan_dir=$3

for x in $wav_scp $weight $stft_conf; do [ ! -f $x ] && echo "$0: missing file: $x" && exit 1; done

exp_dir=./exp/fixed_beamformer && mkdir -p $exp_dir

wav_split_scp="" && for n in $(seq $nj); do wav_split_scp="$wav_split_scp $exp_dir/wav.$n.scp"; done
./utils/split_scp.pl $wav_scp $wav_split_scp

stft_opts=$(cat $stft_conf | xargs)

mkdir -p $enhan_dir
$cmd JOB=1:$nj $exp_dir/log/run_beamformer.JOB.log \
  ./scripts/sptk/apply_fix_beamformer.py \
  $stft_opts \
  $exp_dir/wav.JOB.scp \
  $weight $enhan_dir

echo "$0: Run fixed beamformer done!"

