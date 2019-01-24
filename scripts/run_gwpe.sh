#!/usr/bin/env bash

# wujian@2019

set -eu

nj=20
cmd="run.pl"
iters=3
stft_conf=conf/gwpe.conf
delay=3
taps=10
context=0
fs=16000

echo "$0 $@"

function usage {
  echo "Options:"
  echo "  --nj        <nj>                  # number of jobs to run parallel, (default=$nj)"
  echo "  --cmd       <run.pl|queue.pl>     # how to run jobs, (default=$cmd)"
  echo "  --stft-conf <stft-conf>           # stft configuration files, (default=$stft_conf)"
  echo "  --iters     <iters>               # number of iters to run GWPE, (default=$iters)"
  echo "  --delay     <delay>               # time delay in GWPE, (default=$delay)"
  echo "  --taps      <taps>                # number of taps in GWPE, (default=$taps)"
  echo "  --context   <context>             # left/right context used in PSD matrix estimation, (default=$context)"
  echo "  --fs        <fs>                  # sample rate for source wave, (default=$fs)"
}

. ./path.sh
. ./utils/parse_options.sh || exit 1 

[ $# -ne 2 ] && echo "Script format error: $0 <wav-scp> <dst-dir>" && usage && exit 1

wav_scp=$1
dst_dir=$2

for x in $wav_scp $stft_conf; do [ ! -f $x ] && echo "$0: missing file: $x" && exit 1; done

dirname=$(basename $dst_dir)
exp_dir=./exp/gwpe/$dirname && mkdir -p $exp_dir
stft_opts=$(cat $stft_conf | xargs)

split_wav_scp="" && for n in $(seq $nj); do split_wav_scp="$split_wav_scp $exp_dir/wav.$n.scp"; done

./utils/split_scp.pl $wav_scp $split_wav_scp

mkdir -p $dst_dir
$cmd JOB=1:$nj $exp_dir/log/run_gwpe.JOB.log \
  ./scripts/sptk/apply_gwpe.py \
  $stft_opts --num-iters $iters \
  --context $context \
  --taps $taps --delay $delay \
  $exp_dir/wav.JOB.scp \
  $dst_dir

echo "$0: Run gwpe algorithm done"
