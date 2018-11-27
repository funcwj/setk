#!/usr/bin/env bash

# wujian@2018


set -eu

cmd="run.pl"
nj=40
epochs=20
fs=16000
stft_conf=conf/stft.conf

echo "$0 $@"

function usage {
  echo "Options:"
  echo "  --nj        <nj>                  # number of jobs to run parallel, (default=40)"
  echo "  --cmd       <run.pl|queue.pl>     # how to run jobs, (default=run.pl)"
  echo "  --stft-conf <stft-conf>           # stft configurations files, (default=conf/stft.conf)"
  echo "  --epochs    <epochs>              # number of epochs to run AuxIVA, (default=20)"
  echo "  --fs        <fs>                  # sample frequency for output wave, (default=16000)"
}

. ./path.sh || exit 1
. ./utils/parse_options.sh || exit 1

[ $# -ne 2 ] && echo "Script format error: $0 <wav-scp> <dst-dir>" && usage && exit 1

wav_scp=$1
dst_dir=$2

[ ! -d $dst_dir ] && mkdir -p $dst_dir

dirname=$(basename $dst_dir)
exp_dir=exp/auxiva/$dirname && mkdir -p $exp_dir

split_wav_scp=""
for n in $(seq $nj); do split_wav_scp="$split_wav_scp $exp_dir/wav.$n.scp"; done

./utils/split_scp.pl $wav_scp $split_wav_scp || exit 1

stft_opts=$(cat $stft_conf | xargs)

mkdir -p $dst_dir
$cmd JOB=1:$nj $exp_dir/log/run_auxiva.JOB.log \
  ./scripts/sptk/apply_auxiva.py \
  --sample-frequency $fs \
  --num-epochs $epochs \
  $stft_opts \
  $exp_dir/wav.JOB.scp \
  $dst_dir


echo "$0: Do auxiva for $wav_scp done"
