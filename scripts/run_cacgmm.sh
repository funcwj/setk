#!/usr/bin/env bash

# wujian@2019

set -eu

nj=40
cmd="run.pl"
epoches=50
stft_conf=conf/stft.conf
init_mask=
num_classes=3
mask_format="numpy"

echo "$0 $@"

function usage {
  echo "Options:"
  echo "  --nj          <nj>                  # number of jobs to run parallel, (default=$nj)"
  echo "  --cmd         <run.pl|queue.pl>     # how to run jobs, (default=$cmd)"
  echo "  --stft-conf   <stft-conf>           # stft configurations files, (default=$stft_conf)"
  echo "  --epoches     <epochs>              # number of epoches to run cacgmm, (default=$epoches)"
  echo "  --num-classes <num-classes>         # number of the cluster used in cacgmm model, (default=$num_classes)"
  echo "  --init-mask   <init-mask>           # dir or script for mask initialization, (default=$init_mask)"
  echo "  --mask-format <kaldi|numpy>         # mask storage type, (default=$mask_format)"
}

. ./path.sh
. ./utils/parse_options.sh || exit 1 

[ $# -ne 2 ] && echo "Script format error: $0 <wav-scp> <dst-dir>" && usage && exit 1

wav_scp=$1
dst_dir=$2

for x in $wav_scp $stft_conf; do [ ! -f $x ] && echo "$0: missing file: $x" && exit 1; done

dirname=$(basename $dst_dir)
exp_dir=./exp/cacgmm/$dirname && mkdir -p $exp_dir
stft_opts=$(cat $stft_conf | xargs)

split_wav_scp="" && for n in $(seq $nj); do split_wav_scp="$split_wav_scp $exp_dir/wav.$n.scp"; done

./utils/split_scp.pl $wav_scp $split_wav_scp

cacgmm_opts="--num-epoches $epochs --num-classes $num_classes"
[ ! -z $init_mask ] && cacgmm_opts="$cacgmm_opts --init-mask $init_mask --mask-format $mask_format"

mkdir -p $dst_dir
$cmd JOB=1:$nj $exp_dir/log/run_cacgmm.JOB.log \
  ./scripts/sptk/estimate_cacgmm_masks.py \
  $stft_opts $cacgmm_opts \
  $exp_dir/wav.JOB.scp \
  $dst_dir

echo "$0: Estimate mask using Cacgmm (K = $num_classes) methods done"
