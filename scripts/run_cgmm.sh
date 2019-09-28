#!/usr/bin/env bash

# wujian@2018

set -eu

nj=40
cmd="run.pl"
epoches=20
# stft.conf example:
# --frame-length 1024
# --frame-shift 256
# --window hann
# --center true
stft_conf=conf/stft.conf
init_mask=
mask_format="numpy"

echo "$0 $@"

function usage {
  echo "Options:"
  echo "  --nj          <nj>                  # number of jobs to run parallel, (default=40)"
  echo "  --cmd         <run.pl|queue.pl>     # how to run jobs, (default=run.pl)"
  echo "  --stft-conf   <stft-conf>           # stft configurations files, (default=conf/stft.conf)"
  echo "  --epoches     <epochs>              # number of epoches to run CGMM, (default=20)"
  echo "  --init-mask   <init-mask>           # dir or script for mask initialization, (default="")"
  echo "  --mask-format <kaldi|numpy>         # mask storage type, (default=$mask_format)"
}

. ./path.sh
. ./utils/parse_options.sh || exit 1 

[ $# -ne 2 ] && echo "Script format error: $0 <wav-scp> <dst-dir>" && usage && exit 1

wav_scp=$1
dst_dir=$2

for x in $wav_scp $stft_conf; do [ ! -f $x ] && echo "$0: missing file: $x" && exit 1; done

dirname=$(basename $dst_dir)
exp_dir=./exp/cgmm/$dirname && mkdir -p $exp_dir
stft_opts=$(cat $stft_conf | xargs)

split_wav_scp="" && for n in $(seq $nj); do split_wav_scp="$split_wav_scp $exp_dir/wav.$n.scp"; done

./utils/split_scp.pl $wav_scp $split_wav_scp

cgmm_opts="--num-epoches $epochs"
[ ! -z $init_mask ] && cgmm_opts="$cgmm_opts --init-speech-mask $init_mask --mask-format $mask_format"

mkdir -p $dst_dir
$cmd JOB=1:$nj $exp_dir/log/run_cgmm.JOB.log \
  ./scripts/sptk/estimate_cgmm_masks.py \
  $stft_opts $cgmm_opts \
  $exp_dir/wav.JOB.scp \
  $dst_dir

echo "$0: Estimate mask using CGMM methods done"
