#!/usr/bin/env bash

# wujian@2018

set -eu

nj=40
cmd="run.pl"

mode=3
fs=16000
chunk_size=20
cache_size=5

function usage {
  echo "Options:"
  echo "  --nj          <nj>                 # number of jobs to run parallel, (default=40)"
  echo "  --cmd         <run.pl|queue.pl>    # how to run jobs, (default=run.pl)"
  echo "  --fs          <fs>                 # sample rate for input wave, (default=16000)"
  echo "  --mode        <0-3>                # vad mode (0->3 less->more aggressive) used in webrtc, (default=3)"
  echo "  --chunk-size  <chunk-size>         # frame length in ms, must be x10, (default=20)"
  echo "  --cache-size  <cache-size>         # number of frames remembered in history, (default=5)"
}

echo "$0 $@"

. ./path.sh
. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "Script format error: $0 <wav-scp> <exp-dir> <dst-dir>" && usage && exit 1

wav_scp=$1
exp_dir=$2 && mkdir -p $exp_dir
dst_dir=$3 && mkdir -p $dst_dir

for x in $wav_scp; do [ ! -f $x ] && echo "$0: Missing $wav_scp..." && exit 1; done

split_wav_scp=""
for n in $(seq $nj); do split_wav_scp="$split_wav_scp $exp_dir/wav.$n.scp"; done

./utils/split_scp.pl $wav_scp $split_wav_scp

$cmd JOB=1:$nj $exp_dir/log/cut_silence.JOB.log \
  ./scripts/sptk/remove_sil.py \
  --mode $mode \
  --fs $fs \
  --chunk-size $chunk_size \
  --cache-size $cache_size \
  $exp_dir/wav.JOB.scp \
  $dst_dir

echo "$0: done"

