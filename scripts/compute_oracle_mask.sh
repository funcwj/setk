#!/usr/bin/env bash

set -eu

mask="irm"
stft_conf=conf/stft.conf

compress=true
cmd="run.pl"
nj=40


. ./path.sh
. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "Script format error: $0 <data-dir> <log-dir> <mask-dir>" && exit 1

data_dir=$(cd $1; pwd)
mask_dir=$3

for f in noise.scp clean.scp wav.scp; do
    [ ! -f $data_dir/$f ] && echo "$0: missing $f in $data_dir" && exit 1
done

exp_dir=$2 && mkdir -p $exp_dir
mkdir -p $mask_dir && mask_dir=$(cd $mask_dir; pwd)

split_speech_wav=""
for n in $(seq $nj); do split_speech_wav="$split_speech_wav $exp_dir/clean.$n.scp"; done

./utils/split_scp.pl $data_dir/clean.scp $split_speech_wav || exit 1

mask_opts=$(cat $stft_conf | xargs)
name=$(basename $data_dir)

$cmd JOB=1:$nj $exp_dir/log/compute_mask_$name.JOB.log \
  ./scripts/sptk/compute_mask.py $mask_opts \
  $exp_dir/clean.JOB.scp $data_dir/noise.scp - \| \
  copy-feats --compress=$compress ark:- \
  ark,scp:$mask_dir/$name.$mask.JOB.ark,$mask_dir/$name.$mask.JOB.scp

cat $mask_dir/$name.$mask.*.scp | sort -k1 > $data_dir/mask.scp

echo "$0: compute $mask done"


