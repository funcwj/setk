#!/usr/bin/env bash

# wujian@2018

set -eu

cmd="run.pl"
nj=40
# egs:
# --frame-length 1024
# --frame-shift 256
# --window hann
# --num-bins 40
# --sample-frequency 16000
# --min-freq 0
# --max-freq 8000
# --center
# --apply-log
fbank_conf=conf/fbank_librosa.conf

compress=true

echo "$0 $@"

. ./path.sh

. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "Script format error: $0 <data-dir> <log-dir> <fbank-dir>" && exit 1

src_dir=$(cd $1; pwd)
dst_dir=$3

[ ! -f $src_dir/wav.scp ] && echo "$0: missing wav.scp in $src_dir" && exit 1

fbank_opts=$(cat $fbank_conf | xargs)

exp_dir=$2 && mkdir -p $exp_dir
mkdir -p $dst_dir && dst_dir=$(cd $dst_dir; pwd)

wav_split_scp=""
for n in $(seq $nj); do wav_split_scp="$wav_split_scp $exp_dir/wav.$n.scp"; done

./utils/split_scp.pl $src_dir/wav.scp $wav_split_scp || exit 1

name="fbank"

dir=$(basename $src_dir)

$cmd JOB=1:$nj $exp_dir/log/compute_fbank_$dir.JOB.log \
  ./scripts/sptk/compute_fbank.py \
  $fbank_opts $exp_dir/wav.JOB.scp - \| \
  copy-feats --compress=$compress ark:- \
  ark,scp:$dst_dir/$dir.$name.JOB.ark,$dst_dir/$dir.$name.JOB.scp 

cat $dst_dir/$dir.$name.*.scp | sort -k1 > $src_dir/feats.scp

echo "$0: Compute fbank using librosa done"

