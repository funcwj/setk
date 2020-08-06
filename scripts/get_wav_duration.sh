#!/usr/bin/env bash

# wujian@2020

set -eu

nj=20
cmd="run.pl"
output="sample"

. ./path.sh
. ./utils/parse_options.sh || exit 1

[ $# -ne 2 ] && echo "Script format error: $0 <data-dir> <log-dir>" && exit 1

data_dir=$(cd $1; pwd)
log_dir=$2 && mkdir -p $log_dir

[ ! -f $data_dir/wav.scp ] && echo "Missing $data_dir/wav.scp" && exit 1

split_scp=""
for n in $(seq $nj); do split_scp="$split_scp $log_dir/wav.$n.scp"; done

./utils/split_scp.pl $data_dir/wav.scp $split_scp || exit 1

$cmd JOB=1:$nj $log_dir/log/get_wav_dur.JOB.log \
    ./utils/wav_duration.py --output $output $log_dir/wav.JOB.scp $log_dir/dur.JOB

cat $log_dir/dur.* | sort -k1 > $data_dir/utt2dur
echo "$0: Get duration for $1 done"