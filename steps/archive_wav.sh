#!/usr/bin/env bash

# wujian@2020

# Archive wav.scp to wav.ark

set -eu

nj=32
cmd="run.pl"

. ./path.sh
. ./utils/parse_options.sh || exit 1

[ $# -ne 2 ] && echo "format error: $0 <data-dir> <ark-dir>" && exit 1

data_dir=$(cd $1; pwd)
ark_dir=$2

[ ! -f $data_dir/wav.scp ] && echo "$0: Missing wav.scp in $data_dir" && exit 

mkdir -p $ark_dir && ark_dir=$(cd $ark_dir; pwd)

split_id=$(seq $nj)
mkdir -p $data_dir/split$nj

split_wav_scp=""
for n in $split_id; do split_wav_scp="$split_wav_scp $data_dir/split$nj/wav.$n.scp"; done

./utils/split_scp.pl $data_dir/wav.scp $split_wav_scp

exp=$(basename $data_dir)
$cmd JOB=1:$nj exp/wav_copy/$exp/wav_copy.JOB.log \
   wav-copy scp:$data_dir/split$nj/wav.JOB.scp \
   ark,scp:$ark_dir/wav.JOB.ark,$ark_dir/wav.JOB.scp

echo "$0: Archive wav.scp from $data_dir to $ark_dir done"