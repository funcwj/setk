#!/usr/bin/env bash

# wujian@2018

set -eu

nj=40
cmd="run.pl"
compress=true

stft_conf=conf/stft.conf
fs=16000
num_doa=121
d=0.07
n=6
diag_pair="0,3\;1,4\;2,5"

echo "$0 $@"

function usage {
  echo "Options:"
  echo "  --nj           <nj>               # number of jobs to run parallel, (default=$nj)"
  echo "  --cmd          <run.pl|queue.pl>  # how to run jobs, (default=$cmd)"
  echo "  --compress     <true|false>       # compress feature or not, (default=$compress)"
  echo "  --stft-conf    <stft-conf>        # stft configurations files, (default=$stft_conf)"
  echo "  --fs           <fs>               # sample frequency for source wave, (default=$fs)"
  echo "  --num-doa      <num-doa>          # doa resolution, (default=$num_doa)"
  echo "  --d            <D>                # diameter of circular array, (default=$d)"
  echo "  --n            <N>                # number of arrays, (default=$n)"
  echo "  --diag-pair    <diag-pair>        # diagonal pairs to compute gcc-phat, (default=$diag_pair)"
}

. ./path.sh
. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "Script format error: $0 <data-dir> <log-dir> <feats-dir>" && usage && exit 1

src_dir=$(cd $1; pwd)
dst_dir=$3

for x in $src_dir/wav.scp $stft_conf; do [ ! -f $x ] && echo "$0: missing file: $x" && exit 1; done

srp_opts=$(cat $stft_conf | xargs)
srp_opts="$srp_opts --n $n --d $d --sr $fs --num-doa $num_doa --diag-pair $diag_pair"

echo "$0: Compute srp circular features..."

exp_dir=$2 && mkdir -p $exp_dir
mkdir -p $dst_dir && dst_dir=$(cd $3; pwd)

wav_split_scp=""
for n in $(seq $nj); do wav_split_scp="$wav_split_scp $exp_dir/wav.$n.scp"; done
./utils/split_scp.pl $src_dir/wav.scp $wav_split_scp || exit 1

name=$(basename $src_dir)

if $compress ; then
  $cmd JOB=1:$nj $exp_dir/log/compute_srp.JOB.log \
    ./scripts/sptk/compute_circular_srp.py $srp_opts \
    $exp_dir/wav.JOB.scp - \| \
    copy-feats --compress=$compress ark:- \
    ark,scp:$dst_dir/$name.srp.JOB.ark,$dst_dir/$name.srp.JOB.scp
else
  $cmd JOB=1:$nj $exp_dir/log/compute_srp.JOB.log \
    ./scripts/sptk/compute_circular_srp.py $srp_opts \
    --scp $dst_dir/$name.srp.JOB.scp \
    $exp_dir/wav.JOB.scp $dst_dir/$name.srp.JOB.ark
fi 

cat $dst_dir/$name.srp.*.scp | sort -k1 > $src_dir/srp.scp

echo "$0: Compute srp circular features done"

