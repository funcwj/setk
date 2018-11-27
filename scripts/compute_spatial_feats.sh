#!/usr/bin/env bash

# wujian@2018

set -eu

nj=40
cmd="run.pl"
compress=true

stft_conf=conf/stft.conf
feats=ipd
# ipd cfg
ipd_index="0,1"
ipd_sin=false
# msc cfg
msc_ctx=1
# srp cfg
srp_fs=16000
srp_num_doa=181
srp_topo=""
src_sample_tdoa=false

echo "$0 $@"

. ./path.sh
. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "Script format error: $0 <data-dir> <log-dir> <feats-dir>" && exit 1

src_dir=$(cd $1; pwd)
dst_dir=$3

for x in $src_dir/wav.scp $stft_conf; do [ ! -f $x ] && echo "$0: missing file: $x" && exit 1; done

spatial_opts=$(cat $stft_conf | xargs)
case $feats in 
  "ipd" )
    spatial_opts="$spatial_opts --ipd.index $ipd_index"
    $ipd_sin && spatial_opts="$spatial_opts --ipd.sin"
    ;;
  "msc" )
    spatial_opts="$spatial_opts --msc.ctx $msc_ctx"
    ;;
  "srp" )
    $src_sample_tdoa && spatial_opts="$spatial_opts --srp.sample-tdoa"
    spatial_opts="$spatial_opts --srp.num_doa $srp_num_doa --srp.sample-rate $srp_fs --srp.topo $srp_topo"
    ;;
  * )
    echo "$0: Unknown spatial feats type: $feats" && exit 1
    ;;
esac

echo "$0: Compute $feats spatial features..."

exp_dir=$2 && mkdir -p $exp_dir
mkdir -p $dst_dir && dst_dir=$(cd $3; pwd)

wav_split_scp=""
for n in $(seq $nj); do wav_split_scp="$wav_split_scp $exp_dir/wav.$n.scp"; done
./utils/split_scp.pl $src_dir/wav.scp $wav_split_scp || exit 1

name=$(basename $src_dir)
$cmd JOB=1:$nj $exp_dir/log/compute_$feats.JOB.log \
  ./scripts/sptk/compute_spatial_feats.py \
  --type $feats $spatial_opts \
  $exp_dir/wav.JOB.scp - \| \
  copy-feats --compress=$compress ark:- \
  ark,scp:$dst_dir/$name.$feats.JOB.ark,$dst_dir/$name.$feats.JOB.scp

cat $dst_dir/$name.$feats.*.scp | sort -k1 > $src_dir/feats.scp

echo "$0: Compute $feats spatial features done"

