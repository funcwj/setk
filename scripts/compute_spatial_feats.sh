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
ipd_cos=true
ipd_sin=false
# msc cfg
msc_ctx=1
# srp cfg
srp_fs=16000
srp_num_doa=181
srp_topo=""
src_sample_tdoa=false

echo "$0 $@"

function usage {
  echo "Options:"
  echo "  --nj               <nj>               # number of jobs to run parallel, (default=40)"
  echo "  --cmd              <run.pl|queue.pl>  # how to run jobs, (default=run.pl)"
  echo "  --compress         <true|false>       # compress feature or not, (default=true)"
  echo "  --stft-conf        <stft-conf>        # stft configurations files, (default=conf/stft.conf)"
  echo "  --feats            <srp|ipd|msc>      # type of spatial features, (default=ipd)"
  echo "  --ipd-index        <ipd-index>        # channel index to compute ipd, (default=0,1)"
  echo "  --ipd-cos          <true|false>       # compute cosIPD instead of raw IPD, (default=false)"
  echo "  --ipd-sin          <true|false>       # paste sinIPD to cosIPD features or not, (default=false)"
  echo "  --msc-ctx          <msc-ctx>          # length of context for MSC computation, (default=1)"
  echo "  --srp-fs           <srp-fs>           # sample frequency for source wave, (default=16000)"
  echo "  --srp-topo         <srp-topo>         # microphone topo description, (default="")"
  echo "  --srp-num-doa      <num-doa>          # doa resolution, (default=181)"
  echo "  --srp-sample-tdoa  <true|false>       # sample tdoa instead of doa, (default=false)" 
}

. ./path.sh
. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "Script format error: $0 <data-dir> <log-dir> <feats-dir>" && usage && exit 1

src_dir=$(cd $1; pwd)
dst_dir=$3

for x in $src_dir/wav.scp $stft_conf; do [ ! -f $x ] && echo "$0: missing file: $x" && exit 1; done

spatial_opts=$(cat $stft_conf | xargs)
case $feats in 
  "ipd" )
    spatial_opts="$spatial_opts --ipd.index $ipd_index"
    $ipd_cos && spatial_opts="$spatial_opts --ipd.cos"
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

if $compress ; then
  $cmd JOB=1:$nj $exp_dir/log/compute_$feats.JOB.log \
    ./scripts/sptk/compute_spatial_feats.py \
    --type $feats $spatial_opts \
    $exp_dir/wav.JOB.scp - \| \
    copy-feats --compress=$compress ark:- \
    ark,scp:$dst_dir/$name.$feats.JOB.ark,$dst_dir/$name.$feats.JOB.scp
else
  $cmd JOB=1:$nj $exp_dir/log/compute_$feats.JOB.log \
    ./scripts/sptk/compute_spatial_feats.py \
    --type $feats $spatial_opts \
    --scp $dst_dir/$name.$feats.JOB.scp \
    $exp_dir/wav.JOB.scp $name.$feats.JOB.ark
fi 

cat $dst_dir/$name.$feats.*.scp | sort -k1 > $src_dir/$feats.scp

echo "$0: Compute $feats spatial features done"

