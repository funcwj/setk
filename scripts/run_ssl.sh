#!/usr/bin/env bash

# wujian@2020

set -eu

nj=40
cmd="run.pl"
stft_conf=conf/stft.conf
backend="ml"
srp_pair=""
doa_range="0,180"
mask_scp=""
mask_eps=-1
output="degree"
chunk_len=-1
look_back=125

echo "$0 $@"

function usage {
  echo "Options:"
  echo "  --nj           <nj>                  # number of jobs to run parallel, (default=$nj)"
  echo "  --cmd          <run.pl|queue.pl>     # how to run jobs, (default=$cmd)"
  echo "  --backend      <ml|srp>              # backend algorithm to choose, (default=$backend)"
  echo "  --srp-pair     <srp-pair>            # microphone index pair to compute srp response, (default=$srp_pair)"
  echo "  --doa-range    <doa-range>           # doa range, (default=$doa_range)"
  echo "  --output       <radian|degree>       # output type of the DoA, (default=$output)"
  echo "  --mask-scp     <mask-scp>            # scripts of the speaker masks (default=$mask_scp)"
  echo "  --mask-eps     <mask-eps>            # value of eps used in winner-take-all (default=$mask_eps)"

}

. ./path.sh
. ./utils/parse_options.sh || exit 1 

[ $# -ne 3 ] && echo "Script format error: $0 <src-dir> <steer-vector> <doa-scp>" && usage && exit 1

wav_scp=$1/wav.scp
doa_scp=$3
steer_vector=$2

for x in $wav_scp $stft_conf; do [ ! -f $x ] && echo "$0: missing file: $x" && exit 1; done

dirname=$(basename $1)
exp_dir=./exp/ssl/$dirname && mkdir -p $exp_dir

ssl_opts=$(cat $stft_conf | xargs)
[ ! -z $srp_pair ] && ssl_opts="$ssl_opts --srp-pair $srp_pair"
[ ! -z $mask_scp ] && ssl_opts="$ssl_opts --mask-scp $mask_scp --mask-eps $mask_eps"
[ $chunk_len -lt 1 ] && ssl_opts="$ssl_opts --chunk-len $chunk_len --look-back $look_back"

split_wav_scp="" && for n in $(seq $nj); do split_wav_scp="$split_wav_scp $exp_dir/wav.$n.scp"; done

./utils/split_scp.pl $wav_scp $split_wav_scp

$cmd JOB=1:$nj $exp_dir/log/do_ssl.JOB.log \
  ./scripts/sptk/do_ssl.py $ssl_opts \
  --backend $backend \
  --doa-range $doa_range \
  --output $output \
  $exp_dir/wav.JOB.scp \
  $steer_vector \
  $exp_dir/doa.JOB.scp

cat $exp_dir/doa.*.scp | sort -k1 > $doa_scp

echo "$0: Do SSL for $wav_scp done (backend = $backend)"