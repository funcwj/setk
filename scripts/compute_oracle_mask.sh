#!/usr/bin/env bash

set -eu

mask="irm"
# for iam(FFT-mask)/psm etc
cutoff=10
stft_conf=conf/stft.conf

compress=true
cmd="run.pl"
nj=40

echo "$0 $@"

function usage {
  echo "Options:"
  echo "  --nj        <nj>                    # number of jobs to run parallel, (default=$nj)"
  echo "  --cmd       <run.pl|queue.pl>       # how to run jobs, (default=$cmd)"
  echo "  --compress  <true|false>            # compress feature or not, (default=$compress)"
  echo "  --stft-conf <stft-conf>             # stft configurations files, (default=$stft_conf)"
  echo "  --cutoff    <cutoff>                # values to cutoff when compute iam/psm, (default=$cutoff)"
  echo "  --mask   <ibm|iam|psm|irm|psa|crm>  # type of TF-masks to compute, (default=$mask)"
}

. ./path.sh
. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "Script format error: $0 <data-dir> <log-dir> <mask-dir>" && usage && exit 1

data_dir=$(cd $1; pwd)
mask_dir=$3

denominator_scp=noise.scp
case $mask in 
  "iam"|"psm"|"psa"|"crm" )
    denominator_scp=wav.scp
    ;;
  "ibm"|"irm" )
    ;;
  * )
    echo "$0: Unknown type of mask: $mask" && exit 1
    ;;
esac

for f in clean.scp $denominator_scp; do
    [ ! -f $data_dir/$f ] && echo "$0: missing $f in $data_dir" && exit 1
done

exp_dir=$2 && mkdir -p $exp_dir
mkdir -p $mask_dir && mask_dir=$(cd $mask_dir; pwd)

split_speech_wav=""
for n in $(seq $nj); do split_speech_wav="$split_speech_wav $exp_dir/clean.$n.scp"; done

./utils/split_scp.pl $data_dir/clean.scp $split_speech_wav || exit 1

mask_opts=$(cat $stft_conf | xargs)
mask_opts="$mask_opts --mask $mask"
name=$(basename $data_dir)

$cmd JOB=1:$nj $exp_dir/log/compute_mask_$name.JOB.log \
  ./scripts/sptk/compute_mask.py $mask_opts --cutoff $cutoff \
  $exp_dir/clean.JOB.scp $data_dir/$denominator_scp - \| \
  copy-feats --compress=$compress ark:- \
  ark,scp:$mask_dir/$name.$mask.JOB.ark,$mask_dir/$name.$mask.JOB.scp

cat $mask_dir/$name.$mask.*.scp | sort -k1 > $data_dir/mask.scp

echo "$0: Compute $mask done"


