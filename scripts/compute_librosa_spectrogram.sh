#!/usr/bin/env bash

# wujian@2018

set -eu

cmd="run.pl"
nj=40

sample_normalize=true
apply_log=true
apply_pow=false
# egs:
# --frame-len 1024
# --frame-hop 256
# --center true
# --window hann
stft_conf=conf/stft.conf

compress=true

echo "$0 $@"

function usage {
  echo "Options:"
  echo "  --nj                <nj>                # number of jobs to run parallel, (default=40)"
  echo "  --cmd               <run.pl|queue.pl>   # how to run jobs, (default=run.pl)"
  echo "  --compress          <true|false>        # compress feature or not, (default=true)"
  echo "  --apply-log         <true|false>        # use log or linear spectrogram, (default=true)"
  echo "  --apply-pow         <true|false>        # use power or magnitude spectrogram, (default=false)"
  echo "  --stft-conf         <stft-conf>         # stft configurations files, (default=conf/stft.conf)"
  echo "  --sample-normalize  <true|false>        # normalize wav samples into [0, 1] or not, (default=true)"
}

. ./path.sh

. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "Script format error: $0 <data-dir> <log-dir> <spectrogram-dir>" && usage && exit 1

src_dir=$(cd $1; pwd)
dst_dir=$3

for x in $src_dir/wav.scp $stft_conf; do [ ! -f $x ] && echo "$0: missing file: $x" && exit 1; done

spectrogram_opts=$(cat $stft_conf | xargs)

spectrogram_opts="$spectrogram_opts --normalize-samples $sample_normalize"
spectrogram_opts="$spectrogram_opts --apply-log $apply_log --apply-pow $apply_pow"

exp_dir=$2 && mkdir -p $exp_dir
mkdir -p $dst_dir && dst_dir=$(cd $dst_dir; pwd)

wav_split_scp=""
for n in $(seq $nj); do wav_split_scp="$wav_split_scp $exp_dir/wav.$n.scp"; done

./utils/split_scp.pl $src_dir/wav.scp $wav_split_scp || exit 1

$apply_log && $apply_pow && echo "$0: Using log-amplitude feature instead" && exit 1

name="linear_amp_spectrogram"
$apply_log && ! $apply_pow && name="log_amp_spectrogram"
! $apply_log && $apply_pow && name="linear_pow_spectrogram"

dir=$(basename $src_dir)

if $compress ; then
  $cmd JOB=1:$nj $exp_dir/log/compute_spectrogram_$dir.JOB.log \
    ./scripts/sptk/compute_spectrogram.py \
    $spectrogram_opts $exp_dir/wav.JOB.scp - \| \
    copy-feats --compress=$compress ark:- \
    ark,scp:$dst_dir/$dir.$name.JOB.ark,$dst_dir/$dir.$name.JOB.scp 
else
  $cmd JOB=1:$nj $exp_dir/log/compute_spectrogram_$dir.JOB.log \
    ./scripts/sptk/compute_spectrogram.py $spectrogram_opts \
    --scp $dst_dir/$dir.$name.JOB.scp \
    $exp_dir/wav.JOB.scp $dst_dir/$dir.$name.JOB.ark 
fi 

cat $dst_dir/$dir.$name.*.scp | sort -k1 > $src_dir/feats.scp

echo "$0: Compute spectrogram using librosa done"

