#!/usr/bin/env bash

# wujian@2018

set -eu

nj=40
cmd="run.pl"
fs=16000
speed=340
topo="0,0.2,0.4,0.6"
doa_list="30 70 110 150"
utt2doa=""
stft_conf=./conf/stft.conf

echo "$0 $@"

function usage {
  echo "Options:"
  echo "  --nj          <nj>                # number of jobs to run parallel, (default=$nj)"
  echo "  --cmd         <run.pl|queue.pl>   # how to run jobs, (default=$cmd)"
  echo "  --stft-conf   <stft-conf>         # stft configurations files, (default=$stft_conf)"
  echo "  --fs          <fs>                # sample frequency for source signal, (default=$fs)"
  echo "  --topo        <topo>              # topology for linear microphone arrays, (default=$topo)"
  echo "  --doa-list    <doa-list>          # list of DoA to be processed, (default=$doa_list)"
  echo "  --utt2doa     <utt2doa>           # utt2doa file, (default=$utt2doa)"
  echo "  --speed       <speed>             # sound speed, (default=$speed)"
}

. ./path.sh 
. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "Script format error: $0 <data-dir> <log-dir> <enhan-dir>" && usage && exit 1

wav_scp=$1/wav.scp
exp_dir=$2
dst_dir=$3

for x in $stft_conf $wav_scp; do [ ! -f $x ] && echo "$0: Missing file: $x" && exit 1; done
[ ! -d $exp_dir ] && mkdir -p $exp_dir

split_wav_scp=""
for n in $(seq $nj); do split_wav_scp="$split_wav_scp $exp_dir/wav.$n.scp"; done

./utils/split_scp.pl $wav_scp $split_wav_scp
stft_opts=$(cat $stft_conf | xargs)
beamformer_opts="--fs $fs --speed $speed --linear-topo $topo"

if [ ! -z $utt2doa ]; then
  echo "$0: Run supper-directive beamformer on $utt2doa ..."
  mkdir -p $dst_dir/doa${doa}_$dirname
  $cmd JOB=1:$nj $exp_dir/run_sd.JOB.log \
    ./scripts/sptk/apply_sd_beamformer.py \
    $stft_opts $beamformer_opts \
    --utt2doa $utt2doa \
    $exp_dir/wav.JOB.scp \
    $dst_dir
    echo "$0: Run supper-directive beamformer -- $utt2doa done"
else
  dirname=$(basename $1)
  for doa in $doa_list; do
    echo "$0: Run supper-directive beamformer on DoA $doa ..."
    mkdir -p $dst_dir/doa${doa}_$dirname
    $cmd JOB=1:$nj $exp_dir/$dirname.$doa.sd.JOB.log \
      ./scripts/sptk/apply_sd_beamformer.py \
      $stft_opts $beamformer_opts \
      --doa $doa \
      $exp_dir/wav.JOB.scp \
      $dst_dir/doa${doa}_$dirname
  done
    echo "$0: Run supper-directive beamformer -- $doa_list done"
fi