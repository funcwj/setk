#!/usr/bin/env bash

# wujian@2018

set -eu

nj=40
cmd="run.pl"
fs=16000
speed=340
geometry="linear"
linear_topo="0,0.2,0.4,0.6"
circular_radius=0.5
circular_center=false
circular_around=6
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
  echo "  --geometry    <linear|circular>   # geometry of the array, (default=$geometry)"
  echo "  --linear-topo <topo>              # topology for linear microphone arrays, (default=$linear_topo)"
  echo "  --circular-center <center>        # is there a microphone in the center, (default=$circular_center)"
  echo "  --circular-radius <radius>        # radius of the array, (default=$circular_radius)"
  echo "  --circular-around <around>        # number microphones around the center, (default=$circular_around)"
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
beamformer_opts="--sr $fs --speed $speed --geometry $geometry"

case $geometry in 
  "linear" )
    beamformer_opts="$beamformer_opts --linear-topo $linear_topo"
    ;;
  "circular" )    
    beamformer_opts="$beamformer_opts --circular-around $circular_around"
    beamformer_opts="$beamformer_opts --circular-radius $circular_radius"
    beamformer_opts="$beamformer_opts --circular_center $circular_center"
    ;;
  * )
    echo "$0: Unknown type of geometry: $geometry" && exit 1
    ;;
esac
if [ ! -z $utt2doa ]; then
  echo "$0: Run DS beamformer on $utt2doa ..."
  mkdir -p $dst_dir/doa${doa}_$dirname
  $cmd JOB=1:$nj $exp_dir/run_ds.JOB.log \
    ./scripts/sptk/apply_ds_beamformer.py \
    $stft_opts $beamformer_opts \
    --utt2doa $utt2doa \
    $exp_dir/wav.JOB.scp \
    $dst_dir
  echo "$0: Run delay and sum beamformer -- $utt2doa done"
else
  dirname=$(basename $1)
  for doa in $doa_list; do
    echo "$0: Run DS beamformer on DoA $doa ..."
    mkdir -p $dst_dir/doa${doa}_$dirname
    $cmd JOB=1:$nj $exp_dir/$dirname.$doa.ds.JOB.log \
      ./scripts/sptk/apply_ds_beamformer.py \
      $stft_opts $beamformer_opts \
      --doa $doa \
      $exp_dir/wav.JOB.scp \
      $dst_dir/doa${doa}_$dirname
  done
  echo "$0: Run delay and sum beamformer -- $doa_list done"
fi


