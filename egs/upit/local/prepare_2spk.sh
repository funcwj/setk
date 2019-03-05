#!/usr/bin/env bash

# for 3spk upit usage
# wujian@2018

set -eu

function prepare_wav {
    find $1 -name "*.wav" | awk -F '/' '{print $NF"\t"$0}' \
      | sed "s:\.wav::" | sort -k1
}

function prepare_perutt {
  [ ! -f $1/wav.scp ] && echo "missing $1/wav.scp" && exit 1
  awk '{print $1" "$1}' $1/wav.scp > $1/utt2spk
  awk '{print $1" "$1}' $1/wav.scp > $1/spk2utt
}

nj=40
use_psa=true

echo "$0 $@"

. ./path.sh
. ./utils/parse_options.sh

[ $# -ne 2 ] && echo "Script format error: $0 <simu-dir> <data-dir>" && exit 1

simu_dir=$1
data_dir=$2

for dir in train dev test; do
  [ -d $data_dir/$dir ] && echo "$0: clean $dir..." && rm -rf $data_dir/$dir

  for x in mix spk{1..2}; do 
    [ ! -d $simu_dir/$dir/$x ] && echo "$0: missing dir $simu_dir/$dir/$x" && exit 1
  done

  mkdir -p $data_dir/$dir/mix
  prepare_wav $simu_dir/$dir/mix > $data_dir/$dir/mix/wav.scp
  prepare_perutt $data_dir/$dir/mix


  for spk in mix; do
    echo "$0: prepare features $dir - $spk ..."
    # log spectrogram
    ./scripts/compute_librosa_spectrogram.sh \
      --nj $nj \
      --compress false \
      --stft-conf ./conf/16k.stft.conf \
      --apply-log true \
      $data_dir/$dir/$spk \
      ./exp/make_stft/2spk_upit/$dir/$spk \
      ./stft/2spk_upit/$dir/$spk
    # compute global cmvn for log-spectrum
    compute-cmvn-stats scp:$data_dir/$dir/mix/feats.scp $data_dir/$dir/mix/log_gcmvn.mat
    # linear spectrogram
    ./scripts/compute_librosa_spectrogram.sh \
      --nj $nj \
      --compress false \
      --stft-conf ./conf/16k.stft.conf \
      --apply-log false \
      $data_dir/$dir/$spk \
      ./exp/make_stft/2spk_upit/$dir/$spk \
      ./stft/2spk_upit/$dir/$spk
    # compute global cmvn for linear spectrum
    compute-cmvn-stats scp:$data_dir/$dir/mix/feats.scp $data_dir/$dir/mix/gcmvn.mat
  done

  mkdir -p $data_dir/$dir/{spk1,spk2}
  if $use_psa; then
    prepare_wav $simu_dir/$dir/spk1 > $data_dir/$dir/spk1/clean.scp 
    prepare_wav $simu_dir/$dir/spk2 > $data_dir/$dir/spk2/clean.scp
    # phase sensitive amplitude
    for spk in spk1 spk2; do
      echo "$0: prepare psa $dir - $spk ..."
      cp $data_dir/$dir/mix/wav.scp $data_dir/$dir/$spk/
      ./scripts/compute_oracle_mask.sh \
        --nj $nj \
        --mask "psa" \
        --cutoff -1 \
        --compress false \
        --stft-conf ./conf/16k.stft.conf \
        $data_dir/$dir/$spk \
        ./exp/make_stft/2spk_upit/$dir/$spk \
        ./stft/2spk_upit/$dir/$spk
    done
  else
    prepare_wav $simu_dir/$dir/spk1 > $data_dir/$dir/spk1/wav.scp
    prepare_wav $simu_dir/$dir/spk2 > $data_dir/$dir/spk2/wav.scp
    # linear spectrogram
    for spk in spk1 spk2; do
      echo "$0: prepare spectogram $dir - $spk ..."
      ./scripts/compute_librosa_spectrogram.sh \
        --nj $nj \
        --compress false \
        --stft-conf ./conf/16k.stft.conf \
        --apply-log false \
        $data_dir/$dir/$spk \
        ./exp/make_stft/2spk_upit/$dir/$spk \
        ./stft/2spk_upit/$dir/$spk
    done
  fi
done

echo "$0: done"