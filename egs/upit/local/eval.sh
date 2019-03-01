#!/usr/bin/env bash

set -eu

echo "$0 $@"

nj=40
stage=1
data_dir=$PWD/data/2spk
spatial=""

. ./path.sh
. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "Script format error: $0 <cpt-dir> <gpu-id> <enhan-dir>" && exit 1

cpt_dir=$1
gpu_id=$2
enhan_dir=$3

exp_id=$(basename $cpt_dir)

if [ $stage -le 1 ]; then
  ./nnet/compute_mask.py $cpt_dir \
    --feats $data_dir/test/mix/feats.scp \
    --gpu $gpu_id \
    --spatial "$spatial" \
    --dump-dir $enhan_dir \
    > eval.$(basename $data_dir).$exp_id.log 2>&1
fi

if [ $stage -le 2 ]; then
  for spk in $(ls $enhan_dir); do
    ./scripts/run_tf_masking.sh \
      --keep-length true \
      --numpy true \
      --stft-conf conf/16k.stft.conf \
      --nj $nj \
      $data_dir/test/mix/wav.scp \
      $enhan_dir/$spk \
      $enhan_dir/$spk
  done
fi

# generate sps/2spk/spk{1,2}.scp and eval in si-snr
# ./scripts/sptk/compute_si_snr.py \
#   data/2spk_ch1/test/spk1/clean.scp,data/2spk_ch1/test/spk2/clean.scp \
#   sps/2spk/spk1.scp,sps/2spk/spk2.scp