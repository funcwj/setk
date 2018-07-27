#!/usr/bin/env bash
# wujian@2018

set -eu

beamformer=mvdr
clear_mask=false
chunk_width=140
extra_left_context=40
extra_right_context=40
iter=80
nj=40
feature=fbank
cmd=run.pl
ivector_dir=

. ./path.sh

. ./utils/parse_options.sh || exit 1

[ $# -ne 4 ] && echo "format error: $0 <mask-channel-dir> <multiple-channel-scripts> <model-dir> <enhan-dir>" && exit 1

data_dir=$(cd $1; pwd)
mul_scp=$2
nnet_dir=$3
enhan_dir=$4

./scripts/prepare_feats.sh \
  --nj $nj \
  --feature $feature $data_dir

./scripts/mono_mask_enhance.sh \
  --iter $iter \
  --nj $nj \
  --chunk-width $chunk_width \
  --extra-left-context $extra_right_context \
  --extra-right-context $extra_right_context \
  --online-ivector-dir "$ivector_dir" \
  $data_dir $nnet_dir $enhan_dir

./utils/split_scp.pl \
  $enhan_dir/mask/output.scp \
  $enhan_dir/mask/mask.{1..40}.scp

./utils/split_scp.pl \
  $mul_scp $enhan_dir/mask/wav.{1..40}.scp

$cmd JOB=1:40 ./exp/adaptbeam/run_${beamformer}.JOB.log \
./sptk/apply_adaptive_beamformer.py \
  --beamformer $beamformer \
  --frame-length 1024 --frame-shift 256 \
  $enhan_dir/mask/wav.JOB.scp \
  $enhan_dir/mask/mask.JOB.scp \
  $enhan_dir

if $clear_mask; then rm -rf $enhan_dir/mask; fi

echo "$0: do $beamformer done!"

