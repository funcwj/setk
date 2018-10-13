#!/usr/bin/env bash
# wujian@2018

set -eu

nj=40
stage=1
iter=final
cmd=run.pl

online_ivector_dir=
chunk_width=64
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
fbank=false

. ./path.sh

. ./utils/parse_options.sh || exit 1

[ $# -ne 2 ] && echo "format error: $0 <data-dir> <nnet-dir>" && exit 1

src_dir=$(cd $1; pwd)
mdl_dir=$(cd $2; pwd)

dir=$(basename $src_dir)
dst_dir=$mdl_dir/${dir}_mask_$iter

[ ! -d $dst_dir ] && mkdir -p $dst_dir

if [ $stage -le 1 ]; then
  echo "$0: prepare features for $src_dir"
  if $fbank; then
    ./scripts/compute_librosa_fbank.sh --nj $nj \
      $src_dir ./exp/librosa_fbank ./fbank/$dir
    ./steps/compute_cmvn_stats.sh $src_dir ./exp/librosa_fbank ./fbank/$dir
  else
    ./scripts/compute_librosa_spectrogram.sh --nj $nj \
      $src_dir ./exp/librosa_stft ./spectrogram/$dir
    ./steps/compute_cmvn_stats.sh $src_dir ./exp/librosa_stft ./spectrogram/$dir
  fi
  ./utils/validate_data_dir.sh $src_dir || exit 1
fi

# compute mask
if [ $stage -le 2 ]; then
  echo "$0: compute mask for $src_dir using nnet in $mdl_dir" 
  ./steps/nnet3/compute_output.sh --nj $nj --frames-per-chunk $chunk_width \
    --extra-left-context $extra_left_context --extra-right-context $extra_right_context \
    --extra-left-context-initial $extra_left_context_initial \
    --extra-right-context-final $extra_right_context_final \
    --online-ivector-dir "$online_ivector_dir" --iter $iter \
    $src_dir $mdl_dir $dst_dir
fi

echo "$0: done!"
