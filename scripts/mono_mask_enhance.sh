#!/usr/bin/env bash
# wujian@2018

# Mask based single channel speech enhancement
# X_{enhan} = ISTFT(STFT(X_{noisy}) \dot M)
# M is mask estimated from nnet, could be IBM, IAM, IRM, PSM .etc

set -eu

nj=10
stage=1
cmd=run.pl
keep_mask=true
# using for wav-separate
mask_conf=conf/mask.conf
online_ivector_dir=
chunk_width=150
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1

. ./path.sh

. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "format error: $0 <data-dir> <nnet-dir> <enhan-dir>" && exit 1

src_dir=$1
mdl_dir=$2
dst_dir=$3

[ ! -d $dst_dir ] && mkdir -p $dst_dir

dst_dir=$(cd $dst_dir; pwd)

./utils/validate_data_dir.sh --no-text $src_dir || exit 1

# compute mask
if [ $stage -le 1 ]; then
  echo "$0: compute mask for $src_dir using nnet in $mdl_dir" 
  ./steps/nnet3/compute_output.sh --nj $nj --frames-per-chunk $chunk_width \
    --extra-left-context $extra_left_context --extra-right-context $extra_right_context \
    --extra-left-context-initial $extra_left_context_initial \
    --extra-right-context-final $extra_right_context_final \
    --online-ivector-dir "$online_ivector_dir" \
    $src_dir $mdl_dir $dst_dir/mask
fi

sdata=$src_dir/split$nj 

[ ! -d $sdata ] && echo "**error** $0: run stage 1 first" && exit 1

if [ $stage -le 2 ]; then
  echo "$0: estimate enhanced wave using masks in $dst_dir/mask"    
  for x in $(seq $nj); do
      awk -v dst_dir=$dst_dir '{printf("%s\t%s/%s.wav\n", $1, dst_dir, $1)}' \
        $sdata/$x/wav.scp > $sdata/$x/enh.scp
  done
  $cmd JOB=1:$nj exp/mono_enhan/wav_separate.JOB.log \
    wav-separate --config=$mask_conf scp:$sdata/JOB/wav.scp \
    scp:$dst_dir/mask/output.JOB.scp scp:$sdata/JOB/enh.scp
  [ ! $keep_mask ] && rm -rf $dst_dir/mask && echo "$0: clear mask in $dst_dir/mask"
fi

echo "$0: done!"













