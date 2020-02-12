#!/usr/bin/env bash

# wujian@2018

set -eu

nj=40
cmd="run.pl"

mask_format="numpy"
keep_length=false
fs=16000
stft_conf=conf/stft.conf
phase_ref=

echo "$0 $@"

function usage {
  echo "Options:"
  echo "  --nj            <nj>                # number of jobs to run parallel, (default=$nj)"
  echo "  --cmd           <run.pl|queue.pl>   # how to run jobs, (default=$cmd)"
  echo "  --stft-conf     <stft-conf>         # stft configurations files, (default=$stft_conf)"
  echo "  --mask-format   <kaldi|numpy>       # load masks from np.ndarray instead, (default=$mask_format)"
  echo "  --keep-length   <true|false>        # keep same length as original or not, (default=$keep_length)"
  echo "  --phase-ref     <phase-ref>         # use phase reference or mixture, (default=$phase_ref)"
  echo "  --fs            <fs>                # sample frequency for output wave, (default=$fs)"
}

. ./path.sh
. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "Script format error: $0 <wav-scp> <mask-dir/mask-scp> <enhan-dir>" && usage && exit 1

wav_scp=$1
enhan_dir=$3

for x in $wav_scp $stft_conf; do [ ! -f $x ] && echo "$0: missing file: $x" && exit 1; done

dirname=$(basename $enhan_dir)
exp_dir=exp/tf_masking/$dirname && mkdir -p $exp_dir

# if second parameter is a directory
if [ -d $2 ]; then
  [ $mask_format != "numpy" ] && echo "$0: $2 is a directory, expected to set --mask-format numpy" && exit 1
  find $2 -name "*.npy" | awk -F '/' '{printf("%s\t%s\n", $NF, $0)}' | \
    sed 's:\.npy::' | sort -k1 > $exp_dir/masks.scp
  echo "$0: Got $(cat $exp_dir/masks.scp | wc -l) numpy's masks"
else
  cp $2 $exp_dir/masks.scp
fi

awk '{print $1}' $exp_dir/masks.scp | ./utils/filter_scp.pl -f 1 - $wav_scp | sort -k1 > $exp_dir/wav.scp
echo "$0: Reduce $(cat $wav_scp | wc -l) utterances to $(cat $exp_dir/wav.scp | wc -l)"

split_wav_scp=""
for n in $(seq $nj); do split_wav_scp="$split_wav_scp $exp_dir/wav.$n.scp"; done

./utils/split_scp.pl $wav_scp $split_wav_scp || exit 1

mask_opts=$(cat $stft_conf | xargs)
mask_opts="$mask_opts --keep-length $keep_length"
[ ! -z $phase_ref ] && mask_opts="$mask_opts --phase-ref $phase_ref"

mkdir -p $enhan_dir
$cmd JOB=1:$nj $exp_dir/log/wav_separate.JOB.scp \
  ./scripts/sptk/wav_separate.py \
  --sample-frequency $fs \
  --mask-format $mask_format \
  $mask_opts \
  $exp_dir/wav.JOB.scp \
  $exp_dir/masks.scp \
  $enhan_dir 

echo "$0: Run TF-masking done"
