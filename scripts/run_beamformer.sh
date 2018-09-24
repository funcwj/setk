#!/usr/bin/env bash

# wujian@2018

set -eu

nj=40
cmd="run.pl"
center=true
transpose=true
beamformer="mvdr"

. ./path.sh
. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "format error: $0 <wav-scp> <mask-dir> <enhan-dir>" && exit 1

wav_scp=$1
mask_dir=$(cd $2; pwd)
enhan_dir=$3

exp_dir=./exp/beamformer && mkdir -p $exp_dir

find $mask_dir -name "*.npy" | awk -F '/' '{printf("%s\t%s\n", $NF, $0)}' | \
  sed 's:\.npy::' | sort -k1 > $exp_dir/masks.scp
echo "$0: Got $(cat $exp_dir/masks.scp | wc -l) numpy's masks"

awk '{print $1}' $exp_dir/masks.scp | ./utils/filter_scp.pl -f 1 - $wav_scp | sort -k1 > $exp_dir/wav.scp
echo "$0: Reduce $(cat $wav_scp | wc -l) utterances to $(cat $exp_dir/wav.scp | wc -l)"

wav_split_scp="" && for n in $(seq $nj); do wav_split_scp="$wav_split_scp $exp_dir/wav.$n.scp"; done
./utils/split_scp.pl $exp_dir/wav.scp $wav_split_scp

beamformer_opts="--beamformer $beamformer --numpy"
$transpose && beamformer_opts="$beamformer_opts --transpose-mask"
$center && beamformer_opts="$beamformer_opts --center"

$cmd JOB=1:$nj $exp_dir/log/run_beamformer.JOB.log \
  ./scripts/sptk/apply_adaptive_beamformer.py \
  $beamformer_opts \
  $exp_dir/wav.JOB.scp \
  $exp_dir/masks.scp \
  $enhan_dir

echo "$0: Run $beamformer done!"
