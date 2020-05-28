#!/usr/bin/env bash

# wujian@2018

set -eu

nj=40
cmd="run.pl"
compress=true

stft_conf=conf/stft.conf
mask_format="kaldi"

echo "$0 $@"

function usage {
  echo "Options:"
  echo "  --nj          <nj>                  # number of jobs to run parallel, (default=$nj)"
  echo "  --cmd         <run.pl|queue.pl>     # how to run jobs, (default=$cmd)"
  echo "  --compress    <true|false>          # compress feature or not, (default=$compress)"
  echo "  --stft-conf   <stft-conf>           # stft configurations files, (default=$stft_conf)"
  echo "  --mask-format <kaldi|numpy>         # load masks from np.ndarray instead, (default=$mask_format)"
}

. ./path.sh
. ./utils/parse_options.sh || exit 1

[ $# -ne 4 ] && echo "Script format error: $0 <data-dir> <mask-dir/mask-scp> <log-dir> <feats-dir>" && usage && exit 1

src_dir=$(cd $1; pwd)
dst_dir=$4

for x in $src_dir/wav.scp $stft_conf; do [ ! -f $x ] && echo "$0: Missing file: $x..." && exit 1; done

echo "$0: Compute directional features for $1..."

exp_dir=$3 && mkdir -p $exp_dir
mkdir -p $dst_dir && dst_dir=$(cd $4; pwd)

mask_scp_or_dir=$2
if [ -d $mask_scp_or_dir ]; then
  [ $mask_format != "numpy" ] && echo "$0: $mask_scp_or_dir is a directory, expected to set --mask-format numpy" && exit 1
  find $mask_scp_or_dir -name "*.npy" | awk -F '/' '{printf("%s\t%s\n", $NF, $0)}' | \
    sed 's:\.npy::' | sort -k1 > $exp_dir/masks.scp
  echo "$0: Got $(cat $exp_dir/masks.scp | wc -l) numpy's masks"
else
  cp $mask_scp_or_dir $exp_dir/masks.scp
fi

awk '{print $1}' $exp_dir/masks.scp | ./utils/filter_scp.pl -f 1 - $src_dir/wav.scp | sort -k1 > $exp_dir/wav.scp
echo "$0: Reduce $(cat $src_dir/wav.scp | wc -l) utterances to $(cat $exp_dir/wav.scp | wc -l)"

wav_split_scp="" && for n in $(seq $nj); do wav_split_scp="$wav_split_scp $exp_dir/wav.$n.scp"; done
./utils/split_scp.pl $exp_dir/wav.scp $wav_split_scp

name="df"
dir=$(basename $src_dir)

df_opts=$(cat $stft_conf | xargs)
df_opts="$df_opts --mask-format $mask_format"

if $compress ; then
  $cmd JOB=1:$nj $exp_dir/log/compute_df_$dir.JOB.log \
    ./scripts/sptk/compute_linear_df_mask.py \
    $df_opts $exp_dir/wav.JOB.scp \
    $exp_dir/masks.scp - \| copy-feats --compress=$compress ark:- \
    ark,scp:$dst_dir/$dir.$name.JOB.ark,$dst_dir/$dir.$name.JOB.scp
else
  $cmd JOB=1:$nj $exp_dir/log/compute_df_$dir.JOB.log \
    ./scripts/sptk/compute_linear_df_mask.py $df_opts \
    --scp $dst_dir/$dir.$name.JOB.scp \
    $exp_dir/wav.JOB.scp $exp_dir/masks.scp \
    $dst_dir/$dir.$name.JOB.ark
fi

cat $dst_dir/$dir.$name.*.scp | sort -k1 > $src_dir/df.scp

echo "$0: Compute directional features done"

