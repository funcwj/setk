#!/bin/bash
# wujian@2018

set -ue

nj=10
mask=irm
cmd=run.pl
mask_config=conf/mask.conf
compress=true
noise=false

. ./path.sh || exit 1

. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "format error: $0 <data-in-dir> <log-out-dir> <mask-out-dir>" && exit 1

data_dir=$1
logg_dir=$2
mask_dir=$(cd $3; pwd)

name=$(basename $data_dir)

for x in noise.scp clean.scp; do [ ! -f $data_dir/$x ] && echo "$data_dir/$x do not exists!" && exit 1; done
for x in $logg_dir $mask_dir; do [ ! -d $x ] && mkdir -p $x; done

for x in noise clean; do 
    dest_parts=$(for n in `seq $nj`; do echo $logg_dir/${name}_$x.$n.scp; done)
    cat $data_dir/$x.scp | ./utils/split_scp.pl - $dest_parts
done

if ! $noise; then
    echo "$0: compute $mask for clean parts in $data_dir..."
    $cmd JOB=1:$nj $logg_dir/compute_${name}_${mask}.JOB.log \
        compute-masks --verbose=2 --config=$mask_config scp:$logg_dir/${name}_noise.JOB.scp \
        scp:$logg_dir/${name}_clean.JOB.scp ark:- \| \
        copy-feats --compress=$compress ark:- \
        ark,scp:$mask_dir/${name}_${mask}.JOB.ark,$mask_dir/${name}_${mask}.JOB.scp || exit 1;
else
    echo "$0: compute $mask for noise parts in $data_dir..."
    $cmd JOB=1:$nj $logg_dir/compute_${name}_${mask}.JOB.log \
        compute-masks --verbose=2 --config=$mask_config scp:$logg_dir/${name}_clean.JOB.scp \
        scp:$logg_dir/${name}_noise.JOB.scp ark:- \| \
        copy-feats --compress=$compress ark:- \
        ark,scp:$mask_dir/${name}_${mask}.JOB.ark,$mask_dir/${name}_${mask}.JOB.scp || exit 1;
fi

for n in `seq $nj`; do cat $mask_dir/${name}_${mask}.$n.scp; done | sort -k1 > $data_dir/masks.scp

echo "$0: Compute $mask for $data_dir done!"


