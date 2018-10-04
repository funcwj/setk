#!/bin/bash
# wujian@2018

set -ue


nj=32
cmd="run.pl"

. ./path.sh
. parse_options.sh || exit 1

[ $# -ne 2 ] && echo "format error: $0 <data-dir> <segment-dir>" && exit 1

data_dir=$(cd $1; pwd)
segment_dir=$2 && mkdir -p $segment_dir && segment_dir=$(cd $segment_dir; pwd)

./utils/split_data.sh $data_dir $nj

sdata_dir=$data_dir/split$nj

for x in `seq $nj`; do 
  awk -v dst_dir=$segment_dir '{print $1"\t"dst_dir"/"$1".wav"}' \
  $sdata_dir/$x/segments > $sdata_dir/$x/dst.scp; 
done

$cmd JOB=1:$nj exp/segment/extract_segment.JOB.log \
   extract-segments scp:$sdata_dir/JOB/wav.scp \
   $sdata_dir/JOB/segments scp:$sdata_dir/JOB/dst.scp 

echo "$0: extract segments from $data_dir done"
