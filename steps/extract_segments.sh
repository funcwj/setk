#!/bin/bash
# wujian@2018

set -ue

nj=32
cmd="run.pl"

. ./path.sh
. parse_options.sh || exit 1

[ $# -ne 2 ] && echo "format error: $0 <data-dir> <segment-dir>" && exit 1

data_dir=$(cd $1; pwd)
segment_dir=$2

mkdir -p $segment_dir && segment_dir=$(cd $segment_dir; pwd)

split_id=$(seq $nj)
mkdir -p $data_dir/split$nj

split_segments=""
for n in $split_id; do split_segments="$split_segments $data_dir/split$nj/$n.seg"; done

./utils/split_scp.pl $data_dir/segments $split_segments

for n in $split_id; do 
  awk -v dst_dir=$segment_dir '{print $1"\t"dst_dir"/"$1".wav"}' \
  $data_dir/split$nj/$n.seg > $data_dir/split$nj/$n.scp; 
done

$cmd JOB=1:$nj exp/segment/extract_segment.JOB.log \
   extract-segments scp:$data_dir/wav.scp \
   $data_dir/split$nj/JOB.seg \
   scp:$data_dir/split$nj/JOB.scp 

echo "$0: Extract segments from $data_dir done"
