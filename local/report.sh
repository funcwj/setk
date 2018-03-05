#!/bin/bash

[ $# -ne 1 ] && echo "format error: $0 <nn-log-dir>" && exit 1

log_dir=$1

valid_list=`find $log_dir -name "compute_prob_valid.*.log" | \
    awk -F "/" '{split($NF, a, "."); print a[2]"\t"$0}' \
    | sort -k1 -n | awk '{print $2}'`

for x in $valid_list; do cat $x | grep object $x | awk '{print $8}'; done | awk '{print NR"\t"$0}'
