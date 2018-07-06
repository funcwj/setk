#!/bin/bash
# wujian@2018
# Logging loss for train/dev dataset

set -eu

[ $# -ne 1 ] && echo "format error: $0 <nn-log-dir>" && exit 1

for x in train valid; do
  find $1 -name "compute_prob_${x}.*.log" | sed 's:\(.*\)\.\(.*\)\.\(.*\):\2\t\0:' | sort -k1 -n | \
    awk '{print $2}' | xargs -n1 -I F grep objective F | awk '{print $8;}' > $1/${x}.logprob
done

paste $1/train.logprob $1/valid.logprob | awk '{printf("%s\t%f\t%f\n", NR, $1, $2);}' && rm -f $1/*.logprob