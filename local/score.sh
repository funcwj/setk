#!/usr/bin/env bash
# wujian@2018

# compute PESQ values

cur_num=0
nj=40

[ $# -ne 1 ] && echo "format error: $0 <test-dir>" && exit 1

test_dir=$1

[ ! -f $test_dir/clean.scp ] && echo "$0: no reference wave exists" && exit 1

cat `find $test_dir -name "dst.scp"` | sort -k1 > $test_dir/enhan.scp

for x in enhan.scp wav.scp; do
    join $test_dir/clean.scp $test_dir/$x | awk '{print $2" "$3}' | \
        while read record; do
            ./local/pesq $record +16000 > /dev/null &
            ((cur_num++))
            if [ $((cur_num % $nj)) -eq 0 ] ; then wait; fi
        done
    sed "1d" _pesq_results.txt | awk '{sum += $2}END{print sum / NR}' >> $test_dir/score && rm -rf *.txt
done
