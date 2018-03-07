#!/bin/bash
# wujian@2018

set -ue

. ./path.sh || exit 1

train_dir=data/spectrum/noisy.gmvn
exp_dir=exp/tune/c3

target_dir=data/spectrum/clean.gmvn
target_scp=$target_dir/feats.scp

train_stage=-1
train_cmd="run.pl"

initial_effective_lrate=0.0001
final_effective_lrate=0.00005
num_epochs=20
num_jobs_initial=1
num_jobs_final=1
remove_egs=false
use_gpu=true

minbatch=256
momentum=0.8
# samples_per_iter=80000

stage=1
egs_opts="--nj 20"
egs_dir=./exp/tune/c2/egs
cmvn_opts="--norm-vars=false --norm-means=false"
nj=20

. parse_options.sh || exit 1

# it's a RNN model
if [ $stage -eq 1 ]; then
    echo "$0: config networks..."
    mkdir -p $exp_dir/configs
    stft_dim=$(feat-to-dim scp:$train_dir/feats.scp -)
    cat <<EOF > $exp_dir/configs/network.xconfig
input dim=$stft_dim name=input
fast-lstm-batchnorm-layer name=lstm1 cell-dim=512 
fast-lstm-batchnorm-layer name=lstm2 cell-dim=512 
fast-lstm-batchnorm-layer name=lstm3 cell-dim=512 
fast-lstm-batchnorm-layer name=lstm4 cell-dim=512 
# a version of modified output-layer config in steps/libs/nnet3/xconfig/basic_layers.py
output-layer name=output input=lstm4 dim=$stft_dim include-sigmoid=true masked-input=input objective-type=quadratic
EOF
    steps/nnet3/xconfig_to_configs.py --xconfig-file $exp_dir/configs/network.xconfig --config-dir $exp_dir/configs/
fi

if [ $stage -eq 2 ]; then
    echo "$0: training spectrum prediction networks..."
    steps/nnet3/train_raw_rnn.py --stage=$train_stage \
        --cmd=$train_cmd \
        --feat.cmvn-opts="$cmvn_opts" \
        --trainer.num-epochs $num_epochs \
        --trainer.optimization.num-jobs-initial $num_jobs_initial \
        --trainer.optimization.num-jobs-final $num_jobs_final \
        --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
        --trainer.optimization.final-effective-lrate $final_effective_lrate \
        --trainer.rnn.num-chunk-per-minibatch $minbatch \
        --trainer.optimization.momentum $momentum \
        --cleanup.remove-egs $remove_egs \
        --targets-scp $target_scp \
        --feat-dir $train_dir \
        --use-gpu $use_gpu \
        --egs.opts "$egs_opts" \
        --egs.dir "$egs_dir" \
        --nj $nj \
        --dir $exp_dir >> train_cm.log 2>&1 || exit 1;
    mv train_cm.log $exp_dir
fi

test_dir=data/dev

if [ $stage -eq 3 ]; then
    echo "$0: estimate wave for $test_dir..."

    ./utils/split_data.sh $test_dir $nj && sep_dir=$test_dir/split$nj

    for n in `seq $nj`; do sed "s:isolated:enhan:g" $sep_dir/$n/wav.scp > $sep_dir/$n/dst.scp; done
    mkdir -p $(cat $sep_dir/?/dst.scp | awk '{print $2}' | awk -F "/[^/]*$" '{print $1}' | sort -u)  

    noisy_feats="ark:copy-feats scp:$sep_dir/JOB/feats.scp ark:- | apply-cmvn --norm-vars=true $train_dir/gmvn.mat ark:- ark:- |"
    clean_spect="nnet3-compute $exp_dir/final.raw \"$noisy_feats\" ark:- | apply-cmvn --reverse --norm-vars=true $target_dir/gmvn.mat ark:- ark:- |"

    # mkdir -p $exp_dir/mask
    $train_cmd JOB=1:$nj $exp_dir/estimate/wav-estimate.JOB.log wav-estimate --config=conf/stft.conf \
        "ark:$clean_spect" scp:$sep_dir/JOB/wav.scp scp:$sep_dir/JOB/dst.scp || exit 1
fi

echo "$0: Done"
