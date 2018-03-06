#!/bin/bash
# wujian@2018

set -ue

. ./path.sh || exit 1

train_dir=data/spectrum/noisy
exp_dir=exp/spectrum/1024_dnn2048

target_scp=data/spectrum/clean/feats.scp

train_stage=-6
train_cmd="run.pl"

initial_effective_lrate=0.0015
final_effective_lrate=0.00005
num_epochs=40
num_jobs_initial=1
num_jobs_final=2
remove_egs=true
use_gpu=true

minbatch=512
momentum=0.0
# samples_per_iter=80000

stage=1
egs_opts="--nj 20"
nj=20

. parse_options.sh || exit 1

if [ $stage -eq 1 ]; then
    echo "$0: config networks..."
    mkdir -p $exp_dir/configs
    stft_dim=$(feat-to-dim scp:$train_dir/feats.scp -)
    cat <<EOF > $exp_dir/configs/network.xconfig
input dim=$stft_dim name=input
relu-batchnorm-layer name=dnn1 dim=2048 input=Append(-2,-1,0,1,2)
relu-batchnorm-layer name=dnn2 dim=2048
relu-batchnorm-layer name=dnn3 dim=2048
relu-batchnorm-layer name=dnn4 dim=2048
output-layer name=output input=dnn4 dim=$stft_dim include-log-softmax=false include-sigmoid=false objective-type=quadratic
EOF
    steps/nnet3/xconfig_to_configs.py --xconfig-file $exp_dir/configs/network.xconfig --config-dir $exp_dir/configs/
fi

if [ $stage -eq 2 ]; then
    echo "$0: training spectrum prediction networks..."
    # --trainer.samples-per-iter $samples_per_iter \
    steps/nnet3/train_raw_dnn.py --stage=$train_stage \
        --cmd=$train_cmd \
        --feat.cmvn-opts="--norm-means=true --norm-vars=true" \
        --trainer.num-epochs $num_epochs \
        --trainer.optimization.num-jobs-initial $num_jobs_initial \
        --trainer.optimization.num-jobs-final $num_jobs_final \
        --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
        --trainer.optimization.final-effective-lrate $final_effective_lrate \
        --trainer.optimization.minibatch-size $minbatch \
        --trainer.optimization.momentum $momentum \
        --cleanup.remove-egs $remove_egs \
        --cleanup.preserve-model-interval 500 \
        --targets-scp $target_scp \
        --feat-dir $train_dir \
        --use-gpu $use_gpu \
        --egs.opts "$egs_opts" \
        --nj $nj \
        --dir $exp_dir >> train_sp.log 2>&1 || exit 1;
    mv train_sp.log $exp_dir
fi

test_dir=data/dev

if [ $stage -eq 3 ]; then
    echo "$0: estimate wave for $test_dir..."

    ./utils/split_data.sh $test_dir $nj && sep_dir=$test_dir/split$nj

    for n in `seq $nj`; do sed "s:isolated:enhan:g" $sep_dir/$n/wav.scp > $sep_dir/$n/dst.scp; done
    mkdir -p $(cat $sep_dir/?/dst.scp | awk '{print $2}' | awk -F "/[^/]*$" '{print $1}' | sort -u)  

    cmvn_opts=$(cat $exp_dir/cmvn_opts) 

    noisy_feats="ark:copy-feats scp:$sep_dir/JOB/feats.scp ark:- | apply-cmvn $cmvn_opts --utt2spk=ark:$sep_dir/JOB/utt2spk \
        scp:$sep_dir/JOB/cmvn.scp ark:- ark:- |"
    clean_spect="nnet3-compute $exp_dir/test.nnet \"$noisy_feats\" ark:- |"

    # mkdir -p $exp_dir/mask
    $train_cmd JOB=1:$nj $exp_dir/estimate/wav-estimate.JOB.log wav-estimate --config=conf/stft.conf \
        "ark:$clean_spect" scp:$sep_dir/JOB/wav.scp scp:$sep_dir/JOB/dst.scp || exit 1
fi

echo "$0: Done"
