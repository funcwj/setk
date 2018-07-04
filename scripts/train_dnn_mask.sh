#!/bin/bash
# wujian@2018

set -ue

. ./path.sh || exit 1

train_dir=data/train_simu

train_stage=-6
train_cmd="run.pl"

initial_effective_lrate=0.0015
final_effective_lrate=0.00005
num_epochs=6
num_jobs_initial=1
num_jobs_final=1
remove_egs=true
use_gpu=true

minbatch=512
frames_per_eg=150
momentum=0.8
cmvn_opts="--norm-means=true --norm-vars=true"
# default 400000
samples_per_iter=400000
# default 1
backstitch_training_interval=1
# default 0.0
backstitch_training_scale=0.0

stage=1
stft_dim=513
egs_nj=16
egs_dev_subset=300
egs_opts="--nj $egs_nj --num-utts-subset $egs_dev_subset"
egs_dir=
nj=40

rewrite=false
mdl=dnn
exp_dir=exp/mask/$mdl

. parse_options.sh || exit 1

# generate same feed forward network configs
if [ $stage -eq 1 ]; then

    [[ -d $exp_dir && ! $rewrite ]] && echo "$0: $exp_dir already exists" && exit 1
    [[ -d $exp_dir && $rewrite ]] && echo "$0: clear $exp_dir" && rm -rf $exp_dir
    
    mkdir -p $exp_dir/configs
    input_dim=$(feat-to-dim scp:$train_dir/feats.scp -)

    case $mdl in
        dnn)
            echo "$0: config relu-batchnorm dnn networks..."
            cat <<EOF > $exp_dir/configs/network.xconfig
input dim=$input_dim name=input
relu-batchnorm-layer name=dnn1 dim=2048 input=Append(-2,-1,0,1,2)
relu-batchnorm-layer name=dnn2 dim=2048
relu-batchnorm-layer name=dnn3 dim=2048
relu-batchnorm-layer name=dnn4 dim=2048
# a version of modified output-layer config in steps/libs/nnet3/xconfig/basic_layers.py
# add include-sigmoid & include-relu same as include-log-softmax
output-layer name=output input=dnn4 dim=$stft_dim include-log-softmax=false include-sigmoid=true objective-type=quadratic
EOF
        ;;
        tdnn)
            echo "$0: config relu-renorm tdnn networks..."
            cat <<EOF > $exp_dir/configs/network.xconfig
input dim=$input_dim name=input
relu-renorm-layer name=tdnn1 dim=512 input=Append(-2,-1,0,1,2)
relu-renorm-layer name=tdnn2 dim=512 input=Append(-1,0,1)
relu-renorm-layer name=tdnn3 dim=512 input=Append(-1,0,1)
relu-renorm-layer name=tdnn4 dim=512 input=Append(-3,0,3)
relu-renorm-layer name=tdnn5 dim=512 input=Append(-3,0,3)
relu-renorm-layer name=tdnn6 dim=512 input=Append(-6,-3,0)
# a version of modified output-layer config in steps/libs/nnet3/xconfig/basic_layers.py
output-layer name=output input=tdnn6 dim=$stft_dim include-log-softmax=false include-sigmoid=true objective-type=quadratic
EOF
        ;;
        cnn)
            echo "$0: config relu-renorm cnn-tdnn networks..."
            cat <<EOF > $exp_dir/configs/network.xconfig
input dim=$input_dim name=input
conv-relu-batchnorm-layer name=cnn1 height-in=$input_dim height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=32
conv-relu-batchnorm-layer name=cnn2 height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=32
conv-relu-batchnorm-layer name=cnn3 height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
conv-relu-batchnorm-layer name=cnn4 height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
conv-relu-batchnorm-layer name=cnn5 height-in=20 height-out=20 time-offsets=-3,0,3 height-offsets=-1,0,1 num-filters-out=32
relu-renorm-layer name=tdnn6 dim=512 input=Append(-3,0,3)
relu-renorm-layer name=tdnn7 dim=512 input=Append(-6,-3,0)
# a version of modified output-layer config in steps/libs/nnet3/xconfig/basic_layers.py
output-layer name=output input=tdnn7 dim=$stft_dim include-log-softmax=false include-sigmoid=true objective-type=quadratic
EOF
        ;;
        *)
            echo "Unknown model type $mdl"
            exit 1
        ;;
    esac
    steps/nnet3/xconfig_to_configs.py --xconfig-file $exp_dir/configs/network.xconfig --config-dir $exp_dir/configs/
fi


if [ $stage -eq 2 ]; then
    echo "$0: training mask prediction networks on $train_dir ..."
    steps/nnet3/train_raw_dnn.py --stage=$train_stage \
        --cmd=$train_cmd \
        --feat.cmvn-opts="$cmvn_opts" \
        --trainer.num-epochs $num_epochs \
        --trainer.optimization.num-jobs-initial $num_jobs_initial \
        --trainer.optimization.num-jobs-final $num_jobs_final \
        --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
        --trainer.optimization.final-effective-lrate $final_effective_lrate \
        --trainer.optimization.minibatch-size $minbatch \
        --trainer.optimization.momentum $momentum \
        --trainer.optimization.backstitch-training-interval $backstitch_training_interval \
        --trainer.optimization.backstitch-training-scale $backstitch_training_scale \
        --trainer.samples-per-iter $samples_per_iter \
        --cleanup.remove-egs $remove_egs \
        --targets-scp $train_dir/masks.scp \
        --feat-dir $train_dir \
        --use-gpu $use_gpu \
        --egs.opts "$egs_opts" \
        --egs.dir "$egs_dir" \
        --egs.frames-per-eg $frames_per_eg \
        --nj $nj \
        --dir $exp_dir >> $exp_dir/train_${mdl}.log 2>&1 || exit 1;
fi

echo "$0: Done"
