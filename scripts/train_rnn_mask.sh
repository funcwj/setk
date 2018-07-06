#!/bin/bash
# wujian@2018

set -ue

. ./path.sh || exit 1

train_dir=data/mask_train_simu

train_stage=-6
train_cmd="run.pl"

initial_effective_lrate=0.0001
final_effective_lrate=0.00005
num_epochs=40
num_jobs_initial=1
num_jobs_final=1
remove_egs=true
use_gpu=true

minbatch=256
momentum=0.8
# samples_per_iter=80000

stage=2
egs_nj=16
egs_dev_subset=500
egs_opts="--nj $egs_nj --num-utts-subset $egs_dev_subset"
cmvn_opts="--norm-vars=true --norm-means=true"
# default 400000
samples_per_iter=50000
preserve_model_interval=10

chunk_width=150
chunk_left_context=40
chunk_right_context=40

nj=20
stft_dim=513

label_delay=5

mdl=lstm
exp_dir=exp/mask/$mdl

. parse_options.sh || exit 1

# it's a RNN model
if [ $stage -eq 1 ]; then

  mkdir -p $exp_dir/configs
  input_dim=$(feat-to-dim scp:$train_dir/feats.scp -)
  opts="dropout-proportion=0.2 decay-time=20 l2-regularize=0.0"

  case $mdl in
    blstmp )
    echo "$0: config fast-bi-lstmp-batchnorm networks..."
    cat <<EOF > $exp_dir/configs/network.xconfig
input dim=$input_dim name=input
fast-lstmp-batchnorm-layer name=f-blstm1 cell-dim=512 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $opts
fast-lstmp-batchnorm-layer name=b-blstm1 cell-dim=512 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=3 $opts
fast-lstmp-batchnorm-layer name=f-blstm2 input=Append(f-blstm1, b-blstm1) cell-dim=512 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $opts
fast-lstmp-batchnorm-layer name=b-blstm2 input=Append(f-blstm1, b-blstm1) cell-dim=512 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=3 $opts
fast-lstmp-batchnorm-layer name=f-blstm3 input=Append(f-blstm2, b-blstm2) cell-dim=512 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $opts
fast-lstmp-batchnorm-layer name=b-blstm3 input=Append(f-blstm2, b-blstm2) cell-dim=512 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=3 $opts
output-layer name=output input=Append(f-blstm3, b-blstm3) output-delay=$0 dim=513 include-activation=sigmoid include-log-softmax=false objective-type=quadratic
EOF
      ;;

    lstm )
      echo "$0: config fast-lstm-batchnorm networks..."
      cat <<EOF > $exp_dir/configs/network.xconfig
input dim=$input_dim name=input
fast-lstm-batchnorm-layer name=lstm1 cell-dim=512 
fast-lstm-batchnorm-layer name=lstm2 cell-dim=512 
fast-lstm-batchnorm-layer name=lstm3 cell-dim=512 
fast-lstm-batchnorm-layer name=lstm4 cell-dim=512 
output-layer name=output input=lstm4 output-delay=$label_delay dim=$stft_dim include-activation=sigmoid include-log-softmax=false objective-type=quadratic
EOF
      ;;
    lstmp )
      echo "$0: config fast-uni-lstmp-batchnorm networks..."
      cat <<EOF > $exp_dir/configs/network.xconfig
input dim=$input_dim name=input
fast-lstmp-batchnorm-layer name=lstm1 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256
fast-lstmp-batchnorm-layer name=lstm2 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256
fast-lstmp-batchnorm-layer name=lstm3 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256
fast-lstmp-batchnorm-layer name=lstm4 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256
output-layer name=output input=lstm4 output-delay=$label_delay dim=$stft_dim include-activation=sigmoid include-log-softmax=false objective-type=quadratic
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
    echo "$0: training mask prediction networks..."
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
        --trainer.samples-per-iter $samples_per_iter \
        --cleanup.remove-egs $remove_egs \
        --cleanup.preserve-model-interval $preserve_model_interval \
        --targets-scp $train_dir/masks.scp \
        --egs.chunk-width $chunk_width \
        --egs.chunk-left-context $chunk_left_context \
        --egs.chunk-right-context $chunk_right_context \
        --feat-dir $train_dir \
        --use-gpu $use_gpu \
        --egs.opts "$egs_opts" \
        --nj $nj \
        --dir $exp_dir >> $exp_dir/train_rnn.log 2>&1 || exit 1;
fi

echo "$0: Done"
