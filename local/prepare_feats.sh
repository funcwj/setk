#!/usr/bin/env bash
# wujian@2018

set -eu
nj=40
extractor_dir=
feature=fbank

. ./utils/parse_options.sh 

[ $# -ne 1 ] && echo "format error: $0 <data-dir>" && exit 1

data_dir=$(cd $1; pwd)
name=$(basename $data_dir)

case $feature in
  fbank )
    ./steps/make_fbank.sh \
      --fbank-config ./conf/fbank.conf.enh \
      --nj $nj \
      $data_dir exp/make_fbank fbank/$name
    ;;
  spectrogram )
    ./scripts/compute_stft_stats.sh \
    --stft-config ./conf/stft.conf \
    --nj $nj \
    $data_dir exp/make_stft stft/$name
    ;;
  * )
    echo "$0: Unknown feature type: $feature" && exit 1
    ;;
esac

if [ ! -z $extractor_dir ]; then
  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.enh.conf $data_dir ./exp/make_mfcc mfcc/$name
  steps/online/nnet2/extract_ivectors_online.sh --nj $nj $data_dir $extractor_dir ivectors/$name
fi


steps/compute_cmvn_stats.sh $data_dir exp/make_fbank fbank/$name
