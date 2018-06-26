#!/usr/bin/env bash
# wujian@2018.6.26

# config same as example_{1..4}.m in https://github.com/ehabets/RIR-Generator

./bin/rir-simulate --report --sound-velocity=340 \
  --samp-frequency=16000 --receiver-location=2,1.5,2 \
  --source-location=2,3.5,2 --room-topo=5,4,6 \
  --beta=0.4 --number-samples=4096 rir1.wav

./bin/rir-simulate --report --sound-velocity=340 \
  --samp-frequency=16000 --receiver-location=2,1.5,2 \
  --source-location=2,3.5,2 --room-topo=5,4,6 \
  --beta=0.4 --number-samples=2048 --order=2 \
  --microphone-type=omnidirectional \
  --angle=0 --hp-filter=true rir2.wav

./bin/rir-simulate --report --sound-velocity=340 \
  --samp-frequency=16000 --receiver-location="2,1.5,2;1,1.5,2" \
  --source-location=2,3.5,2 --room-topo=5,4,6 \
  --beta=0.4 --number-samples=4096 --order=-1 \
  --microphone-type=omnidirectional \
  --angle=0 --hp-filter=true rir3.wav

./bin/rir-simulate --report --sound-velocity=340 \
  --samp-frequency=16000 --receiver-location=2,1.5,2 \
  --source-location=2,3.5,2 --room-topo=5,4,6 \
  --beta=0.4 --number-samples=4096 --order=-1 \
  --microphone-type=hypercardioid \
  --hp-filter=false --angle=1.57,0 rir4.wav
