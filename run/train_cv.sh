#!/bin/bash

# 定数定義
DOWNSAMPLE_RATE=2
DURATION=5760
EXP_NAME=exp061
BATCH_SIZE=32

# fold 0から4までループ
for fold in {0..4}
do
  # トレーニングコマンド実行
  python run/train.py \
    downsample_rate=$DOWNSAMPLE_RATE \
    duration=$DURATION \
    exp_name=${EXP_NAME}_fold${fold} \
    batch_size=$BATCH_SIZE \
    split=fold_${fold} \
    seed=464646
done
