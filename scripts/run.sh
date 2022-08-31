#!/bin/bash

MODEL=$1
STRATEGY=$2
DATA=$3
CACHE=$4

# TGN
if [ $MODEL == "tgn" ] || [ $MODEL == "TGN" ];then
    cmd="python train.py --dataset $DATASET --model TGN"
fi

# TGAT
if [ $MODEL == "tgat" ] || [ $MODEL == "TGAT" ];then
    cmd="python train.py --model TGAT --dropout 0.1 --attn-dropout 0.1 \
                    --sample-layer 2 --sample-neighbor 10 10"
fi

# DySAT
if [ $MODEL == "dysat" ] || [ $MODEL == "DySAT" ] || [ $MODEL == "DYSAT" ];then
    cmd="python train.py --dataset $DATASET --model DySAT --epoch 50 \
        --dropout 0.1 --attn-dropout 0.1 \
        --sample-layer 2 --sample-neighbor 10 10 \
        --sample-strategy uniform --sample-history 3 \
        --sample-duration 10000 \
        --prop-time"
fi

if [ -n "$DATA" ]; then
    cmd="$cmd --data $DATA"
fi

if [ -n "$STRATEGY" ]; then
    cmd="$cmd --sample-strategy $STRATEGY"
fi

if [ -n "$CACHE" ]; then
    cmd="$cmd --cache $CACHE"
fi

echo $cmd
exec $cmd


