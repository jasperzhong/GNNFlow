#!/bin/bash

MODEL=$1

# TGN
if [ $MODEL == "tgn" ] || [ $MODEL == "TGN" ];then
    cmd="python train.py --model TGN"
fi

# TGAT
if [ $MODEL == "tgat" ] || [ $MODEL == "TGAT" ];then
    cmd="python train.py --model TGAT --sample-layer 2 --sample-neighbor 10 10 \
                    --sample-strategy uniform"
fi

# JODIE
if [ $MODEL == "jodie" ] || [ $MODEL == "JODIE" ];then
    cmd="python train.py --model JODIE --no-sample"
fi

# APAN
if [ $MODEL == "apan" ] || [ $MODEL == "APAN" ];then
    cmd="python train.py --model APAN --no-neg --deliver-to-neighbors"
fi

# DySAT
if [ $MODEL == "dysat" ] || [ $MODEL == "DySAT" ] || [ $MODEL == "DYSAT" ];then
    cmd="python train.py --model DySAT --epoch 50 \
                    --sample-layer 2 --sample-neighbor 10 10 \
                    --sample-strategy uniform --sample-history 100 \
                    --sample-duration 10000 \
                    --prop-time"
fi

echo $cmd
exec $cmd


