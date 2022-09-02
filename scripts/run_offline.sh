#!/bin/bash
MODEL=$1
RETRAIN=$2
DATA=$3

# TGAT
if [ $MODEL == "tgat" ] || [ $MODEL == "TGAT" ];then
    cmd="python offline.py --model TGAT --dropout 0.1 --attn-dropout 0.1 \
                    --sample-layer 2 --sample-neighbor 10 10 \
                    --data $DATA --sample-strategy uniform"
fi

# TGN
if [ $MODEL == "tgn" ] || [ $MODEL == "TGN" ];then
    cmd="python offline.py --dataset REDDIT --model TGN"
fi

if [ -n "$RETRAIN" ]; then
    cmd="$cmd --retrain $RETRAIN"
fi


echo $cmd
exec $cmd
