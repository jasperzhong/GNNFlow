#!/bin/bash

MODEL=$1

# TGN
if [ "$MODEL" == "tgn" ];then
    cmd="python train.py --model tgn"
fi

# TGAT
if ["$MODEL" == "tgat" ];then
    cmd="python train.py --model tgat --sample-layer 2 --sample-neighbor [10, 10] \
                    --sample-strategy uniform"
fi

# JODIE
if ["$MODEL" == "jodie"];then
    cmd="python train.py --model jodie --no-sample"
fi

# APAN
if ["$MODEL" == "apan"];then
    cmd="python train.py --model apan --no-neg --deliver-to-neighbors"
fi

# DySAT
if ["$MODEL" == "dysat"];then
    cmd="python train.py --model dysat --epoch 50 \
                    --sample-layer 2 --sample-neighbor [10, 10] \
                    --sample-strategy uniform --sample-history 100 \
                    --sample-duration 10000 \
                    --prop-time"
fi

echo $cmd


