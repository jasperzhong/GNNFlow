#!/bin/bash

MODEL=$1

# TGN
if [[ $MODEL -eq 'tgn' ]];then
    cmd="python train.py --model tgn"
fi

# TGAT
if [[ MODEL -eq 'tgat' -o MODEL -eq 'TGAT']];then
    cmd="python train.py --model tgat --sample-layer 2 --sample-neighbor [10, 10] \
                    --sample-strategy uniform"
fi

# JODIE
if [[ MODEL -eq 'jodie' -o MODEL -eq 'JODIE']];then
    cmd="python train.py --model jodie --no-sample"
fi

# APAN
if [[ MODEL -eq 'apan' -o MODEL -eq 'APAN']];then
    cmd="python train.py --model apan --no-neg --deliver-to-neighbors"
fi

# DySAT
if [[ MODEL -eq 'dysat' -o MODEL -eq 'DySAT'] -o [ MODEL -eq 'DYSAT']];then
    cmd="python train.py --model dysat --epoch 50 \
                    --sample-layer 2 --sample-neighbor [10, 10] \
                    --sample-strategy uniform --sample-history 100 \
                    --sample-duration 10000 \
                    --prop-time"
fi

echo $cmd


