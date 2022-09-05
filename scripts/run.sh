#!/bin/bash

MODEL=$1
DATA=$2
CACHE=$3

cmd="python offline_edge_prediction.py --model $MODEL --data $DATA"

if [ -n "$CACHE" ]; then
    cmd="$cmd --cache $CACHE"
fi

echo $cmd
exec $cmd


