#!/bin/bash

MODEL=$1
DATA=$2
CACHE="${3:-LFUCache}"
CACHE_RATIO="${4:-0.2}" # default 20% of cache

cmd="python offline_edge_prediction.py --model $MODEL --data $DATA "

if [ -n "$CACHE" ]; then
    cmd="$cmd --cache $CACHE --cache-ratio $CACHE_RATIO"
fi

echo $cmd
exec $cmd


