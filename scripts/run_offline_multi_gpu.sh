#!/bin/bash

NPROC_PER_NODE=2

MODEL=$1
DATA=$2
CACHE="${3:-LFUCache}"
CACHE_RATIO="${4:-0.2}" # default 20% of cache

cmd="torchrun \
    --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
    --standalone \
    offline_edge_prediction_multi_gpu.py --model $MODEL --data $DATA "

if [ -n "$CACHE" ]; then
    cmd="$cmd --cache $CACHE --cache-ratio $CACHE_RATIO"
fi

echo $cmd
OMP_NUM_THREADS=8 exec $cmd


