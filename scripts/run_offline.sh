#!/bin/bash

MODEL=$1
DATA=$2
CACHE="${3:-LFUCache}"
EDGE_CACHE_RATIO="${4:-0.2}" # default 20% of cache
NODE_CACHE_RATIO="${5:-0.2}" # default 20% of cache
NPROC_PER_NODE=${6:-1}

if [[ $NPROC_PER_NODE -gt 1 ]] ; then
    cmd="torchrun \
        --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
        --standalone \
        offline_edge_prediction.py --model $MODEL --data $DATA \
        --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO --node-cache-ratio $NODE_CACHE_RATIO"
else
    cmd="python offline_edge_prediction.py --model $MODEL --data $DATA \
        --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO --node-cache-ratio $NODE_CACHE_RATIO"
fi

rm -rf /dev/shm/*
echo $cmd
OMP_NUM_THREADS=8 exec $cmd > $MODEL-$DATA-$CACHE-$EDGE_CACHE_RATIO-$NODE_CACHE_RATIO-$NPROC_PER_NODE.log 2>&1


