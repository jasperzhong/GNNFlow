#!/bin/bash

MODEL=$1
DATA=$2
CACHE="${3:-LFUCache}"
EDGE_CACHE_RATIO="${4:-0.2}" # default 20% of cache
NODE_CACHE_RATIO="${5:-0.2}" # default 20% of cache
TIME_WINDOW="${6:-0}" # default 0
NPROC_PER_NODE=${7:-1}

if [[ $NPROC_PER_NODE -gt 1 ]]; then
    cmd="torchrun \
        --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
        --standalone \
        offline_edge_prediction.py --model $MODEL --data $DATA \
        --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO \
        --node-cache-ratio $NODE_CACHE_RATIO --snapshot-time-window $TIME_WINDOW \
        --ingestion-batch-size 10000000"
else
    cmd="python offline_edge_prediction.py --model $MODEL --data $DATA \
        --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO \
        --node-cache-ratio $NODE_CACHE_RATIO --snapshot-time-window $TIME_WINDOW \
        --ingestion-batch-size 10000000"
fi

echo $cmd
OMP_NUM_THREADS=8 exec $cmd > ${MODEL}_${DATA}_${CACHE}_${EDGE_CACHE_RATIO}_${NODE_CACHE_RATIO}_${TIME_WINDOW}_baseline_memory_sync_4000_new.log 2>&1
