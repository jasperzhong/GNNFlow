#!/bin/bash

MODEL=$1
DATA=$2
CACHE="${3:-LFUCache}"
EDGE_CACHE_RATIO="${4:-0.2}" # default 20% of cache
NODE_CACHE_RATIO="${5:-0.2}" # default 20% of cache
NODE_EMBED_CACHE_RATIO="${6:-0.2}" # default 20% of cache
CACHE_DELAY_EPOCH="${7:-1}"
TIME_WINDOW="${8:-0}" # default 0
NPROC_PER_NODE=${9:-1}

if [[ $NPROC_PER_NODE -gt 1 ]]; then
    cmd="torchrun \
        --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
        --standalone \
        offline_edge_prediction.py --model $MODEL --data $DATA \
        --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO \
        --node-cache-ratio $NODE_CACHE_RATIO --snapshot-time-window $TIME_WINDOW \
        --ingestion-batch-size 10000000 --node-embed-cache-ratio $NODE_EMBED_CACHE_RATIO \
        --cache-delay-epoch $CACHE_DELAY_EPOCH --epoch 10"
else
    cmd="python offline_edge_prediction.py --model $MODEL --data $DATA \
        --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO \
        --node-cache-ratio $NODE_CACHE_RATIO --snapshot-time-window $TIME_WINDOW \
        --ingestion-batch-size 10000000 --node-embed-cache-ratio $NODE_EMBED_CACHE_RATIO \
        --cache-delay-epoch $CACHE_DELAY_EPOCH --epoch 10"
fi

echo $cmd
OMP_NUM_THREADS=8 exec $cmd > ${MODEL}_${DATA}_${CACHE}_${EDGE_CACHE_RATIO}_${NODE_CACHE_RATIO}_${NODE_EMBED_CACHE_RATIO}_${CACHE_DELAY_EPOCH}_${TIME_WINDOW}.log 2>&1
