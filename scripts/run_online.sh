#!/bin/bash

MODEL=$1
DATA=$2
CACHE="${3:-LFUCache}"
EDGE_CACHE_RATIO="${4:-0.2}" # default 20% of cache
NODE_CACHE_RATIO="${5:-0.2}" # default 20% of cache
NPROC_PER_NODE=${6:-1}
REPLAY="${7:-0}"
PHASE1_RATIO="${8:-0.3}" # default 50% of replay
TIME_WINDOW="${9:-0}"

if [[ $NPROC_PER_NODE -gt 1 ]] ; then
    cmd="torchrun \
        --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
        --standalone \
        online_edge_prediction.py --model $MODEL --data $DATA \
        --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO --node-cache-ratio $NODE_CACHE_RATIO
        --replay-ratio $REPLAY --phase1-ratio $PHASE1_RATIO --snapshot-time-window $TIME_WINDOW"
else
    cmd="python online_edge_prediction.py --model $MODEL --data $DATA \
        --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO --node-cache-ratio $NODE_CACHE_RATIO
        --replay-ratio $REPLAY --phase1-ratio $PHASE1_RATIO --snapshot-time-window $TIME_WINDOW"
fi

rm -rf /dev/shm/rmm_*
echo $cmd
OMP_NUM_THREADS=8 exec $cmd > no-retrain-30%online-$MODEL-$DATA-$CACHE-$EDGE_CACHE_RATIO-$NODE_CACHE_RATIO-$NPROC_PER_NODE-$REPLAY-$PHASE1_RATIO-$TIME_WINDOW.log 2>&1
