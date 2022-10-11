#!/bin/bash

MODEL=$1
DATA=$2
CACHE="${3:-LFUCache}"
CACHE_RATIO="${4:-0.2}" # default 20% of cache
NPROC_PER_NODE=${5:-1}

if [[ $NPROC_PER_NODE -gt 1 ]] ; then
    cmd="torchrun \
        --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
        --standalone \
        offline_edge_prediction.py --model $MODEL --data $DATA \
        --cache $CACHE --cache-ratio $CACHE_RATIO"
else
    cmd="python offline_edge_prediction.py --model $MODEL --data $DATA \
        --cache $CACHE --cache-ratio $CACHE_RATIO"
fi

echo $cmd
#OMP_NUM_THREADS=8 exec $cmd > $MODEL-$DATA-$CACHE-$CACHE_RATIO-$NPROC_PER_NODE.log 2>&1


