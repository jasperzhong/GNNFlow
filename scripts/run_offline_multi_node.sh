#!/bin/bash

MODEL=$1
DATA=$2
CACHE="${3:-LFUCache}"
CACHE_RATIO="${4:-0.2}" # default 20% of cache

HOST_NODE_ADDR=10.28.1.31
HOST_NODE_PORT=11211
NNODES=2
NPROC_PER_NODE=4

export NCCL_SOCKET_IFNAME=enp225s0

cmd="torchrun \
    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=1234 --rdzv_backend=c10d \
    --rdzv_endpoint=$HOST_NODE_ADDR:$HOST_NODE_PORT \
    offline_edge_prediction_multi_node.py --model $MODEL --data $DATA \
    --cache $CACHE --cache-ratio $CACHE_RATIO"

echo $cmd
OMP_NUM_THREADS=8 exec $cmd > $MODEL-$DATA-$CACHE-$CACHE_RATIO-$NNODES-$NPROC_PER_NODE.log 2>&1


