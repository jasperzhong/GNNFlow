#!/bin/bash

MODEL=$1
DATA=$2
CACHE="${3:-LFUCache}"
CACHE_RATIO="${4:-0.2}" # default 20% of cache

HOST_NODE_ADDR=10.28.1.30
HOST_NODE_PORT=29400
NNODES=2
NPROC_PER_NODE=4

CURRENT_NODE_IP=$(hostname -I | awk '{print $1}')
if [ $CURRENT_NODE_IP = $HOST_NODE_ADDR ]; then
    IS_HOST=true
else
    IS_HOST=false
fi

export NCCL_SOCKET_IFNAME=enp225s0

cmd="torchrun \
    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=1234 --rdzv_backend=c10d \
    --rdzv_endpoint=$HOST_NODE_ADDR:$HOST_NODE_PORT \
    --rdzv_conf is_host=$IS_HOST \
    offline_edge_prediction_multi_node.py --model $MODEL --data $DATA \
    --cache $CACHE --cache-ratio $CACHE_RATIO"

echo $cmd
LOGLEVEL=INFO OMP_NUM_THREADS=8 exec $cmd 


