#!/bin/bash
INTERFACE="eth2"

MODEL=$1
DATA=$2
CACHE="${3:-LFUCache}"
CACHE_RATIO="${4:-0.2}" # default 20% of cache
PARTITION_STRATEGY="${5:-hash}"

HOST_NODE_ADDR=10.28.1.16
HOST_NODE_PORT=29400
NNODES=2
NPROC_PER_NODE=2

CURRENT_NODE_IP=$(ip -4 a show dev ${INTERFACE} | grep inet | cut -d " " -f6 | cut -d "/" -f1)
if [ $CURRENT_NODE_IP = $HOST_NODE_ADDR ]; then
    IS_HOST=true
else
    IS_HOST=false
fi

export NCCL_SOCKET_IFNAME=${INTERFACE}
export GLOO_SOCKET_IFNAME=${INTERFACE}
export TP_SOCKET_IFNAME=${INTERFACE}

cmd="torchrun \
    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=1234 --rdzv_backend=c10d \
    --rdzv_endpoint=$HOST_NODE_ADDR:$HOST_NODE_PORT \
    --rdzv_conf is_host=$IS_HOST \
    offline_edge_prediction_multi_node.py --model $MODEL --data $DATA \
    --cache $CACHE --cache-ratio $CACHE_RATIO \
    --partition --ingestion-batch-size 100000 \
    --partition-strategy $PARTITION_STRATEGY"

echo $cmd
NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO CUDA_LAUNCH_BLOCKING=1 LOGLEVEL=DEBUG OMP_NUM_THREADS=8 exec $cmd


