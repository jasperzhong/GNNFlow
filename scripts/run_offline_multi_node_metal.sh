#!/bin/bash
INTERFACE="ens8"

MODEL=$1
DATA=$2
CACHE="${3:-LFUCache}"
EDGE_CACHE_RATIO="${4:-0.2}" # default 20% of cache
NODE_CACHE_RATIO="${5:-0.2}" # default 20% of cache
PARTITION_STRATEGY="${6:-hash}"
CHUNKSIZE="${7:-100000000}"
DYNAMIC_SCHEDULING="${8:-false}"

HOST_NODE_ADDR=172.31.44.144
HOST_NODE_PORT=29400
NNODES=4
NPROC_PER_NODE=8

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
    offline_edge_prediction_multi_node_kvstore.py --model $MODEL --data $DATA \
    --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO --node-cache-ratio $NODE_CACHE_RATIO\
    --partition --ingestion-batch-size 1000000 \
    --initial-ingestion-batch-size 1000000 \
    --partition-strategy $PARTITION_STRATEGY \
    --num-workers 8 --chunksize $CHUNKSIZE"

if [ $DYNAMIC_SCHEDULING = true ]; then
    cmd="$cmd --dynamic-scheduling"
fi

rm -rf /dev/shm/rmm_pool_*

echo $cmd
LOGLEVEL=INFO OMP_NUM_THREADS=8 exec $cmd

