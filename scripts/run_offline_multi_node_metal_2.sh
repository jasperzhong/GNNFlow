#!/bin/bash
INTERFACE="ens8"

MODEL=$1
DATA=$2
CACHE="${3:-LFUCache}"
EDGE_CACHE_RATIO="${4:-0.2}" # default 20% of cache
NODE_CACHE_RATIO="${5:-0.2}" # default 20% of cache

HOST_NODE_ADDR=172.31.27.162
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
    offline_edge_prediction.py --model $MODEL --data $DATA \
    --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO --node-cache-ratio $NODE_CACHE_RATIO\
    --ingestion-batch-size 10000000"

rm -rf /dev/shm/rmm_pool_*
rm -rf /dev/shm/edge_feats
rm -rf /dev/shm/node_feats

echo $cmd
OMP_NUM_THREADS=8 exec $cmd > ${MODEL}_${DATA}_${CACHE}_${EDGE_CACHE_RATIO}_${NODE_CACHE_RATIO}_${NNODES}_${NPROC_PER_NODE}.log 2>&1

