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
NPROC_PER_NODE=1

CURRENT_NODE_IP=$(ip -4 a show dev ${INTERFACE} | grep inet | cut -d " " -f6 | cut -d "/" -f1)
if [ $CURRENT_NODE_IP = $HOST_NODE_ADDR ]; then
    IS_HOST=true
else
    IS_HOST=false
fi

export NCCL_SOCKET_IFNAME=${INTERFACE}
export GLOO_SOCKET_IFNAME=${INTERFACE}
export TP_SOCKET_IFNAME=${INTERFACE}

cmd="LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 RANK=0 WORLD_SIZE=2 MASTER_ADDR=$HOST_NODE_ADDR MASTER_PORT=$HOST_NODE_PORT python \
    offline_edge_prediction_multi_node.py --model $MODEL --data $DATA \
    --cache $CACHE --cache-ratio $CACHE_RATIO \
    --partition --ingestion-batch-size 100000 \
    --partition-strategy $PARTITION_STRATEGY"

echo $cmd
TP_VERBOSE_LOGGING=9 LOGLEVEL=DEBUG OMP_NUM_THREADS=8 exec $cmd
