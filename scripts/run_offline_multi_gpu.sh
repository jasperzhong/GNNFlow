#!/bin/bash


MODEL=$1
DATA=$2
NPROC_PER_NODE=${3:-1}

cmd="torchrun \
    --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
    --standalone \
    offline_edge_prediction_multi_gpu.py --model $MODEL --data $DATA "

echo $cmd
OMP_NUM_THREADS=8 exec $cmd


