#!/bin/bash

models=("TGN" "TGAT" "DySAT")

for model in "${models[@]}"; do
    nsys profile -s cpu -o nsight_report_${model} -f true --cudabacktrace=true python benchmark_sampler.py --model ${model}
done
