# GNNFlow

A comprehensive framework for training graph neural networks on dynamic graphs.

NB: this is an ongoing work.

## Install

Our development environment:
- Ubuntu 20.04LTS
- g++ 9.4
- CUDA 11.3 / 11.6
- cmake 3.23

Dependencies:
- torch >= 1.10
- dgl (CUDA version) 

Compile and install: 
```sh
python setup.py install
```

For debug mode,
```sh
DEBUG=1 pip install -v -e .
```

## Prepare data

```sh
cd scripts/ && ./download_data.sh
```

## Train

**Multi-GPU single machine**

Training [TGN](https://arxiv.org/pdf/2006.10637v2.pdf) model on the REDDIT dataset with LRU feature cache (cache ratio=0.2) on four GPUs.
```sh
./scripts/run_offline.sh TGN REDDIT LRUCache 0.2 4
```

**Distributed training**

Training TGN model on the REDDIT dataset with LRU feature cache (cache ratio=0.2) and hash-based graph partitioning strategy.
```sh
./scripts/run_offline.sh TGN REDDIT LRUCache 0.2 hash 
```

