import argparse
import time

import numpy as np
# pip install nvidia-ml-py3
import nvidia_smi

from gnnflow.config import get_default_config
from gnnflow.utils import build_dynamic_graph, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="REDDIT")
parser.add_argument("--ingestion-batch-size", type=int, default=100000)
parser.add_argument("--mem-resource-type", type=str,
                    choices=["cuda", "unified", "pinned"],
                    default="cuda", help="memory resource type")
args = parser.parse_args()

MB = 1 << 20
GB = 1 << 30


def main():
    nvidia_smi.nvmlInit()
    # Create a dynamic graph
    _, _, _, df = load_dataset(args.dataset)
    build_start = time.time()
    _, dataset_config = get_default_config("TGN", args.dataset)
    dataset_config["mem_resource_type"] = args.mem_resource_type
    dgraph = build_dynamic_graph(
        **dataset_config)
    for i in range(0, len(df), args.ingestion_batch_size):
        batch = df[i:i + args.ingestion_batch_size]
        src_nodes = batch["src"].values.astype(np.int64)
        dst_nodes = batch["dst"].values.astype(np.int64)
        timestamps = batch["time"].values.astype(np.float32)
        eids = batch["eid"].values.astype(np.int64)
        dgraph.add_edges(src_nodes, dst_nodes, timestamps,
                         eids, add_reverse=True)
    build_end = time.time()
    build_time = build_end - build_start

    gpu_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(
        nvidia_smi.nvmlDeviceGetHandleByIndex(0))
    print('build graph time: {:.4f}s with mem_resource_type={}, '
          'gpu memory usage: {:.2f}GB'.format(
              build_time, args.mem_resource_type, gpu_mem.used / GB))
    nvidia_smi.nvmlShutdown()


if __name__ == "__main__":
    main()
