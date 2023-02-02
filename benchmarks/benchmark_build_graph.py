import argparse
import time

import numpy as np
from tqdm import tqdm

from gnnflow.config import get_default_config
from gnnflow.utils import build_dynamic_graph, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="REDDIT")
parser.add_argument("--ingestion-batch-size", type=int, default=100000)
parser.add_argument("--disable-adaptive-block-size", action="store_true")
parser.add_argument("--mem-resource-type", type=str,
                    choices=["cuda", "unified", "pinned"],
                    default="cuda", help="memory resource type")
args = parser.parse_args()

MB = 1 << 20
GB = 1 << 30


def main():
    # Create a dynamic graph
    _, _, _, df = load_dataset(args.dataset)
    print("Dataset loaded")
    build_start = time.time()
    _, dataset_config = get_default_config("TGN", args.dataset)
    dataset_config["mem_resource_type"] = args.mem_resource_type
    dgraph = build_dynamic_graph(
        **dataset_config,
        adaptive_block_size=not args.disable_adaptive_block_size)
    print("Dynamic graph built")
    
    for i in tqdm(range(0, len(df), args.ingestion_batch_size)):
        batch = df[i:i + args.ingestion_batch_size]
        src_nodes = batch["src"].values.astype(np.int64)
        dst_nodes = batch["dst"].values.astype(np.int64)
        timestamps = batch["time"].values.astype(np.float32)
        eids = batch["eid"].values.astype(np.int64)
        dgraph.add_edges(src_nodes, dst_nodes, timestamps,
                         eids, add_reverse=False)
    build_end = time.time()
    build_time = build_end - build_start

    print('build graph time: {:.2f}s, avg_linked_list_length: {:.2f}, graph mem usage: {:.2f}MB, metadata (on GPU) mem usage: {:.2f}MB (adaptive-block-size: {}, mem-resource-type: {})'.format(
        build_time, dgraph.avg_linked_list_length(),
        dgraph.get_graph_memory_usage() / MB,
        dgraph.get_metadata_memory_usage() / MB,
        not args.disable_adaptive_block_size, args.mem_resource_type))


if __name__ == "__main__":
    main()
