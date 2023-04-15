import argparse
import time
import os

import numpy as np
from tqdm import tqdm

from gnnflow.config import get_default_config
from gnnflow.utils import build_dynamic_graph, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="REDDIT")
parser.add_argument("--ingestion-batch-size", type=int, default=1000)
parser.add_argument("--adaptive-block-size-strategy",
                    type=str, default="naive")
args = parser.parse_args()

MB = 1 << 20
GB = 1 << 30


def main():
    # Create a dynamic graph
    _, _, _, df = load_dataset(args.dataset)
    # print("Dataset loaded")
    build_start = time.time()
    _, dataset_config = get_default_config("TGN", args.dataset)
    dgraph = build_dynamic_graph(
        **dataset_config,
        device=0,
        adaptive_block_size_strategy=args.adaptive_block_size_strategy)
    # print("Dynamic graph built")

    insertion_times = 0
    for i in range(0, len(df), args.ingestion_batch_size):
        batch = df[i:i + args.ingestion_batch_size]
        src_nodes = batch["src"].values.astype(np.int64)
        dst_nodes = batch["dst"].values.astype(np.int64)
        timestamps = batch["time"].values.astype(np.float32)
        eids = batch["eid"].values.astype(np.int64)
        dgraph.add_edges(src_nodes, dst_nodes, timestamps,
                         eids, add_reverse=False)
        insertion_times += 1
    build_end = time.time()
    build_time = build_end - build_start

    print('build graph time: {:.2f}s, avg_linked_list_length: {:.2f}, graph mem usage: {:.2f}MB, metadata (on GPU) mem usage: {:.2f}MB (adaptive-block-size-strategy: {}) insertion times: {} node size: {}'.format(
        build_time, dgraph.avg_linked_list_length(),
        dgraph.get_graph_memory_usage() / MB,
        dgraph.get_metadata_memory_usage() / MB,
        args.adaptive_block_size_strategy, insertion_times, dgraph.num_vertices()))

    # res = np.array([build_time, dgraph.avg_linked_list_length(),
    #                 dgraph.get_graph_memory_usage() / MB,
    #                 dgraph.get_metadata_memory_usage() / MB])

    nodes = dgraph.nodes()
    out_degree = dgraph.out_degree(nodes)
    num_insertions = dgraph.num_insertions(nodes)
    num_blocks = dgraph.num_blocks(nodes)

    subdir = "tmp_res/adaptive_block_size_insights/"
    os.makedirs(subdir, exist_ok=True)
    np.save(os.path.join(subdir, "out_degree_{}_{}.npy".format(
        args.dataset, args.adaptive_block_size_strategy)), out_degree)
    np.save(os.path.join(subdir, "num_insertions_{}_{}.npy".format(
        args.dataset, args.adaptive_block_size_strategy)), num_insertions)
    np.save(os.path.join(subdir, "num_blocks_{}_{}.npy".format(
        args.dataset, args.adaptive_block_size_strategy)), num_blocks)


if __name__ == "__main__":
    main()
