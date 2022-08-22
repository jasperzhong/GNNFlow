import argparse

import time

from dgnn.utils import build_dynamic_graph, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="REDDIT")
parser.add_argument("--mem-resource-type", type=str,
                    choices=["cuda", "unified", "pinned"],
                    default="cuda", help="memory resource type")
args = parser.parse_args()

MB = 1 << 20
GB = 1 << 30

default_config = {
    "initial_pool_size": 20 * MB,
    "maximum_pool_size": 50 * MB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 64,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
}


def main():
    # Create a dynamic graph
    _, _, _, df = load_dataset(args.dataset)
    build_start = time.time()
    config = default_config.copy()
    config["mem_resource_type"] = args.mem_resource_type
    dgraph = build_dynamic_graph(df, **config)
    build_end = time.time()
    build_time = build_end - build_start
    print('build graph time: {:.4f}s with mem_resource_type={}'.format(
        build_time, args.mem_resource_type))


if __name__ == "__main__":
    main()
