import argparse
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from gnnflow.config import get_default_config
from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import build_dynamic_graph, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="REDDIT")
parser.add_argument("--model", type=str)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stat", action="store_true", help="print statistics")
parser.add_argument("--mem-resource-type", type=str,
                    choices=["cuda", "unified", "pinned"],
                    default="cuda", help="memory resource type")

args = parser.parse_args()

MB = 1 << 20
GB = 1 << 30


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

args.model = args.model.lower()


class NegLinkSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(0, self.num_nodes, (n, ))


def main():
    # Create a dynamic graph
    _, _, _, df = load_dataset(args.dataset)
    model_config, dataset_config = get_default_config(args.model, args.dataset)
    dataset_config["mem_resource_type"] = args.mem_resource_type
    dgraph = build_dynamic_graph(
        **dataset_config, dataset_df=df)
    sampler = TemporalSampler(dgraph, **model_config)
    batch_size = model_config["batch_size"]

    neg_link_sampler = NegLinkSampler(dgraph.num_vertices())

    total_target_nodes = 0
    total_sampled_nodes = 0
    start = time.time()
    for _, rows in tqdm(df.groupby(df.index // batch_size)):
        # Sample a batch of data
        root_nodes = np.concatenate(
            [rows.src.values, rows.dst.values,
                neg_link_sampler.sample(len(rows))]).astype(np.int64)
        ts = np.concatenate(
            [rows.time.values, rows.time.values, rows.time.values]).astype(
            np.float32)
        total_target_nodes += len(root_nodes)

        if args.stat:
            blocks = sampler.sample(root_nodes, ts)
            for block in blocks:
                for b in block:
                    total_sampled_nodes += b.num_src_nodes() - b.num_dst_nodes()
        else:
            sampler._sampler.sample(root_nodes, ts)
    end = time.time()
    print("Sampling throughput (samples/sec): {:.2f}, elapased time (sec): {:.2f}".format(
        total_target_nodes / (end - start), end-start))

    if args.stat:
        print("Total sampled nodes: {}".format(total_sampled_nodes))


if __name__ == "__main__":
    main()
