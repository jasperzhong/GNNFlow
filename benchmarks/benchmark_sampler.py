import argparse

import numpy as np
import random
import torch
from tqdm import tqdm

from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import build_dynamic_graph, load_dataset
from gnnflow.config import get_default_config

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="REDDIT")
parser.add_argument("--model", type=str)
parser.add_argument("--batch_size", type=int, default=600)
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
    _, dataset_config = get_default_config(args.model, args.dataset)
    dataset_config["mem_resource_type"] = args.mem_resource_type
    dgraph = build_dynamic_graph(
        **dataset_config, dataset_df=df)

    # Create a temporal sampler
    if args.model == "tgn":
        sampler = TemporalSampler(
            dgraph, fanouts=[10], strategy="recent")
    elif args.model == "tgat":
        sampler = TemporalSampler(
            dgraph, fanouts=[10, 10], strategy="uniform", seed=args.seed)
    elif args.model == "dysat":
        sampler = TemporalSampler(
            dgraph, fanouts=[10, 10], num_snapshots=3,
            snapshot_time_window=10000, prop_time=True,
            strategy="uniform", seed=args.seed)
    else:
        raise ValueError("Unknown model: {}".format(args.model))

    neg_link_sampler = NegLinkSampler(dgraph.num_vertices())

    total_sampled_nodes = 0
    for _, rows in tqdm(df.groupby(df.index // args.batch_size)):
        # Sample a batch of data
        root_nodes = np.concatenate(
            [rows.src.values, rows.dst.values,
                neg_link_sampler.sample(len(rows))]).astype(np.int64)
        ts = np.concatenate(
            [rows.time.values, rows.time.values, rows.time.values]).astype(
            np.float32)

        if args.stat:
            blocks = sampler.sample(root_nodes, ts)
            for block in blocks:
                for b in block:
                    total_sampled_nodes += b.num_src_nodes() - b.num_dst_nodes()
        else:
            sampler._sampler.sample(root_nodes, ts)

    if args.stat:
        print("Total sampled nodes: {}".format(total_sampled_nodes))


if __name__ == "__main__":
    main()
