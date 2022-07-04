import argparse

import numpy as np
from tqdm import tqdm

from dgnn.utils import build_dynamic_graph, load_dataset
from dgnn.temporal_sampler import TemporalSampler

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="REDDIT")
parser.add_argument("--model", type=str)
parser.add_argument("--batch_size", type=int, default=600)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

np.random.seed(args.seed)

args.model = args.model.lower()


class NegLinkSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(0, self.num_nodes, (n, ))


def main():
    # Create a dynamic graph
    _, _, _, df = load_dataset(args.dataset)
    dgraph = build_dynamic_graph(df, add_reverse=True)

    # Create a temporal sampler

    if args.model == "tgn":
        sampler = TemporalSampler(
            dgraph, fanouts=[10], strategy="recent")
    elif args.model == "tgat":
        sampler = TemporalSampler(
            dgraph, fanouts=[10, 10], strategy="uniform")
    elif args.model == "dysat":
        sampler = TemporalSampler(
            dgraph, fanouts=[10, 10], num_snapshots=3,
            snapshot_time_window=10000, prop_time=True,
            strategy="uniform")
    else:
        raise ValueError("Unknown model: {}".format(args.model))

    neg_link_sampler = NegLinkSampler(dgraph.num_vertices())

    for _, rows in tqdm(df.groupby(df.index // args.batch_size)):
        # Sample a batch of data
        root_nodes = np.concatenate(
            [rows.src.values, rows.dst.values, neg_link_sampler.sample(
                len(rows))]).astype(np.int64)
        ts = np.concatenate(
            [rows.time.values, rows.time.values, rows.time.values]).astype(
            np.float32)

        sampler._sampler.sample(root_nodes, ts)


if __name__ == "__main__":
    main()
