import argparse
import random
import time
import os

import numpy as np
import torch
from tqdm import tqdm

from gnnflow.config import get_default_config
from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import build_dynamic_graph, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="REDDIT")
parser.add_argument("--model", type=str)
parser.add_argument("--batch_size", type=int, default=600)
parser.add_argument("--ingestion-batch-size", type=int, default=100000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--repeat", type=int, default=10)
parser.add_argument("--sort", action="store_true")

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
    dgraph = build_dynamic_graph(**dataset_config)
    # insert in batch
    for i in tqdm(range(0, len(df), args.ingestion_batch_size)):
        batch = df[i:i + args.ingestion_batch_size]
        src_nodes = batch["src"].values.astype(np.int64)
        dst_nodes = batch["dst"].values.astype(np.int64)
        timestamps = batch["time"].values.astype(np.float32)
        eids = batch["eid"].values.astype(np.int64)
        dgraph.add_edges(src_nodes, dst_nodes, timestamps,
                         eids, add_reverse=dataset_config["undirected"])

    sampler = TemporalSampler(
        dgraph, **model_config)

    neg_link_sampler = NegLinkSampler(dgraph.num_vertices())

    throughput_list = []
    for _ in range(args.repeat):
        total_time = 0
        for _, rows in tqdm(df.groupby(df.index // args.batch_size)):
            # Sample a batch of data
            root_nodes = np.concatenate(
                [rows.src.values, rows.dst.values,
                    neg_link_sampler.sample(len(rows))]).astype(np.int64)
            ts = np.concatenate(
                [rows.time.values, rows.time.values, rows.time.values]).astype(
                np.float32)

            start = time.time()
            _, sort_time = sampler._sample(root_nodes, ts, sort=args.sort)
            end = time.time()
            total_time += end - start - sort_time

        throughput_list.append(len(df) / total_time)

    print("Throughput for {}'s sampling on {}: {:.2f} samples/s, std: {:.2f}, std/mean: {:.2f}".format(
        args.model, args.dataset, np.mean(
            throughput_list), np.std(throughput_list),
        np.std(throughput_list) / np.mean(throughput_list)))

    subdir = "tmp_res/sampling_nextdoor_test_{}/".format(
        "sorted" if args.sort else "unsorted")
    os.makedirs(subdir, exist_ok=True)
    np.save(os.path.join(subdir, "sampling_throughput_{}_{}.npy".format(
        args.model, args.dataset)), np.mean(throughput_list))


if __name__ == "__main__":
    main()
