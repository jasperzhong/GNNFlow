import argparse
import os
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
parser.add_argument("--models", type=str, nargs='+', default=[])
parser.add_argument("--ingestion-batch-size", type=int, default=100000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--repeat", type=int, default=10000)
parser.add_argument("--sort", action="store_true")
parser.add_argument("--adaptive-block-size-strategy",
                    type=str, default="naive")
parser.add_argument("--mem-resource-type", type=str, default="pinned")
args = parser.parse_args()

MB = 1 << 20
GB = 1 << 30


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)


class NegLinkSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(0, self.num_nodes, (n, ))


def main():
    # Create a dynamic graph
    _, _, _, df = load_dataset(args.dataset)
    _, dataset_config = get_default_config('TGN', args.dataset)
    dataset_config['adaptive_block_size_strategy'] = args.adaptive_block_size_strategy
    dataset_config['mem_resource_type'] = args.mem_resource_type
    try:
        dgraph = build_dynamic_graph(
            **dataset_config, device=0)
    except Exception as e:
        print("Failed to build dynamic graph with error: {}".format(e))
        subdir = 'tmp_res'
        os.makedirs(subdir, exist_ok=True)
        for model in args.models:
            np.save(f'{subdir}/{model}_{args.dataset}_{args.adaptive_block_size_strategy}_{args.mem_resource_type}.npy',
                    0)
        return

    # insert in batch
    for i in tqdm(range(0, len(df), args.ingestion_batch_size)):
        batch = df[i:i + args.ingestion_batch_size]
        src_nodes = batch["src"].values.astype(np.int64)
        dst_nodes = batch["dst"].values.astype(np.int64)
        timestamps = batch["time"].values.astype(np.float32)
        eids = batch["eid"].values.astype(np.int64)
        dgraph.add_edges(src_nodes, dst_nodes, timestamps,
                         eids, add_reverse=dataset_config["undirected"])

    for model in args.models:
        print(model)
        model_config, dataset_config = get_default_config(model, args.dataset)
        if model == 'DySAT' and args.dataset == 'GDELT':
            print("snapshot_time_window set to 25")
            model_config['snapshot_time_window'] = 25

        sampler = TemporalSampler(
            dgraph, **model_config)

        neg_link_sampler = NegLinkSampler(dgraph.num_vertices())

        batch_size = model_config['batch_size']
        i = 0
        total_time = 0
        t = tqdm()
        while True:
            for _, rows in df.groupby(df.index // batch_size):
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
                i += 1
                t.update(1)
                if i == args.repeat:
                    break
            if i == args.repeat:
                break
        t.close()

        print("Throughput for {}'s sampling on {} with {} and {}: {:.2f} samples/s".format(
            model, args.dataset, args.adaptive_block_size_strategy,
            args.mem_resource_type,
            args.repeat * batch_size / total_time))

        subdir = 'tmp_res'
        os.makedirs(subdir, exist_ok=True)
        np.save(f'{subdir}/{model}_{args.dataset}_{args.adaptive_block_size_strategy}_{args.mem_resource_type}.npy',
                args.repeat * batch_size / total_time)


if __name__ == "__main__":
    main()
