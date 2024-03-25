import argparse
import os
import random
import time

import numpy as np
import pandas as pd 
import torch
from tqdm import tqdm

from gnnflow.config import get_default_config
from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import build_dynamic_graph, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="NETFLIX")
parser.add_argument("--model", type=str, default='TGAT')
parser.add_argument("--ingestion-batch-size", type=int, default=int(1e9))
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

MB = 1 << 20
GB = 1 << 30


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_chunks_by_days_movielens(df: pd.DataFrame, days=1):
    df['time_day'] = (df['time'] / 86400 // days).astype(int)
    grouped = df.groupby('time_day')
    chunks = [group for _, group in grouped]
    return chunks



set_seed(args.seed)


class NegLinkSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(0, self.num_nodes, (n, ))


def main():
    # Create a dynamic graph
    df = pd.read_feather("/data/hgnn/ml-25m/ratings.feather")
    # _, _, _, df = load_dataset(args.dataset)

    _, dataset_config = get_default_config('TGAT', args.dataset)
    dgraph = build_dynamic_graph(
            **dataset_config, device=0)


    # insert in batch
    for i in tqdm(range(0, len(df), args.ingestion_batch_size)):
        batch = df[i:i + args.ingestion_batch_size]
        src_nodes = batch["src"].values.astype(np.int64)
        dst_nodes = batch["dst"].values.astype(np.int64)
        timestamps = batch["time"].values.astype(np.float32)
        eids = batch["eid"].values.astype(np.int64)
        dgraph.add_edges(src_nodes, dst_nodes, timestamps,
                         eids, add_reverse=dataset_config["undirected"])

    model_config, dataset_config = get_default_config(args.model, args.dataset)
    if args.model == 'DySAT' and args.dataset == 'GDELT':
        print("snapshot_time_window set to 25")
        model_config['snapshot_time_window'] = 25

    sampler = TemporalSampler(
        dgraph, **model_config)

    chunks = split_chunks_by_days_movielens(df)
    import pdb 
    pdb.set_trace()

    cache = set()
    num_reused_list = []
    num_tot_list = []
    for chunk in tqdm(chunks):
        num_reused = 0
        num_tot = 0
        # Sample a batch of data
        root_nodes = np.concatenate(
            [chunk.src.values, chunk.dst.values]).astype(np.int64)
        timestamps = np.concatenate(
            [chunk.time.values, chunk.time.values]).astype(
            np.float32)

        mfgs = sampler.sample(root_nodes, timestamps)
        b = mfgs[1][0]
        num_dst_nodes = b.num_dst_nodes()
        sampled_nodes = b.srcdata['ID'][num_dst_nodes:].tolist()
        sampled_ts = b.srcdata['ts'][num_dst_nodes:].tolist()

        num_tot += len(sampled_nodes)

        for node, ts in zip(sampled_nodes, sampled_ts):
            key = (node, ts)
            if key in cache:
                num_reused += 1

        keys = list(zip(b.srcdata['ID'].tolist(), b.srcdata['ts'].tolist()))
        cache.update(keys)

        num_reused_list.append(num_reused)
        num_tot_list.append(num_tot)

    import pdb 
    pdb.set_trace()


if __name__ == "__main__":
    main()
