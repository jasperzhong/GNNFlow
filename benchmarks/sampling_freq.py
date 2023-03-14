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


def main():
    # Create a dynamic graph
    _, _, _, df = load_dataset(args.dataset)
    model_config,  dataset_config = get_default_config(
        args.model, args.dataset)
    dataset_config["mem_resource_type"] = args.mem_resource_type
    dgraph = build_dynamic_graph(
        **dataset_config, dataset_df=df)

    sampler = TemporalSampler(dgraph, **model_config)

    node_to_cnt = {}
    edge_to_cnt = {}
    for _, rows in tqdm(df.groupby(df.index // args.batch_size)):
        # Sample a batch of data
        root_nodes = np.concatenate(
            [rows.src.values, rows.dst.values]).astype(np.int64)
        ts = np.concatenate(
            [rows.time.values, rows.time.values]).astype(
            np.float32)

        if args.stat:
            blocks = sampler.sample(root_nodes, ts)
            for block in blocks:
                for b in block:
                    all_nodes = b.srcdata['ID'].tolist()
                    for node in all_nodes:
                        if node not in node_to_cnt:
                            node_to_cnt[node] = 0
                        node_to_cnt[node] += 1

                    all_edges = b.edata['ID'].tolist()
                    for edge in all_edges:
                        if edge not in edge_to_cnt:
                            edge_to_cnt[edge] = 0
                        edge_to_cnt[edge] += 1
        else:
            sampler._sampler.sample(root_nodes, ts)

    if args.stat:
        # print("Total sampled nodes: {}".format(total_sampled_nodes))
        import pickle

        with open("{}_{}_node_to_cnt.pickle".format(args.model, args.dataset), 'wb') as f:
            pickle.dump(node_to_cnt, f)
        with open("{}_{}_edge_to_cnt.pickle".format(args.model, args.dataset), 'wb') as f:
            pickle.dump(edge_to_cnt, f)


if __name__ == "__main__":
    main()
