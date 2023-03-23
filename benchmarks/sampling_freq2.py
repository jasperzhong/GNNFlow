import argparse

import numpy as np
import random
import torch
from tqdm import tqdm

from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import build_dynamic_graph, load_dataset, mfgs_to_cuda
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

    node_sequence = []
    edge_sequence = []
    for _, rows in tqdm(df.groupby(df.index // args.batch_size)):
        # Sample a batch of data
        root_nodes = np.concatenate(
            [rows.src.values, rows.dst.values]).astype(np.int64)
        ts = np.concatenate(
            [rows.time.values, rows.time.values]).astype(
            np.float32)

        if args.stat:
            blocks = sampler.sample(root_nodes, ts)
            blocks = mfgs_to_cuda(blocks, 'cuda')
            for block in blocks:
                for b in block:
                    all_nodes = b.srcdata['ID'].cpu().numpy()
                    all_edges = b.edata['ID'].cpu().numpy()

                    node_sequence.append(all_nodes)
                    edge_sequence.append(all_edges)
        else:
            sampler._sampler.sample(root_nodes, ts)

    if args.stat:
        sub_dir = "tmp_res/"
        node_sequence = np.concatenate(node_sequence)
        torch.save(node_sequence, sub_dir+"{}_{}_node_sequence.pt".format(
            args.model, args.dataset))
        edge_sequence = np.concatenate(edge_sequence)
        torch.save(edge_sequence, sub_dir+"{}_{}_edge_sequence.pt".format(
            args.model, args.dataset))


if __name__ == "__main__":
    main()
