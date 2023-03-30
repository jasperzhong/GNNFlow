import argparse
import os
import numpy as np
import random
import torch
from tqdm import tqdm

from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import build_dynamic_graph, load_dataset, mfgs_to_cuda, get_pinned_buffers, load_feat
from gnnflow.config import get_default_config
import gnnflow.cache as caches

cache_names = sorted(name for name in caches.__dict__
                     if not name.startswith("__")
                     and callable(caches.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="REDDIT")
parser.add_argument("--model", type=str)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cache", choices=cache_names, help="feature cache:" +
                    '|'.join(cache_names))
parser.add_argument("--edge-cache-ratio", type=float, default=0,
                    help="cache ratio for edge feature cache")
parser.add_argument("--node-cache-ratio", type=float, default=0,
                    help="cache ratio for node feature cache")
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
    train_df, _, _, df = load_dataset(args.dataset)
    model_config,  dataset_config = get_default_config(
        args.model[len("shuffle_"):], args.dataset)
    dgraph = build_dynamic_graph(
        **dataset_config, dataset_df=df)
    batch_size = model_config['batch_size']

    sampler = TemporalSampler(dgraph, **model_config)

    num_nodes = dgraph.max_vertex_id() + 1

    num_edges = dgraph.num_edges()
    # put the features in shared memory when using distributed training
    node_feats, edge_feats = load_feat(args.dataset)

    if node_feats is None:
        node_feats = torch.randn(num_nodes, 128)
    dim_node = 0 if node_feats is None else node_feats.shape[1]
    dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(
        model_config['fanouts'], model_config['num_snapshots'], batch_size,
        dim_node, dim_edge)

    # Cache
    cache = caches.__dict__[args.cache](args.edge_cache_ratio, args.node_cache_ratio,
                                        num_nodes, num_edges, "cuda",
                                        node_feats, edge_feats,
                                        dim_node, dim_edge,
                                        pinned_nfeat_buffs,
                                        pinned_efeat_buffs,
                                        None,
                                        False)

    # only gnnlab static need to pass param
    if args.cache == 'GNNLabStaticCache':
        cache.init_cache(sampler=sampler, train_df=train_df,
                         pre_sampling_rounds=2)
    else:
        cache.init_cache()

    print("cache mem size: {:.2f} MB".format(
        cache.get_mem_size() / 1000 / 1000))

    cache_edge_ratio_sum = 0
    cache_node_ratio_sum = 0
    i = 0

    if args.model.startswith("shuffle_"):
        df = df.sample(frac=1).reset_index(drop=True)

    for e in range(10):
        cache.reset()
        for _, rows in tqdm(df.groupby(df.index // batch_size)):
            # Sample a batch of data
            root_nodes = np.concatenate(
                [rows.src.values, rows.dst.values]).astype(np.int64)
            ts = np.concatenate(
                [rows.time.values, rows.time.values]).astype(
                np.float32)

            mfgs = sampler.sample(root_nodes, ts)
            mfgs = mfgs_to_cuda(mfgs, 'cuda')
            mfgs = cache.fetch_feature(mfgs)

            cache_edge_ratio_sum += cache.cache_edge_ratio
            cache_node_ratio_sum += cache.cache_node_ratio
            i += 1
    cache_node_hit_ratio = cache_node_ratio_sum / (i + 1)
    cache_edge_hit_ratio = cache_edge_ratio_sum / (i + 1)
    return cache_node_hit_ratio.item(), cache_edge_hit_ratio.item()


if __name__ == "__main__":
    try:
        cache_node_hit_ratio, cache_edge_hit_ratio = main()
    except Exception as e:
        cache_node_hit_ratio = 0
        cache_edge_hit_ratio = 0
        print(e)

    sub_dir = "tmp_res/cache_hit_rate_10epochs/"
    os.makedirs(sub_dir, exist_ok=True)
    print(
        f"node cache ratio: {args.node_cache_ratio} cache_node_hit_ratio: {cache_node_hit_ratio:.2f}\nedge cache ratio: {args.edge_cache_ratio} cache_edge_hit_ratio: {cache_edge_hit_ratio:.2f}")
    np.save(
        sub_dir + f"cache_node_hit_ratio_{args.model}_{args.dataset}_{args.cache}_{args.node_cache_ratio:.1f}.npy", cache_node_hit_ratio)
    np.save(
        sub_dir + f"cache_edge_hit_ratio_{args.model}_{args.dataset}_{args.cache}_{args.edge_cache_ratio:.1f}.npy", cache_edge_hit_ratio)
