import argparse
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed
import torch.nn
import torch.nn.parallel
import torch.utils.data
from tsne_torch import TorchTSNE as TSNE

import gnnflow.cache as caches
from gnnflow.config import get_default_config
from gnnflow.models.dgnn import DGNN
from gnnflow.models.gat import GAT
from gnnflow.models.graphsage import SAGE
from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import (build_dynamic_graph, get_batch_no_neg,
                           get_pinned_buffers, get_project_root_dir,
                           load_dataset, load_feat, mfgs_to_cuda,
                           prepare_input)

datasets = ['REDDIT', 'GDELT', 'LASTFM', 'MAG', 'MOOC', 'WIKI']
model_names = ['TGN', 'TGAT', 'DySAT', 'GRAPHSAGE', 'GAT']
cache_names = sorted(name for name in caches.__dict__
                     if not name.startswith("__")
                     and callable(caches.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=model_names, required=True,
                    help="model architecture" + '|'.join(model_names))
parser.add_argument("--data", choices=datasets, required=True,
                    help="dataset:" + '|'.join(datasets))
parser.add_argument("--seed", type=int, default=42)

# optimization
parser.add_argument("--cache", choices=cache_names, help="feature cache:" +
                    '|'.join(cache_names), default='LRUCache')
parser.add_argument("--edge-cache-ratio", type=float, default=0.2,
                    help="cache ratio for edge feature cache")
parser.add_argument("--node-cache-ratio", type=float, default=0.2,
                    help="cache ratio for node feature cache")

args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
logging.info(args)

checkpoint_path = os.path.join(get_project_root_dir(),
                               '{}.pt'.format(args.model))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)


def main():
    args.distributed = False
    args.local_rank = args.rank = 0
    args.local_world_size = args.world_size = 1
    model_config, data_config = get_default_config(args.model, args.data)
    args.use_memory = model_config['use_memory']

    _, _, test_data, full_data = load_dataset(args.data)
    dgraph = build_dynamic_graph(
        **data_config, device=args.local_rank, dataset_df=full_data)

    num_nodes = dgraph.num_vertices() + 1
    num_edges = dgraph.num_edges()

    batch_size = 4000
    node_feats, edge_feats = load_feat(
        args.data, shared_memory=args.distributed,
        local_rank=args.local_rank, local_world_size=args.local_world_size)

    dim_node = 0 if node_feats is None else node_feats.shape[1]
    dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

    device = torch.device('cuda:{}'.format(args.local_rank))
    logging.debug("device: {}".format(device))

    if args.model == "GRAPHSAGE":
        model = SAGE(dim_node, model_config['dim_embed'])
    elif args.model == 'GAT':
        model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                     memory_device=device, memory_shared=args.distributed)
    else:
        model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                     memory_device=device, memory_shared=args.distributed)
    model.to(device)

    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(
        model_config['fanouts'], model_config['num_snapshots'], batch_size,
        dim_node, dim_edge)

    sampler = TemporalSampler(dgraph, **model_config)

    # Cache
    cache = caches.__dict__[args.cache](args.edge_cache_ratio, args.node_cache_ratio,
                                        num_nodes, num_edges, device,
                                        node_feats, edge_feats,
                                        dim_node, dim_edge,
                                        pinned_nfeat_buffs,
                                        pinned_efeat_buffs,
                                        None,
                                        False)

    # only gnnlab static need to pass param
    if args.cache == 'GNNLabStaticCache':
        cache.init_cache(sampler=sampler, train_df=test_data,
                         pre_sampling_rounds=2)
    else:
        cache.init_cache()

    logging.info("cache mem size: {:.2f} MB".format(
        cache.get_mem_size() / 1000 / 1000))

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if args.use_memory:
        model.memory.restore(ckpt['memory'])
    logging.info("model loaded from {}".format(checkpoint_path))

    def get_embed(snapshot_time_window, repeat=10):
        embeds_list = []
        model.eval()
        for _ in range(repeat):
            embeds = []
            model_config['snapshot_time_window'] = snapshot_time_window
            with torch.no_grad():
                for target_nodes, ts, eid in get_batch_no_neg(test_data, batch_size):
                    mfgs = sampler.sample(target_nodes, ts)

                    mfgs_to_cuda(mfgs, device)
                    mfgs = cache.fetch_feature(mfgs, eid)

                    embed = model(mfgs, return_embed=True)

                    embeds.append(embed.cpu().numpy())

                    if args.use_memory:
                        with torch.no_grad():
                            # use one function
                            model.memory.update_mem_mail(
                                **model.last_updated, edge_feats=cache.target_edge_features,
                                neg_sample_ratio=0)

            embeds = np.concatenate(embeds, axis=0)[-3000:]
            embeds_list.append(embeds)
        return torch.from_numpy(np.mean(embeds_list, axis=0))

    embed1 = get_embed(0)
    embed2 = get_embed(86400)
    embed3 = get_embed(3600)

    # scale
    embed1 = embed1 / torch.norm(embed1, dim=1, keepdim=True)
    embed2 = embed2 / torch.norm(embed2, dim=1, keepdim=True)
    embed3 = embed3 / torch.norm(embed3, dim=1, keepdim=True)

    tsne = TSNE(n_components=2, initial_dims=100, verbose=True)
    embed1 = tsne.fit_transform(embed1)
    embed2 = tsne.fit_transform(embed2)
    embed3 = tsne.fit_transform(embed3)

    plt.rcParams.update({'font.size': 32, 'font.family': 'Myriad Pro'})
    plt.title(f"{args.model} on {args.data} node embedding visualization")
    plt.scatter(embed1[:, 0], embed1[:, 1], c='r', label='full data', s=5)
    plt.scatter(embed2[:, 0], embed2[:, 1], c='b',
                label='sliding time window (T=1d)', s=5)
    plt.scatter(embed3[:, 0], embed3[:, 1], c='g',
                label='sliding time window (T=1hr)', s=5)
    plt.legend()
    plt.savefig(f"{args.model}_{args.data}.png", dpi=400, bbox_inches='tight')


if __name__ == "__main__":
    main()
