import unittest

import torch
import numpy as np
from gnnflow.config import get_default_config
from gnnflow.models.dgnn import DGNN
from gnnflow.models.gat import GAT
from gnnflow.models.graphsage import SAGE
from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import (build_dynamic_graph, get_batch, load_dataset,
                           load_feat, mfgs_to_cuda, prepare_input)
from gnnflow.cache import LFUCache


class TestModel(unittest.TestCase):
    def test_tgn_forward(self):
        node_feats, edge_feats = load_feat('REDDIT')
        train_df, val_df, test_df, df = load_dataset('REDDIT')
        model_config, data_config = get_default_config('TGN', 'REDDIT')
        dgraph = build_dynamic_graph(**data_config, dataset_df=df)
        gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
        gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
        batch_size = 600
        device = torch.device("cuda:0")
        model = DGNN(
            gnn_dim_node, gnn_dim_edge, **model_config,
            num_nodes=dgraph.num_vertices(),
            memory_device=device).to(device)

        sampler = TemporalSampler(dgraph, [10])
        it = iter(get_batch(train_df, batch_size))
        target_nodes, ts, eid = it.__next__()
        mfgs = sampler.sample(target_nodes, ts)

        mfgs = prepare_input(mfgs, node_feats, edge_feats)
        mfgs_to_cuda(mfgs, device)

        pred_pos, pred_neg = model(mfgs, eid=eid, edge_feats=edge_feats,
                                   neg_sample_ratio=0)

    def test_graph_sage_forward(self):
        edge_feats = torch.randn(411749, 172)
        node_feats = torch.randn(7144, 172)
        train_df, val_df, test_df, df = load_dataset('MOOC')
        model_config, data_config = get_default_config('TGN', 'MOOC')
        dgraph = build_dynamic_graph(**data_config, dataset_df=df)
        gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
        batch_size = 600
        device = torch.device("cuda:0")
        model = SAGE(gnn_dim_node, 100).to(device)

        sampler = TemporalSampler(dgraph, [3, 3, 3])
        it = iter(get_batch(train_df, batch_size))
        target_nodes, ts, eid = it.__next__()
        mfgs = sampler.sample(target_nodes, np.full(
            target_nodes.shape, np.finfo(np.float32).max))

        cache = LFUCache(0.0005, 7144,
                         dgraph.num_edges(), device,
                         node_feats, edge_feats,
                         172, 172,
                         None,
                         None,
                         None, False, 0)
        mfgs_to_cuda(mfgs, device)
        mfgs = cache.fetch_feature(mfgs, eid)
        mfgs = prepare_input(mfgs, node_feats, edge_feats)

        pred_pos, pred_neg = model(mfgs, neg_sample_ratio=0)

    def test_gat_forward(self):
        edge_feats = torch.randn(411749, 172)
        node_feats = torch.randn(7144, 172)
        train_df, val_df, test_df, df = load_dataset('MOOC')
        model_config, data_config = get_default_config('TGN', 'MOOC')
        dgraph = build_dynamic_graph(**data_config, dataset_df=df)
        gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
        batch_size = 600
        device = torch.device("cuda:0")
        model = GAT(gnn_dim_node, 100, allow_zero_in_degree=True).to(device)

        sampler = TemporalSampler(dgraph, [3, 3])
        it = iter(get_batch(train_df, batch_size))
        target_nodes, ts, eid = it.__next__()
        mfgs = sampler.sample(target_nodes, np.full(
            target_nodes.shape, np.finfo(np.float32).max))

        mfgs = prepare_input(mfgs, node_feats, edge_feats)
        mfgs_to_cuda(mfgs, device)

        pred_pos, pred_neg = model(mfgs, neg_sample_ratio=0)


if __name__ == "__main__":
    unittest.main()
