import unittest

import torch

from dgnn.config import get_default_config
from dgnn.models.dgnn import DGNN
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import (build_dynamic_graph, get_batch, load_dataset,
                        load_feat, mfgs_to_cuda, prepare_input)


class TestModel(unittest.TestCase):
    def test_tgn_forward(self):
        node_feats, edge_feats = load_feat('REDDIT')
        train_df, val_df, test_df, df = load_dataset('REDDIT')
        model_config, data_config = get_default_config('TGN', 'REDDIT')
        dgraph = build_dynamic_graph(df, **data_config)
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
