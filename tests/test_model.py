import unittest

from dgnn.models import TGN
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import load_dataset, load_feat, build_dynamic_graph, mfgs_to_cuda, prepare_input, get_batch

MB = 1 << 20
GB = 1 << 30

default_config = {
    "initial_pool_size": 1 * GB,
    "maximum_pool_size": 1 * GB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 64,
    "blocks_to_preallocate": 128,
    "insertion_policy": "insert",
}


class TestModel(unittest.TestCase):

    def setUp(self):
        # The same setup using REDDIT Dataset
        self.node_feats = None
        self.gnn_dim_node = 0 if self.node_feats is None else self.node_feats.shape[1]
        self.gnn_dim_edge = 172
        self.combined = False

    def test_tgn_forward(self):
        node_feats, edge_feats = load_feat('REDDIT')
        train_df, val_df, test_df, df = load_dataset('REDDIT')
        dgraph = build_dynamic_graph(df, **default_config)
        gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
        gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
        model = TGN(gnn_dim_node, gnn_dim_edge, dgraph.num_vertices())
        model.cuda()

        sampler = TemporalSampler(dgraph, [10])
        it = iter(get_batch(train_df))
        target_nodes, ts, eid = it.__next__()
        mfgs = sampler.sample(target_nodes, ts)

        mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=False)
        mfgs_to_cuda(mfgs)

        pred_pos, pred_neg = model(mfgs)
