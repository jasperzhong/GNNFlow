from typing import List, Optional

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.heterograph import DGLBlock


class SAGE(nn.Module):
    def __init__(self, dim_node: int, dim_out: int,
                 num_layers: int = 2,
                 aggregator: Optional[str] = 'mean'):

        if aggregator not in ['mean', 'gcn', 'pool', 'lstm']:
            raise ValueError(
                "aggregator {} is not in ['mean', 'gcn', 'pool', 'lstm']".format(aggregator))
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleDict()
        for l in range(num_layers):
            # static graph doesn't have snapshot
            key = 'l' + str(l) + 'h' + str(0)
            if l == 0:
                self.layers[key] = dglnn.SAGEConv(
                    dim_node, dim_out, aggregator)
            else:
                self.layers[key] = dglnn.SAGEConv(
                    dim_out, dim_out, aggregator)

        self.dim_out = dim_out
        self.predictor = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, 1))

    def reset(self):
        pass

    def forward(self, mfgs: List[List[DGLBlock]], neg_sample_ratio: int = 1, *args, **kwargs):
        """
        Args:
            b: sampled message flow graph (mfg), where
                `b.num_dst_nodes()` is the number of target nodes to sample,
                `b.srcdata['h']` is the embedding of all nodes,
                `b.edge['f']` is the edge features of sampled edges, and
                `b.edata['dt']` is the delta time of sampled edges.

        Returns:
            output: output embedding of target nodes (shape: (num_dst_nodes, dim_out))
        """
        for l in range(self.num_layers):
            key = 'l' + str(l) + 'h' + str(0)
            h = self.layers[key](mfgs[l][0], mfgs[l][0].srcdata['h'])
            if l != self.num_layers - 1:
                h = F.relu(h)
                mfgs[l + 1][0].srcdata['h'] = h

        num_edge = h.shape[0] // (neg_sample_ratio + 2)
        src_h = h[:num_edge]
        pos_dst_h = h[num_edge:2 * num_edge]
        neg_dst_h = h[2 * num_edge:]
        h_pos = self.predictor(src_h * pos_dst_h)
        # TODO: it seems that neg sample of static graph is different from dynamic
        h_neg = self.predictor(src_h.tile(neg_sample_ratio, 1) * neg_dst_h)
        return h_pos, h_neg
