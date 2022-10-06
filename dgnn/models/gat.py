from typing import List
import torch.nn as nn
import torch.nn.functional as F

import dgl.nn as dglnn
from dgl.heterograph import DGLBlock


class GAT(nn.Module):
    def __init__(self, dim_in: int, dim_out: int,
                 num_layers: int = 2,
                 attn_head: List[int] = [8, 1],
                 feat_drop: float = 0.6,
                 attn_drop: float = 0.6,
                 allow_zero_in_degree: bool = False):
        if num_layers != len(attn_head):
            raise ValueError(
                "length of attn head {} must equal to num_layers {}".format(
                    attn_head, num_layers))
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleDict()
        # TODO: gat should deal with zero in-degree problem
        for l in range(num_layers):
            # static graph doesn't have snapshot
            key = 'l' + str(l) + 'h' + str(0)
            if l == 0:
                self.layers[key] = dglnn.GATConv(
                    dim_in,
                    dim_out,
                    attn_head[0],
                    feat_drop=0.6,
                    attn_drop=0.6,
                    activation=F.elu,
                    allow_zero_in_degree=allow_zero_in_degree
                )
            else:
                self.layers[key] = dglnn.GATConv(
                    dim_out,
                    dim_out,
                    attn_head[l],
                    feat_drop=0.6,
                    attn_drop=0.6,
                    activation=None,
                    allow_zero_in_degree=allow_zero_in_degree
                )

        self.dim_out = dim_out
        # use the same predictor as graphSage
        self.predictor = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, 1))

    def forward(self, mfgs: List[List[DGLBlock]], neg_sample_ratio: int = 1):
        for l in range(self.num_layers):
            key = 'l' + str(l) + 'h' + str(0)
            h = self.layers[key](mfgs[l][0], mfgs[l][0].srcdata['h'])
            if l != self.num_layers - 1:  # not last layer
                h = h.mean(1)
                mfgs[l + 1][0].srcdata['h'] = h
            else:
                h = h.flatten(1)

        # TODO:use neg_sample_ratio
        src_h, pos_dst_h, neg_dst_h = h.tensor_split(3)
        h_pos = self.predictor(src_h * pos_dst_h)
        # TODO: it seems that neg sample of static graph is different from dynamic
        h_neg = self.predictor(src_h * neg_dst_h)
        return h_pos, h_neg
