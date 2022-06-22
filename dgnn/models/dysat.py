import torch

from .layers import *
from .memory_updater import *


class dysat(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, sample_history=3, memory_dim_out=0,
                 layer=2, gnn_dim_out=100, gnn_dim_time=0, gnn_attn_head=2,
                 dropout=0.1, attn_dropout=0.1, combined=False):
        super(dysat, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge

        self.sample_history = sample_history
        self.memory_dim_out = memory_dim_out
        self.gnn_dim_out = gnn_dim_out
        self.gnn_dim_time = gnn_dim_time
        self.gnn_attn_head = gnn_attn_head
        self.gnn_layer = layer
        self.dropout = dropout
        self.attn_dropout = attn_dropout

        # no memory
        self.layers = torch.nn.ModuleDict()
        for h in range(sample_history):
            self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.
                                                                   dim_node_input, dim_edge, gnn_dim_time,
                                                                   gnn_attn_head,
                                                                   dropout,
                                                                   attn_dropout,
                                                                   gnn_dim_out,
                                                                   combined=combined)
        for l in range(1, layer):
            for h in range(sample_history):
                self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_dim_out, dim_edge, gnn_dim_time,
                                                                                    gnn_attn_head, dropout, attn_dropout, gnn_dim_out, combined=False)
        self.edge_predictor = EdgePredictor(gnn_dim_out)
        # dysat has combiner
        self.combiner = torch.nn.RNN(gnn_dim_out, gnn_dim_out)

    def forward(self, mfgs, neg_samples=1):
        out = list()
        for l in range(self.gnn_layer):
            for h in range(self.sample_history):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if l != self.gnn_layer - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)

        if self.sample_history == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return self.edge_predictor(out, neg_samples=neg_samples)

    def get_emb(self, mfgs):

        out = list()
        for l in range(self.gnn_layer):
            for h in range(self.sample_history):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if l != self.gnn_layer - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        if self.sample_history == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return out
