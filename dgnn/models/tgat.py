import torch

from .layers import *
from .memory_updater import *


class tgat(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, sample_param, memory_param,
                 gnn_param, train_param, combined=False):
        super(tgat, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param

        self.gnn_param = gnn_param
        self.train_param = train_param

        # no memory
        self.layers = torch.nn.ModuleDict()
        for h in range(sample_param['history']):
            self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.
                                                                   dim_node_input, dim_edge, gnn_param['dim_time'],
                                                                   gnn_param['att_head'],
                                                                   train_param['dropout'],
                                                                   train_param['att_dropout'],
                                                                   gnn_param['dim_out'],
                                                                   combined=combined)
        for l in range(1, gnn_param['layer']):
            for h in range(sample_param['history']):
                self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'],
                                                                                    gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=False)

        self.edge_predictor = EdgePredictor(gnn_param['dim_out'])

    def forward(self, mfgs, neg_samples=1):
        # prepare_input_mails
        # 
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])

                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return self.edge_predictor(out, neg_samples=neg_samples)

    def get_emb(self, mfgs):
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return out
