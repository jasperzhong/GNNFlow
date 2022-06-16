import torch

from .layers import *
from .memory_updater import *


class JODIE(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, sample_param, memory_param,
                 gnn_param, train_param, combined=False):
        super(JODIE, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param

        # no dim_out in gnn_param
        gnn_param['dim_out'] = memory_param['dim_out']
        self.gnn_param = gnn_param
        self.train_param = train_param

        # Memory updater
        self.memory_updater = RNNMemeoryUpdater(
            memory_param, 2 * memory_param['dim_out'] + dim_edge,
            memory_param['dim_out'],
            memory_param['dim_time'],
            dim_node)
        self.dim_node_input = memory_param['dim_out']

        self.layers = torch.nn.ModuleDict()

        self.gnn_param['layer'] = 1
        for h in range(sample_param['history']):
            self.layers['l0h' + str(h)] = IdentityNormLayer(self.dim_node_input)
            self.layers['l0h' + str(h) + 't'] = JODIETimeEmbedding(gnn_param['dim_out'])

        self.edge_predictor = EdgePredictor(gnn_param['dim_out'])

    def forward(self, mfgs, neg_samples=1):
        self.memory_updater(mfgs[0])

        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l]
                                                        [h].dstdata['mem_ts'], mfgs[l][h].srcdata['ts'])

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
        self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l]
                                                        [h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
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
