import torch
import dgl
from .memory_updater import *
from .layers import *

sample_param = {
    'layer': 1,
    'neighbor': 10,
    'strategy': 'recent',
    'prop_time': False,
    'history': 1,
    'duration': 0,
    'num_thread': 32
}

memory_param = {
    'type': 'node',
    'dim_time': 100,
    'deliver_to': 'self',
    'mail_combine': 'last',
    'memory_update': 'gru',
    'mailbox_size': 1,
    'combine_node_feature': True,
    'dim_out': 100
}

gnn_param = {
    'arch': 'transformer_attention',
    'layer': 1,
    'att_head': 2,
    'dim_time': 100,
    'dim_out': 100,
}

train_param = {
    'epoch': 100,
    'batch_size': 600,
    # reorder: 16
    'lr': 0.0001,
    'dropout': 0.2,
    'att_dropout': 0.2,
    'all_on_gpu': True
}

class TGN(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, sample_param, memory_param, gnn_param, train_param, combined=False):
        super(TGN, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param

        self.gnn_param = gnn_param
        self.train_param = train_param

        # Memory updater
        self.memory_updater = GRUMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
        self.dim_node_input = memory_param['dim_out']
        
        self.layers = torch.nn.ModuleDict()
        for h in range(sample_param['history']):
            self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=combined)
        for l in range(1, gnn_param['layer']):
            for h in range(sample_param['history']):
                self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=False)
        
        
        self.edge_predictor = EdgePredictor(gnn_param['dim_out'])
    
    def forward(self, mfgs):
        self.memory_updater(mfgs[0])
        
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
        return self.edge_predictor(out)

    def get_emb(self, mfgs):
        self.memory_updater(mfgs[0])
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
