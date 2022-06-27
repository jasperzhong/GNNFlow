import unittest

import numpy as np
import torch

from dgnn.models import TGN


class TestModel(unittest.TestCase):

    def setUp(self):
        # The same setup using REDDIT Dataset
        self.node_feats = None
        self.gnn_dim_node = 0 if self.node_feats is None else self.node_feats.shape[1]
        self.gnn_dim_edge = 172
        self.sample_param = {
            'layer': 1,
            'neighbor': 10,
            'strategy': 'recent',
            'prop_time': False,
            'history': 1,
            'duration': 0,
            'num_thread': 32
        }

        self.memory_param = {
            'type': 'node',
            'dim_time': 100,
            'deliver_to': 'self',
            'mail_combine': 'last',
            'memory_update': 'gru',
            'mailbox_size': 1,
            'combine_node_feature': True,
            'dim_out': 100
        }

        self.gnn_param = {
            'arch': 'transformer_attention',
            'layer': 1,
            'att_head': 2,
            'dim_time': 100,
            'dim_out': 100,
        }

        self.train_param = {
            'epoch': 100,
            'batch_size': 600,
            # reorder: 16
            'lr': 0.0001,
            'dropout': 0.2,
            'att_dropout': 0.2,
            'all_on_gpu': True
        }

        self.combined = False

    def test_tgn_forward(self):

        tgn = TGN(self.gnn_dim_node, self.gnn_dim_edge,
                  self.sample_param, self.memory_param,
                  self.gnn_param, self.train_param, self.combined).cuda()

        mfg = np.load('mfgs.npy', allow_pickle=True)
        mfgs = []
        mfgs.append(mfg)
        pred_pos, pred_neg = tgn(mfgs)

        creterion = torch.nn.BCEWithLogitsLoss()
        loss = creterion(pred_pos, torch.ones_like(pred_pos))
        loss += creterion(pred_neg, torch.zeros_like(pred_neg))
        print("train loss: {}".format(loss))
