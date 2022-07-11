import os
import argparse
import torch
import time
import pandas as pd
from torch_sparse import add_
import dgnn.models as models
from torch.utils.data import BatchSampler, SequentialSampler
from dgnn.sampler import BatchSamplerReorder
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import get_project_root_dir, prepare_input
from dgnn.utils import mfgs_to_cuda, node_to_dgl_blocks
from dgnn.utils import build_dynamic_graph, load_dataset
from dgnn.utils import load_feat, get_batch
from dgnn.dataset import DynamicGraphDataset, default_collate_ndarray
from sklearn.metrics import average_precision_score, roc_auc_score
from test import GeneralModel
from memory_updater import MailBox

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model name", type=str, default='tgat')
parser.add_argument("--epoch", help="training epoch", type=int, default=100)
parser.add_argument("--lr", help='learning rate', type=float, default=0.0001)
parser.add_argument("--batch-size", help="batch size", type=int, default=600)
args = parser.parse_args()


def val(df: pd.DataFrame, sampler: TemporalSampler, mailbox: MailBox, model: torch.nn.Module, node_feats: torch.Tensor, edge_feats: torch.Tensor, creterion: torch.nn.Module, mode='val', neg_samples=1, no_neg=False):
    model.eval()
    val_losses = list()
    aps = list()
    aucs_mrrs = list()

    with torch.no_grad():
        total_loss = 0
        for i, (target_nodes, ts, eid) in enumerate(get_batch(df=df)):

            if sampler is not None:
                if no_neg:
                    pos_root_end = target_nodes.shape[0] * 2 // 3
                    mfgs = sampler.sample(
                        target_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    mfgs = sampler.sample(target_nodes, ts)
            else:
                mfgs = node_to_dgl_blocks(target_nodes, ts)

            mfgs_to_cuda(mfgs)
            mfgs = prepare_input(
                mfgs, node_feats, edge_feats, combine_first=False)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])

            pred_pos, pred_neg = model(mfgs)

            total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze(
                ) < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))

            aps.append(average_precision_score(y_true, y_pred))

            mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            block = None
            if mailbox is not None:
                mailbox.update_mailbox(model.memory_updater.last_updated_nid,
                                       model.memory_updater.last_updated_memory, target_nodes, ts, mem_edge_feats, block)
                mailbox.update_memory(model.memory_updater.last_updated_nid,
                                      model.memory_updater.last_updated_memory, model.memory_updater.last_updated_ts)
        if mode == 'val':
            val_losses.append(float(total_loss))

    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr

# TODO: Use args
# sample_param = {
#             'layer': 1,
#             'neighbor': [10],
#             'strategy': 'recent',
#             'prop_time': False,
#             'history': 1,
#             'duration': 0,
#             'num_thread': 32,
#             # 'no_neg': True
#             }

# memory_param = {
#             'type': 'node',
#             'dim_time': 100,
#             'deliver_to': 'neighbors',
#             'mail_combine': 'last',
#             'memory_update': 'transformer',
#             'attention_head': 2,
#             'mailbox_size': 10,
#             'combine_node_feature': False,
#             'dim_out': 100
#             }

# gnn_param = {
#             'arch': 'identity'
#             }

# train_param = {
#             'epoch': 100,
#             'batch_size': 600,
#             # reorder: 16
#             'lr': 0.0001,
#             'dropout': 0.1,
#             'att_dropout': 0.1,
#             'all_on_gpu': True
#         }


# TGAT
sample_param = {
    'layer': 2,
    'neighbor': [10, 10],
    'strategy': 'uniform',
    'prop_time': False,
    'history': 1,
    'duration': 0,
    'num_thread': 32,
    # 'no_neg': True
}

memory_param = {
    'type': 'none',
            'dim_out': 0
}

gnn_param = {
    'arch': 'transformer_attention',
            'layer': 2,
            'att_head': 2,
            'dim_time': 100,
            'dim_out': 100
}

train_param = {
    'epoch': 100,
    'batch_size': 600,
    # reorder: 16
    'lr': 0.0001,
    'dropout': 0.1,
    'att_dropout': 0.1,
    'all_on_gpu': True
}

# TGN
# sample_param = {
#     'layer': 1,
#     'neighbor': [10],
#     'strategy': 'uniform',
#     'prop_time': False,
#     'history': 1,
#     'duration': 0,
#     'num_thread': 32,
#     'no_neg': True
# }

# memory_param = {
#     'type': 'node',
#             'dim_time': 100,
#             'deliver_to': 'self',
#             'mail_combine': 'last',
#             'memory_update': 'gru',
#             'mailbox_size': 1,
#             'combine_node_feature': True,
#             'dim_out': 100
# }

# gnn_param = {
#     'arch': 'transformer_attention',
#             'layer': 1,
#             'att_head': 2,
#             'dim_time': 100,
#             'dim_out': 100
# }

# train_param = {
#     'epoch': 100,
#     'batch_size': 600,
#     # reorder: 16
#     'lr': 0.0001,
#     'dropout': 0.2,
#     'att_dropout': 0.2,
#     'all_on_gpu': True
# }

# # JODIE
# sample_param = {
#             'no_sample': True,
#             'history': 1
#             }

# memory_param = {
#             'type': 'node',
#             'dim_time': 100,
#             'deliver_to': 'self',
#             'mail_combine': 'last',
#             'memory_update': 'rnn',
#             'mailbox_size': 1,
#             'combine_node_feature': True,
#             'dim_out': 100
#             }

# gnn_param = {
#             'arch': 'identity',
#             'time_transform': 'JODIE'
#             }

# train_param = {
#             'epoch': 100,
#             'batch_size': 600,
#             # reorder: 16
#             'lr': 0.0001,
#     'dropout': 0.1,
#     'all_on_gpu': True
# }

# DySAT
# sample_param = {
#             'layer': 2,
#             'neighbor': [10, 10],
#             'strategy': 'uniform',
#             'prop_time': True,
#             'history': 3,
#             'duration': 10000,
#             'num_thread': 32,
#             'no_neg': True
#             }

# memory_param = {
#             'type': 'none',
#             'dim_out': 0
#             }

# gnn_param = {
#             'arch': 'transformer_attention',
#             'layer': 2,
#             'att_head': 2,
#             'dim_time': 0,
#             'dim_out': 100,
#             'combine': 'rnn'
#             }

# train_param = {
#             'epoch': 50,
#             'batch_size': 600,
#             # reorder: 16
#             'lr': 0.0001,
#             'dropout': 0.1,
#             'att_dropout': 0.1,
#             'all_on_gpu': True
#         }

# Build Graph, block_size = 1024
# path_saver = '../models/{}.pt'.format(args.model)
# path_saver_2 = '../models/tgat_best.pt'
# node_feats, edge_feats = load_feat(None, 'REDDIT')
path_saver = os.path.join(get_project_root_dir(), '{}.pt'.format(args.model))
node_feats, edge_feats = load_feat('REDDIT')
train_df, val_df, test_df, df = load_dataset('REDDIT')
# df[0] consist train part of the data
dgraph = build_dynamic_graph(df, add_reverse=True)
# val_graph = build_dynamic_graph(df[1])
# test_graph = build_dynamic_graph(df[2])

gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

no_neg = ('no_neg' in sample_param and sample_param['no_neg'])

# model = TGN(gnn_dim_node, gnn_dim_edge,
#           sample_param, memory_param,
#           gnn_param, train_param).cuda()

# model = TGAT(gnn_dim_node, gnn_dim_edge,
#           sample_param, memory_param,
#           gnn_param, train_param).cuda()

# model = JODIE(gnn_dim_node, gnn_dim_edge,
#           sample_param, memory_param,
#           gnn_param, train_param).cuda()

# model = APAN(gnn_dim_node, gnn_dim_edge,
#           sample_param, memory_param,
#           gnn_param, train_param).cuda()

# model = DySAT(gnn_dim_node, gnn_dim_edge,
#           sample_param, memory_param,
#           gnn_param, train_param).cuda()

model = GeneralModel(gnn_dim_node, gnn_dim_edge,
                     sample_param, memory_param,
                     gnn_param, train_param).cuda()

# torch.save(model.state_dict(), path_saver)

# TODO: MailBox. Check num_vertices
if memory_param['type'] != 'none':
    mailbox = MailBox(memory_param, dgraph.num_vertices() + 3, gnn_dim_edge)
    mailbox.move_to_gpu()
else:
    mailbox = None

# this sampler only sample train part of the data
sampler = None
val_sampler = None
test_sampler = None
print(sample_param['neighbor'], sample_param['strategy'],
      sample_param['history'], sample_param['duration'])
if not ('no_sample' in sample_param and sample_param['no_sample']):
    sampler = TemporalSampler(dgraph, sample_param['neighbor'], sample_param['strategy'],
                              num_snapshots=sample_param['history'], snapshot_time_window=sample_param['duration'])
    # val_sampler = TemporalSampler(val_graph, sample_param['neighbor'], sample_param['strategy'],
    #                             num_snapshots=sample_param['history'], snapshot_time_window=sample_param['duration'])
    # test_sampler = TemporalSampler(test_graph, sample_param['neighbor'], sample_param['strategy'],
    #                             num_snapshots=sample_param['history'], snapshot_time_window=sample_param['duration'])

creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])

best_ap = 0
best_e = 0

for e in range(args.epoch):
    print("Epoch {}".format(e))
    epoch_time_start = time.time()
    total_loss = 0
    iteration_time = 0
    sample_time = 0
    train_time = 0

    model.train()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None

    for i, (target_nodes, ts, eid) in enumerate(get_batch(train_df)):

        iter_start = time.time()
        # get mfgs from sampler
        # no neg
        if sampler is not None:
            if 'no_neg' in sample_param and sample_param['no_neg']:
                pos_root_end = target_nodes.shape[0] * 2 // 3
                mfgs = sampler.sample(
                    target_nodes[:pos_root_end], ts[:pos_root_end])
                # mfgs = sampler.sample(target_nodes[584:585], ts[584:585])
                # mfgs = sampler.sample(torch.Tensor([242]), torch.Tensor([48066.859]))
            else:
                mfgs = sampler.sample(target_nodes, ts)
        else:
            mfgs = node_to_dgl_blocks(target_nodes, ts)

        mfgs_to_cuda(mfgs)
        sample_end = time.time()
        mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=False)
        prepare_end = time.time()
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0])

        optimizer.zero_grad()
        pred_pos, pred_neg = model(mfgs)

        loss = creterion(pred_pos, torch.ones_like(pred_pos))
        loss += creterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss) * train_param['batch_size']
        loss.backward()

        optimizer.step()

        # MailBox Update: not use edge_feats now
        mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
        block = None
        if mailbox is not None:
            mailbox.update_mailbox(model.memory_updater.last_updated_nid,
                                   model.memory_updater.last_updated_memory, target_nodes, ts, mem_edge_feats, block)
            mailbox.update_memory(model.memory_updater.last_updated_nid,
                                  model.memory_updater.last_updated_memory, model.memory_updater.last_updated_ts)

        train_end = time.time()

        iteration_time += train_end - iter_start
        sample_time += sample_end - iter_start
        train_time += train_end - sample_end

        if i % 100 == 0:
            # print('Iteration:{}. Train loss:{:.4f}'.format(i, loss))
            # print('Iteration time:{:.4f}s; sample time:{:.4f}s; train time:{:.4f}s.'
            #       .format(iteration_time / (i + 1), sample_time / (i + 1), train_time / (i + 1)))
            print(
                "iter {}: iter time: {:.4f}, sample time: {:.4f}, prepare time: {:.4f}, model time: {:.4f}".format(
                    i, train_end - iter_start, sample_end - iter_start, prepare_end - sample_end, train_end - prepare_end))

        torch.save(model.state_dict(), 'tgat_test.pt')

    epoch_time_end = time.time()
    epoch_time = epoch_time_end - epoch_time_start

    # Validation
    print("***Start validation at epoch {}***".format(e))
    val_start = time.time()
    ap, auc = val(val_df, sampler, mailbox, model, node_feats,
                  edge_feats, creterion, mode='val', no_neg=no_neg)
    val_end = time.time()
    val_time = val_end - val_start
    print("epoch train time: {} ; val time: {}; val ap:{:4f}; val auc:{:4f}"
          .format(epoch_time, val_time, ap, auc))
    if e > 1 and ap > best_ap:
        best_e = e
        best_ap = ap
        torch.save(model.state_dict(), path_saver)
        print("Best val AP: {:.4f} & val AUC: {:.4f}".format(ap, auc))

print('Loading model at epoch {}...'.format(best_e))
model.load_state_dict(torch.load(path_saver))
if mailbox is not None:
    mailbox.reset()
    model.memory_updater.last_updated_nid = None
    val(train_df, sampler, mailbox, model, node_feats,
        edge_feats, creterion, mode='train', no_neg=no_neg)
    val(val_df, sampler, mailbox, model, node_feats,
        edge_feats, creterion, mode='val', no_neg=no_neg)
ap, auc = val(test_df, sampler, mailbox, model, node_feats,
              edge_feats, creterion, mode='test', no_neg=no_neg)
print('\ttest ap:{:4f}  test auc:{:4f}'.format(ap, auc))
