import argparse
from random import sample
import torch
import time
from dgnn.build_graph import build_dynamic_graph, load_graph, get_batch, load_feat
from dgnn.model.memory_updater import MailBox
from dgnn.model.TGN import TGN
from dgnn.model.TGAT import TGAT
from dgnn.model.JODIE import JODIE
from dgnn.model.APAN import APAN
from dgnn.model.DySAT import DySAT
from dgnn.model.test import GeneralModel
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import prepare_input, mfgs_to_cuda, node_to_dgl_blocks
from scripts.validation import val

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model name", type=str, default='apan')
parser.add_argument("--epoch", help="training epoch", type=int, default=100)
parser.add_argument("--lr", help='learning rate', type=float, default=0.0001)
parser.add_argument("--batch-size", help="batch size", type=int, default=600)
parser.add_argument("--prop_time", help='use prop time')
parser.add_argument("--no_neg", help='not using neg samples in sampling')

args = parser.parse_args()

# TODO: Use args
# APAN
sample_param = {
            'layer': 1,
            'neighbor': [10],
            'strategy': 'recent',
            'prop_time': False,
            'history': 1,
            'duration': 0,
            'num_thread': 32,
            'no_neg': True
            }

memory_param = {
            'type': 'node',
            'dim_time': 100,
            'deliver_to': 'neighbors',
            'mail_combine': 'last',
            'memory_update': 'transformer',
            'attention_head': 2,
            'mailbox_size': 10,
            'combine_node_feature': False,
            'dim_out': 100
            }

gnn_param = {
            'arch': 'identity'
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

# #TGAT
# sample_param = {
#             'layer': 2,
#             'neighbor': [10, 10],
#             'strategy': 'uniform',
#             'prop_time': False,
#             'history': 1,
#             'duration': 0,
#             'num_thread': 32,
#             # 'no_neg': True
#             }

# memory_param = {
#             'type': 'none',
#             'dim_out': 0
#             }

# gnn_param = {
#             'arch': 'transformer_attention',
#             'layer': 2,
#             'att_head': 2,
#             'dim_time': 100,
#             'dim_out': 100
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

# TGN
# sample_param = {
#             'layer': 1,
#             'neighbor': [10],
#             'strategy': 'uniform',
#             'prop_time': False,
#             'history': 1,
#             'duration': 0,
#             'num_thread': 32,
#             'no_neg': True
#             }

# memory_param = {
#             'type': 'node',
#             'dim_time': 100,
#             'deliver_to': 'self',
#             'mail_combine': 'last',
#             'memory_update': 'gru',
#             'mailbox_size': 1,
#             'combine_node_feature': True,
#             'dim_out': 100
#             }

# gnn_param = {
#             'arch': 'transformer_attention',
#             'layer': 1,
#             'att_head': 2,
#             'dim_time': 100,
#             'dim_out': 100
#             }

# train_param = {
#             'epoch': 100,
#             'batch_size': 600,
#             # reorder: 16
#             'lr': 0.0001,
#             'dropout': 0.2,
#             'att_dropout': 0.2,
#             'all_on_gpu': True
#         }

# JODIE
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
#             'dropout': 0.1,
#             'all_on_gpu': True
#         }

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
path_saver = '../models/{}.pt'.format(args.model)
path_saver_2 = '../models/apan_best.pt'
node_feats, edge_feats = load_feat(None, 'REDDIT')
df = load_graph(None, 'REDDIT')
# df[0] consist train part of the data
# df[3] is the full graph
# use the full data to build graph
dgraph = build_dynamic_graph(df[3])

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

model = APAN(gnn_dim_node, gnn_dim_edge,
          sample_param, memory_param,
          gnn_param, train_param).cuda()

# model = DySAT(gnn_dim_node, gnn_dim_edge,
#           sample_param, memory_param,
#           gnn_param, train_param).cuda()

# model = GeneralModel(gnn_dim_node, gnn_dim_edge,
#           sample_param, memory_param,
#           gnn_param, train_param).cuda()


# TODO: MailBox. Check num_vertices
if memory_param['type'] != 'none':
    mailbox = MailBox(memory_param, dgraph.num_vertices + 3, gnn_dim_edge) 
    mailbox.move_to_gpu()
else:
    mailbox = None

sampler = None
if not ('no_sample' in sample_param and sample_param['no_sample']):
    sampler = TemporalSampler(dgraph, sample_param['neighbor'], sample_param['strategy'], 
                                num_snapshots=sample_param['history'], snapshot_time_window=sample_param['duration'])

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
    
    mfgs_reverse = False
    if 'deliver_to' in memory_param and memory_param['deliver_to'] == 'neighbors':
        mfgs_reverse = True

    for i, (target_nodes, ts, eid) in enumerate(get_batch(df[0])):
        time_start = time.time()
        mfgs = None
        if sampler is not None:
            if 'no_neg' in sample_param and sample_param['no_neg']:
                pos_root_end = target_nodes.shape[0] * 2 // 3
                mfgs = sampler.sample(target_nodes[:pos_root_end], ts[:pos_root_end], prop_time=sample_param['prop_time'], reverse=mfgs_reverse)
                # mfgs = sampler.sample(target_nodes[584:585], ts[584:585])
                # mfgs = sampler.sample(torch.Tensor([242]), torch.Tensor([48066.859]))
            else:
                mfgs = sampler.sample(target_nodes, ts, prop_time=sample_param['prop_time'], reverse=mfgs_reverse)
        # if identity
        mfgs_deliver_to_neighbors = None
        if gnn_param['arch'] == 'identity':
            mfgs_deliver_to_neighbors = mfgs
            mfgs = node_to_dgl_blocks(target_nodes, ts)
        
        sample_end = time.time()
        
        # move mfgs to cuda
        mfgs_to_cuda(mfgs)
        mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=False)
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0])
        
        optimizer.zero_grad()
        pred_pos, pred_neg = model(mfgs)

        loss = creterion(pred_pos, torch.ones_like(pred_pos))
        loss += creterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss) * train_param['batch_size']
        loss.backward()
        
        optimizer.step()
        
        # MailBox Update: 
        if mailbox is not None:
            mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            block = None
            if memory_param['deliver_to'] == 'neighbors':
                block = mfgs_deliver_to_neighbors[0][0]
            mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, target_nodes, ts, mem_edge_feats, block)
            mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, model.memory_updater.last_updated_ts)
        
        train_end = time.time()
        
        iteration_time += train_end - time_start
        sample_time += sample_end - time_start
        train_time += train_end - sample_end
        
        if i % 300 == 0:
            print('Iteration:{}. Train loss:{:.4f}'.format(i, loss))
            print('Iteration time:{:.2f}s; sample time:{:.2f}s; train time:{:.2f}s.'
                  .format(iteration_time / (i + 1), sample_time / (i + 1), train_time / (i + 1)))


    epoch_time_end = time.time()
    epoch_time = epoch_time_end - epoch_time_start
    
    torch.save(model.state_dict(), './ckpts/apan_test_acc_{}.pt'.format(i))
    # Validation
    print("***Start validation at epoch {}***".format(e))
    val_start = time.time()
    ap, auc = val(df[1], sampler, mailbox, model, node_feats, 
        edge_feats, creterion, mode='val', no_neg=no_neg, 
        identity=gnn_param['arch'] == 'identity', 
        deliver_to_neighbor=memory_param['deliver_to'] == 'neighbors',
        prop_time=sample_param['prop_time'])
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
    val(df[0], sampler, mailbox, model, node_feats, 
        edge_feats, creterion, mode='val', no_neg=no_neg, 
        identity=gnn_param['arch'] == 'identity', 
        deliver_to_neighbor=memory_param['deliver_to'] == 'neighbors',
        prop_time=sample_param['prop_time'])
    val(df[1], sampler, mailbox, model, node_feats, 
        edge_feats, creterion, mode='val', no_neg=no_neg, 
        identity=gnn_param['arch'] == 'identity', 
        deliver_to_neighbor=memory_param['deliver_to'] == 'neighbors',
        prop_time=sample_param['prop_time'])
ap, auc = val(df[2], sampler, mailbox, model, node_feats, 
        edge_feats, creterion, mode='val', no_neg=no_neg, 
        identity=gnn_param['arch'] == 'identity', 
        deliver_to_neighbor=memory_param['deliver_to'] == 'neighbors',
        prop_time=sample_param['prop_time'])
print('\ttest ap:{:4f}  test auc:{:4f}'.format(ap, auc))
            