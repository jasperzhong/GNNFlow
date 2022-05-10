import argparse
import torch
import time
from dgnn.build_graph import build_dynamic_graph, load_graph, get_batch, load_feat
from dgnn.model.memory_updater import MailBox
from dgnn.model.tgn import TGN
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import prepare_input
from scripts.validation import val

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model name", type=str, default='tgn')
parser.add_argument("--epoch", help="training epoch", type=int, default=100)
args = parser.parse_args()

# TODO: Use args
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

# Build Graph, block_size = 1024
path_saver = '../models/{}.pt'.format(args.model)
path_saver_2 = '../models/tgn_best.pt'
node_feats, edge_feats = load_feat(None, 'REDDIT')
df = load_graph(None, 'REDDIT')
# df[0] consist train part of the data
dgraph = build_dynamic_graph(df[0])
val_graph = build_dynamic_graph(df[1])
test_graph = build_dynamic_graph(df[2])

gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

model = TGN(gnn_dim_node, gnn_dim_edge,
          sample_param, memory_param,
          gnn_param, train_param).cuda()

torch.save(model.state_dict(), path_saver)

# TODO: MailBox. Check num_vertices
mailbox = MailBox(memory_param, dgraph.num_vertices + 3, gnn_dim_edge)
mailbox.move_to_gpu()

# this sampler only sample train part of the data
sampler = TemporalSampler(dgraph, [10], 'recent')
val_sampler = TemporalSampler(val_graph, [10], 'recent')
test_sampler = TemporalSampler(test_graph, [10], 'recent')

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
    mailbox.reset()
    model.memory_updater.last_updated_nid = None

    for i, (target_nodes, ts, eid) in enumerate(get_batch(df[0])):
        time_start = time.time()
        # get mfgs from sampler
        mfgs = sampler.sample(target_nodes, ts)
        
        sample_end = time.time()
        
        # TODO: where to put features? cuda or cpu. TGL in cuda
        mfgs[0][0] = mfgs[0][0].to('cuda:0')
        mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=False)
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
        mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, target_nodes, ts, mem_edge_feats, block)
        mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, model.memory_updater.last_updated_ts)
        
        train_end = time.time()
        
        iteration_time += train_end - time_start
        sample_time += sample_end - time_start
        train_time += train_end - sample_end
        
        if i % 100 == 0:
            print('Iteration:{}. Train loss:{:.4f}'.format(i, loss))
            print('Iteration time:{:.2f}s; sample time:{:.2f}s; train time:{:.2f}s.'
                  .format(iteration_time / (i + 1), sample_time / (i + 1), train_time / (i + 1)))

    epoch_time_end = time.time()
    epoch_time = epoch_time_end - epoch_time_start
    
    # Validation
    print("***Start validation at epoch {}***".format(e))
    val_start = time.time()
    ap, auc = val(df[1], val_sampler, mailbox, model, node_feats, 
        edge_feats, creterion, mode='val')
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
        edge_feats, creterion, mode='train')
    val(df[1], val_sampler, mailbox, model, node_feats, 
        edge_feats, creterion, mode='val')
ap, auc = val(df[2], test_sampler, mailbox, model, node_feats, 
        edge_feats, creterion, mode='test')
print('\ttest ap:{:4f}  test auc:{:4f}'.format(ap, auc))
            