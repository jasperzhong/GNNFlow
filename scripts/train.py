import argparse
import torch
import time
from dgnn.build_graph import build_dynamic_graph, load_graph, get_batch, load_feat
from dgnn.model.memory_updater import MailBox
from dgnn.model.tgn import TGN
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import prepare_input

parser = argparse.ArgumentParser()
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

# For REDDIT, not use any node or edge features now.
# The origin's dim edge is 172
gnn_dim_node = 0
gnn_dim_edge = 0

# Build Graph, block_size = 1024
node_feats, edge_feats = load_feat(None, 'REDDIT')
df = load_graph(None, 'REDDIT')
dgraph = build_dynamic_graph(df)

model = TGN(gnn_dim_node, gnn_dim_edge,
          sample_param, memory_param,
          gnn_param, train_param).cuda()

# TODO: MailBox. Check num_vertices
mailbox = MailBox(memory_param, dgraph.num_vertices, gnn_dim_edge)
mailbox.move_to_gpu()

sampler = TemporalSampler(dgraph, [10], 'recent')

creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])

for e in range(args.epoch):
    print("Epoch {}".format(e))
    total_loss = 0
    iteration_time = 0
    sample_time = 0
    train_time = 0
    
    model.train()
    mailbox.reset()
    model.memory_updater.last_updated_nid = None

    for i, (target_nodes, ts) in enumerate(get_batch(df)):
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
        mem_edge_feats = None
        block = None
        mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, target_nodes, ts, mem_edge_feats, block)
        mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, model.memory_updater.last_updated_ts)
        
        train_end = time.time()
        
        iteration_time += train_end - time_start
        sample_time += sample_end - time_start
        train_time += train_end - sample_end
        
        if i % 20 == 0:
            print('Iteration:{}. Train loss:{:.4f}'.format(i, loss))
            print('Iteration time:{:.2f}s; sample time:{:.2f}s; train time:{:.2f}s.'
                  .format(iteration_time / (i + 1), sample_time / (i + 1), train_time / (i + 1)))
            
# TODO: Validation