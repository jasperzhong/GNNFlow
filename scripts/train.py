import argparse
from random import sample
import torch
import torchvision
import time
import dgnn.models as models
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import prepare_input, mfgs_to_cuda, node_to_dgl_blocks, build_dynamic_graph, load_dataset, load_feat, get_batch
from scripts.validation import val

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=model_names, default='apan',
                    help="model architecture" + 
                    '|'.join(model_names) + 
                    '(default: tgn)')
parser.add_argument("--epoch", help="training epoch", type=int, default=100)
parser.add_argument("--lr", help='learning rate', type=float, default=0.0001)
parser.add_argument("--batch-size", help="batch size", type=int, default=600)
parser.add_argument("--dropout", help="dropout", type=float, default=0.2)
parser.add_argument("--attn-dropout", help="attention dropout", type=float, default=0.2)
parser.add_argument("--deliver-to-neighbors", help='deliver to neighbors')
parser.add_argument("--use-memory", help='use memory module')
parser.add_argument("--no-sample", help='do not need sampling')
parser.add_argument("--prop_time", help='use prop time')
parser.add_argument("--no_neg", help='not using neg samples in sampling')
parser.add_argument("--sample-layer", help="sample layer", type=int, default=1)
parser.add_argument("--sample-strategy", help="sample strategy", type=str, default='recent')
parser.add_argument("--sample-neighbor", help="how many neighbors to sample in each layer", type=list, default=[10])
parser.add_argument("--sample-history", help="the number of snapshot", type=int, default=1)
parser.add_argument("--sample-duration", help="snapshot duration", type=int, default=0)

args = parser.parse_args()

# TODO: Use args

# TGN
sample_param = {
            'layer': 1,
            'neighbor': [10],
            'strategy': 'uniform',
            'prop_time': False,
            'history': 1,
            'duration': 0,
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

# Build Graph, block_size = 1024
path_saver = '../models/{}.pt'.format(args.model)
path_saver_2 = '../models/apan_best.pt'
node_feats, edge_feats = load_feat(None, 'REDDIT')
df = load_dataset(None, 'REDDIT')
# df[0] consist train part of the data
# df[3] is the full graph
# use the full data to build graph
dgraph = build_dynamic_graph(df[3])

gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

model = models.__dict__[args.model](gnn_dim_node, gnn_dim_edge)
model.cuda()

args.arch_identity = args.model in ['jodie', 'apan']

# assert
assert args.sample_layer == model.gnn_layer , "sample layers must match the gnn layers"
assert args.sample_layer == len(args.sample_neighbor), "sample layer must match the length of sample_neighbors"


# TODO: MailBox. Check num_vertices
if args.use_memory:
    mailbox = MailBox(memory_param, dgraph.num_vertices + 3, gnn_dim_edge) 
    mailbox.move_to_gpu()
else:
    mailbox = None

sampler = None

if not args.no_sample:
    sampler = TemporalSampler(dgraph, args.sample_neighbor, args.sample_strategy, 
                                num_snapshots=args.sample_history, snapshot_time_window=args.sample_duration, reverse=args.deliver_to_neighbors)

creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

best_ap = 0
best_e = 0

for e in range(args.epoch):
    print("Epoch {}".format(e))
    epoch_time_start = time.time()
    total_loss = 0
    iteration_time = 0
    sample_time = 0
    train_time = 0
    
    # TODO: we can overwrite train(): 
    # a new class inherit torch.nn.Module which has self.mailbox = None.
    # if mailbox is not None. reset!
    model.train()

    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None

    for i, (target_nodes, ts, eid) in enumerate(get_batch(df[0])):
        time_start = time.time()
        mfgs = None
        if sampler is not None:
            if args.no_neg:
                pos_root_end = target_nodes.shape[0] * 2 // 3
                mfgs = sampler.sample(target_nodes[:pos_root_end], ts[:pos_root_end])
            else:
                mfgs = sampler.sample(target_nodes, ts)
        # if identity
        mfgs_deliver_to_neighbors = None
        if args.arch_identity:
            mfgs_deliver_to_neighbors = mfgs
            mfgs = node_to_dgl_blocks(target_nodes, ts)
        
        sample_end = time.time()
        
        # move mfgs to cuda
        mfgs_to_cuda(mfgs)
        mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=False)

        # TODO: move this to forward()
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0])
        
        optimizer.zero_grad()
        pred_pos, pred_neg = model(mfgs)

        loss = creterion(pred_pos, torch.ones_like(pred_pos))
        loss += creterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss) * args.batch_size
        loss.backward()
        
        optimizer.step()
        
        # MailBox Update: 
        # TODO: use a new function in model
        if mailbox is not None:
            mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            block = None
            if args.deliver_to_neighbors:
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
        edge_feats, creterion, no_neg=args.no_neg, 
        identity=args.arch_identity, 
        deliver_to_neighbor=args.deliver_to_neighbors)
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
        edge_feats, creterion, no_neg=args.no_neg, 
        identity=args.arch_identity, 
        deliver_to_neighbor=args.deliver_to_neighbors)
    val(df[1], sampler, mailbox, model, node_feats, 
        edge_feats, creterion, no_neg=args.no_neg, 
        identity=args.arch_identity, 
        deliver_to_neighbor=args.deliver_to_neighbors)
ap, auc = val(df[2], sampler, mailbox, model, node_feats, 
        edge_feats, creterion, no_neg=args.no_neg, 
        identity=args.arch_identity, 
        deliver_to_neighbor=args.deliver_to_neighbors)
print('\ttest ap:{:4f}  test auc:{:4f}'.format(ap, auc))
            