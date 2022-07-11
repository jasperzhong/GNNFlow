import argparse
import os
import time

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import BatchSampler, SequentialSampler

import dgnn.models as models
from dgnn.dataset import DynamicGraphDataset, default_collate_ndarray
from dgnn.sampler import BatchSamplerReorder
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import (build_dynamic_graph, get_project_root_dir,
                        load_dataset, load_feat, mfgs_to_cuda,
                        node_to_dgl_blocks, prepare_input)

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=model_names, default='TGN',
                    help="model architecture" +
                    '|'.join(model_names) +
                    '(default: tgn)')
parser.add_argument("--epoch", help="training epoch", type=int, default=100)
parser.add_argument("--lr", help='learning rate', type=float, default=0.0001)
parser.add_argument("--batch-size", help="batch size", type=int, default=600)
parser.add_argument("--num-workers", help="num workers", type=int, default=0)
parser.add_argument("--dropout", help="dropout", type=float, default=0.2)
parser.add_argument(
    "--attn-dropout", help="attention dropout", type=float, default=0.2)
parser.add_argument("--deliver-to-neighbors",
                    help='deliver to neighbors', action='store_true',
                    default=False)
parser.add_argument("--use-memory", help='use memory module',
                    action='store_true', default=True)
parser.add_argument("--no-sample", help='do not need sampling',
                    action='store_true', default=False)
parser.add_argument("--prop-time", help='use prop time',
                    action='store_true', default=False)
parser.add_argument("--no-neg", help='not using neg samples in sampling',
                    action='store_true', default=False)
parser.add_argument("--sample-layer", help="sample layer", type=int, default=1)
parser.add_argument("--sample-strategy",
                    help="sample strategy", type=str, default='recent')
parser.add_argument("--sample-neighbor",
                    help="how many neighbors to sample in each layer",
                    type=int, nargs="+", default=[10])
parser.add_argument("--sample-history",
                    help="the number of snapshot", type=int, default=1)
parser.add_argument("--sample-duration",
                    help="snapshot duration", type=int, default=0)
parser.add_argument("--reorder", help="", type=int, default=0)
parser.add_argument("--graph-reverse",
                    help="build undirected graph", type=bool, default=True)

args = parser.parse_args()


def val(dataloader: torch.utils.data.DataLoader, sampler: TemporalSampler,
        model: torch.nn.Module, node_feats: torch.Tensor,
        edge_feats: torch.Tensor, creterion: torch.nn.Module, neg_samples=1,
        no_neg=False, identity=False, deliver_to_neighbors=False):
    model.eval()
    val_losses = list()
    aps = list()
    aucs_mrrs = list()

    with torch.no_grad():
        total_loss = 0

        mfgs = None
        for i, (target_nodes, ts, eid) in enumerate(dataloader):

            if sampler is not None:
                if no_neg:
                    pos_root_end = target_nodes.shape[0] * 2 // 3
                    mfgs = sampler.sample(
                        target_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    mfgs = sampler.sample(target_nodes, ts)
            # if identity
            mfgs_deliver_to_neighbors = None
            if identity:
                mfgs_deliver_to_neighbors = mfgs
                mfgs = node_to_dgl_blocks(target_nodes, ts)

            mfgs = prepare_input(
                mfgs, node_feats, edge_feats, combine_first=False)
            mfgs_to_cuda(mfgs)

            pred_pos, pred_neg = model(mfgs, neg_samples)

            total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pred_pos.size(0)),
                 torch.zeros(pred_neg.size(0))], dim=0)
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze(
                ) < pred_neg.squeeze().reshape(
                    neg_samples, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))

            aps.append(average_precision_score(y_true, y_pred))

            model.update_mem_mail(target_nodes, ts, edge_feats, eid,
                                  mfgs_deliver_to_neighbors,
                                  deliver_to_neighbors)

        val_losses.append(float(total_loss))

    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr


# Build Graph, block_size = 1024
path_saver = os.path.join(get_project_root_dir(), '{}.pt'.format(args.model))
node_feats, edge_feats = load_feat('REDDIT')
train_df, val_df, test_df, df = load_dataset('REDDIT')

train_ds = DynamicGraphDataset(train_df)
val_ds = DynamicGraphDataset(val_df)
test_ds = DynamicGraphDataset(test_df)

if args.reorder > 0:
    train_sampler = BatchSamplerReorder(
        SequentialSampler(train_ds),
        batch_size=args.batch_size, drop_last=False, num_chunks=args.reorder)
else:
    train_sampler = BatchSampler(
        SequentialSampler(train_ds),
        batch_size=args.batch_size, drop_last=False)

val_sampler = BatchSampler(
    SequentialSampler(val_ds),
    batch_size=args.batch_size, drop_last=False)
test_sampler = BatchSampler(
    SequentialSampler(test_ds),
    batch_size=args.batch_size, drop_last=False)

train_loader = torch.utils.data.DataLoader(
    train_ds, sampler=train_sampler, collate_fn=default_collate_ndarray,
    num_workers=args.num_workers)
val_loader = torch.utils.data.DataLoader(
    val_ds, sampler=val_sampler, collate_fn=default_collate_ndarray,
    num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(
    test_ds, sampler=test_sampler, collate_fn=default_collate_ndarray,
    num_workers=args.num_workers)


# use the full data to build graph
dgraph = build_dynamic_graph(df, add_reverse=args.graph_reverse)

gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

model = models.__dict__[args.model](
    gnn_dim_node, gnn_dim_edge, dgraph.num_vertices())
model.cuda()

args.arch_identity = args.model in ['JODIE', 'APAN']

# assert
assert args.sample_layer == model.gnn_layer, "sample layers must match the gnn layers"
assert args.sample_layer == len(
    args.sample_neighbor), "sample layer must match the length of sample_neighbors"

sampler = None

if not args.no_sample:
    sampler = TemporalSampler(dgraph,
                              fanouts=args.sample_neighbor,
                              strategy=args.sample_strategy,
                              num_snapshots=args.sample_history,
                              snapshot_time_window=args.sample_duration,
                              prop_time=args.prop_time,
                              reverse=args.deliver_to_neighbors)

creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

best_ap = 0
best_e = 0

epoch_time_sum = 0
for e in range(args.epoch):
    print("Epoch {}".format(e))
    epoch_time_start = time.time()
    total_loss = 0
    sample_time = 0
    feature_time = 0
    train_time = 0

    if args.reorder > 0:
        train_sampler.reset()

    # TODO: we can overwrite train():
    # a new class inherit torch.nn.Module which has self.mailbox = None.
    # if mailbox is not None. reset!
    model.train()

    model.mailbox_reset()

    for i, (target_nodes, ts, eid) in enumerate(train_loader):
        time_start = time.time()
        mfgs = None
        if sampler is not None:
            if args.no_neg:
                pos_root_end = target_nodes.shape[0] * 2 // 3
                mfgs = sampler.sample(
                    target_nodes[:pos_root_end], ts[:pos_root_end])
            else:
                mfgs = sampler.sample(target_nodes, ts)

        # if identity
        mfgs_deliver_to_neighbors = None
        if args.arch_identity:
            mfgs_deliver_to_neighbors = mfgs
            mfgs = node_to_dgl_blocks(target_nodes, ts)

        sample_end = time.time()

        # move mfgs to cuda
        mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=False)
        mfgs_to_cuda(mfgs)

        feature_end = time.time()

        optimizer.zero_grad()

        # move pre_input_mail to forward()
        pred_pos, pred_neg = model(mfgs)

        loss = creterion(pred_pos, torch.ones_like(pred_pos))
        loss += creterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss) * args.batch_size
        loss.backward()

        optimizer.step()

        # MailBox Update:
        model.update_mem_mail(target_nodes, ts, edge_feats, eid,
                              mfgs_deliver_to_neighbors,
                              args.deliver_to_neighbors)

        train_end = time.time()

        sample_time += sample_end - time_start
        feature_time += feature_end - sample_end
        train_time += train_end - feature_end
        # torch.cuda.current_stream().synchronize()

    epoch_time_end = time.time()
    epoch_time = epoch_time_end - epoch_time_start
    print('Epoch time:{:.2f}s; dataloader time:{:.2f}s sample time:{:.2f}s; feature time: {:.2f}s train time:{:.2f}s.'.format(
        epoch_time, epoch_time - sample_time - feature_time - train_time, sample_time, feature_time, train_time))

    epoch_time_sum += epoch_time

    # Validation
    print("***Start validation at epoch {}***".format(e))
    val_start = time.time()
    ap, auc = val(val_loader, sampler, model, node_feats,
                  edge_feats, creterion, no_neg=args.no_neg,
                  identity=args.arch_identity,
                  deliver_to_neighbors=args.deliver_to_neighbors)
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

# To update the memory
if model.use_mailbox():
    model.mailbox_reset()
    val(train_loader, sampler, model, node_feats,
        edge_feats, creterion, no_neg=args.no_neg,
        identity=args.arch_identity,
        deliver_to_neighbors=args.deliver_to_neighbors)
    val(val_loader, sampler, model, node_feats,
        edge_feats, creterion, no_neg=args.no_neg,
        identity=args.arch_identity,
        deliver_to_neighbors=args.deliver_to_neighbors)

ap, auc = val(test_loader, sampler, model, node_feats,
              edge_feats, creterion, no_neg=args.no_neg,
              identity=args.arch_identity,
              deliver_to_neighbors=args.deliver_to_neighbors)
print('\ttest ap:{:4f}  test auc:{:4f}'.format(ap, auc))
print('Avg epoch time: {}'.format(epoch_time_sum / args.epoch))
