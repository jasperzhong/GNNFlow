import argparse
import os
import random
import time

import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import BatchSampler, SequentialSampler

import dgnn.cache as caches
import dgnn.models as models
from dgnn.cache.cache import Cache
from dgnn.config import get_default_config
from dgnn.dataset import DynamicGraphDataset, default_collate_ndarray
from dgnn.sampler import BatchSamplerReorder
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import (RandEdgeSampler, build_dynamic_graph,
                        get_pinned_buffers, get_project_root_dir, load_dataset,
                        load_feat, mfgs_to_cuda)

datasets = ['REDDIT', 'GDELT', 'LASTFM', 'MAG', 'MOOC', 'WIKI']
model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))
cache_names = sorted(name for name in caches.__dict__
                     if not name.startswith("__")
                     and callable(caches.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=model_names, default='TGN',
                    help="model architecture" +
                    '|'.join(model_names) +
                    '(default: tgn)')
parser.add_argument("--data", choices=datasets,
                    default='REDDIT', help="dataset:" +
                    '|'.join(datasets) + '(default: REDDIT)')
parser.add_argument("--cache", choices=cache_names,
                    default='LFUCache', help="cache:" +
                    '|'.join(cache_names) + '(default: LFUCache)')
parser.add_argument("--epoch", help="training epoch", type=int, default=100)
parser.add_argument("--lr", help='learning rate', type=float, default=0.0001)
parser.add_argument("--batch-size", help="batch size", type=int, default=600)
parser.add_argument("--num-workers", help="num workers for dataloaders",
                    type=int, default=0)
parser.add_argument("--reorder", help="whether to use a different start point every epoch",
                    type=int, default=0)
parser.add_argument("--seed", type=int, default=42)


parser.add_argument("--cache", choices=cache_names,
                    default='LFUCache', help="cache strategy")
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)


def val(dataloader: torch.utils.data.DataLoader, sampler: TemporalSampler,
        model: torch.nn.Module, cache: Cache, node_feats: torch.Tensor,
        edge_feats: torch.Tensor, creterion: torch.nn.Module, neg_samples=1):
    model.eval()
    val_losses = list()
    aps = list()
    aucs_mrrs = list()

    with torch.no_grad():
        total_loss = 0

        mfgs = None
        for i, (target_nodes, ts, eid) in enumerate(dataloader):

            mfgs = sampler.sample(target_nodes, ts)
            mfgs_to_cuda(mfgs)

            # TODO: update caceh maybe False
            mfgs = cache.fetch_feature(mfgs, update_cache=True)

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

train_df, val_df, test_df, df = load_dataset(args.data)
train_rand_sampler = RandEdgeSampler(
    train_df['src'].to_numpy(), train_df['dst'].to_numpy())
val_rand_sampler = RandEdgeSampler(
    val_df['src'].to_numpy(), val_df['dst'].to_numpy())
test_rand_sampler = RandEdgeSampler(
    test_df['src'].to_numpy(), test_df['dst'].to_numpy())

train_ds = DynamicGraphDataset(train_df, train_rand_sampler)
val_ds = DynamicGraphDataset(val_df, val_rand_sampler)
test_ds = DynamicGraphDataset(test_df, test_rand_sampler)


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
config = get_default_config(args.data)
dgraph = build_dynamic_graph(
    df, **config,
    add_reverse=args.graph_reverse)

edge_count = dgraph.num_edges()
node_count = dgraph.num_vertices()
node_feats, edge_feats = load_feat(
    args.data, rand_de=args.rand_edge_features,
    rand_dn=args.rand_node_features,
    edge_count=edge_count, node_count=node_count)

# for test
node_feats = None

gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

model = models.__dict__[args.model](
    gnn_dim_node, gnn_dim_edge, dgraph.num_vertices())
model.cuda()


pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(
    args.sample_neighbor, args.sample_history, args.batch_size, node_feats, edge_feats)

# Cache
cache = caches.__dict__[args.cache](0.2, dgraph.num_vertices(),
                                    edge_count,
                                    node_feats, edge_feats, 'cuda:0',
                                    pinned_nfeat_buffs, pinned_efeat_buffs)

# assert
assert args.sample_layer == model.gnn_layer, "sample layers must match the gnn layers"
assert args.sample_layer == len(
    args.sample_neighbor), "sample layer must match the length of sample_neighbors"

sampler = TemporalSampler(dgraph,
                          fanouts=args.sample_neighbor,
                          strategy=args.sample_strategy,
                          num_snapshots=args.sample_history,
                          snapshot_time_window=args.sample_duration,
                          prop_time=args.prop_time,
                          seed=args.seed)

# only gnnlab static need to pass param
cache.init_cache()
# cache.init_cache(sampler, train_df, 2)

creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

best_ap = 0
best_e = 0

epoch_time_sum = 0
with open("profile.txt", "a") as f:
    f.write("\n")
    f.write("Data: {}\n".format(args.data))
    f.write("Cache: {}\n".format(args.cache))
    f.write("strategy: {}\n".format(args.sample_strategy))

for e in range(args.epoch):
    print("Epoch {}".format(e))
    epoch_time_start = time.time()
    total_loss = 0
    sample_time = 0
    feature_time = 0
    train_time = 0
    fetch_all_time = 0
    update_node_time = 0
    update_edge_time = 0
    cache_edge_ratio_sum = 0
    cache_node_ratio_sum = 0
    cuda_time = 0

    if args.reorder > 0:
        train_sampler.reset()

    # init cache every epoch
    # TODO: maybe a better way to init cache in every epoch
    # only edge feature need to re-init between different epochs
    # cache.init_cache(sampler, train_df, 2)

    # TODO: we can overwrite train():
    # a new class inherit torch.nn.Module which has self.mailbox = None.
    # if mailbox is not None. reset!
    model.train()

    model.mailbox_reset()
    time_start = torch.cuda.Event(enable_timing=True)
    sample_end = torch.cuda.Event(enable_timing=True)
    feature_end = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    train_end = torch.cuda.Event(enable_timing=True)
    for i, (target_nodes, ts, eid) in enumerate(train_loader):
        time_start.record()
        mfgs = sampler.sample(target_nodes, ts)

        # if identity
        sample_end.record()

        # mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=False)
        mfgs_to_cuda(mfgs)
        cuda_end.record()

        mfgs = cache.fetch_feature(mfgs)
        feature_end.record()
        # Train
        optimizer.zero_grad()

        # move pre_input_mail to forward()
        neg_sample = 0 if args.no_neg else 1
        pred_pos, pred_neg = model(mfgs, neg_samples=neg_sample)

        loss = creterion(pred_pos, torch.ones_like(pred_pos))
        loss += creterion(pred_neg, torch.zeros_like(pred_neg))
        total_loss += float(loss) * args.batch_size
        loss.backward()

        optimizer.step()

        # MailBox Update:
        model.update_mem_mail(target_nodes, ts, edge_feats, eid,
                              mfgs_deliver_to_neighbors)

        train_end.record()

        sample_time += time_start.elapsed_time(sample_end)
        cuda_time += sample_end.elapsed_time(cuda_end)
        feature_time += cuda_end.elapsed_time(feature_end)
        train_end.synchronize()
        train_time += feature_end.elapsed_time(train_end)
        fetch_all_time += cache.fetch_time
        update_node_time += cache.update_node_time
        update_edge_time += cache.update_edge_time
        cache_edge_ratio_sum += cache.cache_edge_ratio
        cache_node_ratio_sum += cache.cache_node_ratio

    epoch_time_end = time.time()
    epoch_time = epoch_time_end - epoch_time_start
    sample_time /= 1000
    cuda_time /= 1000
    feature_time /= 1000
    train_time /= 1000
    with open("profile.txt", "a") as f:
        f.write("Epoch: {}\n".format(e))
        Epoch_time = 'Epoch time:{:.2f}s; dataloader time:{:.2f}s sample time:{:.2f}s; cuda time:{:.2f}s; feature time: {:.2f}s train time:{:.2f}s.; fetch time:{:.2f}s ; update node time:{:.2f}s; cache node ratio: {:.2f}; cache edge ratio: {:.2f}\n'.format(
            epoch_time, epoch_time - sample_time - feature_time - train_time - cuda_time, sample_time, cuda_time, feature_time, train_time, fetch_all_time, update_node_time, cache_node_ratio_sum / (i + 1), cache_edge_ratio_sum / (i + 1))
        f.write(Epoch_time)
    epoch_time_sum += epoch_time

    # Validation
    print("***Start validation at epoch {}***".format(e))
    val_start = time.time()
    ap, auc = val(val_loader, sampler, model, cache, node_feats,
                  edge_feats, creterion, no_neg=args.no_neg)
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
    val(train_loader, sampler, model, cache, node_feats,
        edge_feats, creterion, no_neg=args.no_neg)
    val(val_loader, sampler, model, cache, node_feats,
        edge_feats, creterion, no_neg=args.no_neg)

ap, auc = val(test_loader, sampler, model, cache, node_feats,
              edge_feats, creterion, no_neg=args.no_neg)
print('\ttest ap:{:4f}  test auc:{:4f}'.format(ap, auc))
print('Avg epoch time: {}'.format(epoch_time_sum / args.epoch))
print("*********************")
print("*********************")
