import argparse
import os
import random
import time
import numpy as np
import pandas as pd
from pyrsistent import inc
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from dgnn.cache.cache import Cache

import dgnn.models as models
from dgnn.config import get_default_config
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import (build_dynamic_graph, degree_based_sampling, get_project_root_dir,
                        load_dataset, load_feat, loss_based_samping, mfgs_to_cuda, get_batch,
                        node_to_dgl_blocks, RandEdgeSampler, get_pinned_buffers, prepare_input, update_degree, update_loss)
import dgnn.cache as caches

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))
cache_names = sorted(name for name in caches.__dict__
                     if not name.startswith("__")
                     and callable(caches.__dict__[name]))
print(cache_names)

datasets = ['REDDIT', 'GDELT', 'LASTFM', 'MAG', 'MOOC', 'WIKI']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default="LASTFM", help="dataset name")
parser.add_argument("--model", choices=model_names, default='TGN',
                    help="model architecture" +
                    '|'.join(model_names) +
                    '(default: tgn)')
parser.add_argument("--data", choices=datasets,
                    default='LASTFM', help="dataset:" +
                    '|'.join(datasets) + '(default: REDDIT)')
parser.add_argument("--cache", choices=cache_names,
                    default='LFUCache', help="cache:" +
                    '|'.join(cache_names) + '(default: LFUCache)')
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
                    type=int, nargs="+", default=[10, 10])
parser.add_argument("--sample-history",
                    help="the number of snapshot", type=int, default=1)
parser.add_argument("--sample-duration",
                    help="snapshot duration", type=int, default=0)
parser.add_argument("--reorder", help="", type=int, default=0)
parser.add_argument("--graph-reverse",
                    help="build undirected graph", type=bool, default=True)
parser.add_argument('--rand_edge_features', type=int, default=1,
                    help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0,
                    help='use random node featrues')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--retrain", type=int, default=1e10)
parser.add_argument("--replay_ratio", type=float, default=0.5)
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)


def val(df, rand_sampler, sampler: TemporalSampler,
        model: torch.nn.Module, cache: Cache, node_feats: torch.Tensor,
        edge_feats: torch.Tensor, creterion: torch.nn.Module, neg_samples=1,
        no_neg=False, identity=False, deliver_to_neighbors=False):
    model.eval()
    val_losses = list()
    aps = list()
    aucs_mrrs = list()

    with torch.no_grad():
        total_loss = 0

        mfgs = None
        for i, (target_nodes, ts, eid) in enumerate(get_batch(df, rand_sampler, 1000)):
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

            mfgs_to_cuda(mfgs)
            # TODO: update caceh maybe False
            if cache is not None:
                mfgs = cache.fetch_feature(mfgs, update_cache=True)
            else:
                mfgs = prepare_input(
                    mfgs, node_feats, edge_feats, combine_first=False)
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


def train(args, path_saver, df, rand_sampler, val_df, val_rand_sampler,
          sampler, model, cache, node_feats, edge_feats, creterion, optimizer, save_model, n):

    no_improvement_in_n = n
    epoch_time_sum = 0
    best_ap = -1e10
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

        model.train()

        model.mailbox_reset()
        time_start = torch.cuda.Event(enable_timing=True)
        sample_end = torch.cuda.Event(enable_timing=True)
        feature_end = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        train_end = torch.cuda.Event(enable_timing=True)
        for i, (target_nodes, ts, eid) in enumerate(get_batch(df, rand_sampler)):
            time_start.record()
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
            sample_end.record()

            mfgs_to_cuda(mfgs)
            cuda_end.record()

            mfgs = prepare_input(
                mfgs, node_feats, edge_feats, combine_first=False)
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
                                  mfgs_deliver_to_neighbors,
                                  args.deliver_to_neighbors)

            train_end.record()

            sample_time += time_start.elapsed_time(sample_end)
            cuda_time += sample_end.elapsed_time(cuda_end)
            feature_time += cuda_end.elapsed_time(feature_end)
            train_end.synchronize()
            train_time += feature_end.elapsed_time(train_end)

        epoch_time_end = time.time()
        epoch_time = epoch_time_end - epoch_time_start
        sample_time /= 1000
        cuda_time /= 1000
        feature_time /= 1000
        train_time /= 1000
        # with open("profile.txt", "a") as f:
        #     f.write("Epoch: {}\n".format(e))
        #     Epoch_time = 'Epoch time:{:.2f}s; dataloader time:{:.2f}s sample time:{:.2f}s; cuda time:{:.2f}s; feature time: {:.2f}s train time:{:.2f}s.; fetch time:{:.2f}s ; update node time:{:.2f}s; cache node ratio: {:.2f}; cache edge ratio: {:.2f}\n'.format(
        #         epoch_time, epoch_time - sample_time - feature_time - train_time - cuda_time, sample_time, cuda_time, feature_time, train_time, fetch_all_time, update_node_time, cache_node_ratio_sum / (i + 1), cache_edge_ratio_sum / (i + 1))
        #     f.write(Epoch_time)
        epoch_time_sum += epoch_time

        # Validation
        print("***Start validation at epoch {}***".format(e))
        val_start = time.time()
        ap, auc = val(val_df, val_rand_sampler, sampler, model, None, node_feats,
                      edge_feats, creterion, no_neg=args.no_neg,
                      identity=args.arch_identity,
                      deliver_to_neighbors=args.deliver_to_neighbors)
        val_end = time.time()
        val_time = val_end - val_start
        print("epoch train time: {} ; val time: {}; val ap:{:4f}; val auc:{:4f}"
              .format(epoch_time, val_time, ap, auc))
        if save_model:
            torch.save(model.state_dict(), path_saver)

        if ap <= best_ap:
            counter -= 1
        else:
            best_ap = ap
            counter = no_improvement_in_n

        if counter < 0:
            print("early stop at epoch: {}".format(e))
            break


path_saver = os.path.join(get_project_root_dir(),
                          '{}_{}_offline.pt'.format(args.model, args.data))

_, _, _, df = load_dataset(args.data)
phase1 = int(len(df) * 0.3)
phase1_train = int(phase1 * 0.9) + 1
phase1_train_df = df[:phase1_train]
phase1_val_df = df[phase1_train:phase1]
rand_sampler = RandEdgeSampler(
    phase1_train_df['src'].to_numpy(), phase1_train_df['dst'].to_numpy())
val_rand_sampler = RandEdgeSampler(
    df[:phase1]['src'].to_numpy(), df[:phase1]['dst'].to_numpy())

# use the full data to build graph
config = get_default_config(args.data)

# only use phase1 data since we need to use nodes' degree
dgraph = build_dynamic_graph(
    df[:phase1], **config,
    add_reverse=args.graph_reverse)

edge_count = dgraph.num_edges()
node_count = dgraph.num_vertices()
node_feats, edge_feats = load_feat(
    args.data, rand_de=args.rand_edge_features,
    rand_dn=args.rand_node_features,
    edge_count=edge_count, node_count=node_count)
# for test
node_feats = None
edge_feats.cuda()

gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

model = models.__dict__[args.model](
    gnn_dim_node, gnn_dim_edge, dgraph.num_vertices())
model.cuda()

args.arch_identity = args.model in ['JODIE', 'APAN']

pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(
    args.sample_neighbor, args.sample_history, args.batch_size, node_feats, edge_feats)

if not args.no_sample:
    sampler = TemporalSampler(dgraph,
                              fanouts=args.sample_neighbor,
                              strategy=args.sample_strategy,
                              num_snapshots=args.sample_history,
                              snapshot_time_window=args.sample_duration,
                              prop_time=args.prop_time,
                              reverse=args.deliver_to_neighbors,
                              seed=args.seed)

creterion = torch.nn.BCEWithLogitsLoss()
# for loss-based sample
creterion_n_reduce = torch.nn.BCEWithLogitsLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# train(args, path_saver, phase1_train_df, rand_sampler,
#       phase1_val_df, val_rand_sampler, sampler, model, None,
#       node_feats, edge_feats, creterion, optimizer, True, 5)

# # phase1 training done
# # update rand_sampler
rand_sampler.add_src_dst_list(phase1_val_df['src'].to_numpy(),
                              phase1_val_df['dst'].to_numpy())
model.load_state_dict(torch.load(path_saver))
node_degree = {}
update_degree(dgraph, df[:phase1], node_degree)

print("phase2")
with open("online_{}_ap_{}_{}.txt".format(args.model, args.retrain, args.replay_ratio), "a") as f_phase2:
    f_phase2.write("*********\n".format(args.retrain))
    f_phase2.write("retrain: {}\n".format(args.retrain))
# Phase2: incremental offline training
phase2_df = df[phase1:]
incremental_step = 1000
retrain_count = 1.0
weights = torch.tensor([1.0] * phase1)
for i, (target_nodes, ts, eid) in enumerate(get_batch(phase2_df, None, incremental_step)):
    # add to rand_sampler
    src = target_nodes[:incremental_step]
    dst = target_nodes[incremental_step:incremental_step * 2]
    # add edges here
    dgraph.add_edges(src, dst, ts[:incremental_step], args.graph_reverse)
    rand_sampler.add_src_dst_list(src, dst)
    ap, auc = val(phase2_df[i * incremental_step: (i + 1) * incremental_step],
                  rand_sampler, sampler, model, None, node_feats, edge_feats, creterion)
    print("already add {}k edges".format(i))
    print("test new edges ap: {} auc: {}".format(ap, auc))

    # save the record
    with open("online_{}_ap_{}_{}.txt".format(args.model, args.retrain, args.replay_ratio), "a") as f_phase2:
        f_phase2.write("{}\n".format(ap))

    with open("profile_offline_{}_auc.txt".format(args.model), "a") as f_phase2:
        f_phase2.write("val auc: {}\n".format(auc))

    # retrain by using previous data 50k
    if i % args.retrain == 0 and i != 0:
        # USE WEIGHTED SAMPLING
        # retrain_count += 1
        # phase2_train_df, phase2_val_df, phase2_new_data_end, weights = weighted_sample(
        #     args.replay_ratio, df, weights, phase1,
        #     i, incremental_step, args.retrain, retrain_count)
        # reconstruct the rand_sampler again(may not be necessary)

        # USE DEGREE BASED SAMPLING
        phase2_train_df, phase2_val_df, phase2_new_data_end = degree_based_sampling(
            dgraph, args.replay_ratio, df, phase1,
            i, incremental_step, args.retrain, 0.5, node_degree)

        rand_sampler = RandEdgeSampler(
            phase2_train_df['src'].to_numpy(), phase2_train_df['dst'].to_numpy())
        val_rand_sampler = RandEdgeSampler(
            df[:phase2_new_data_end]['src'].to_numpy(), df[:phase2_new_data_end]['dst'].to_numpy())

        # model = models.__dict__[args.model](
        #     gnn_dim_node, gnn_dim_edge, dgraph.num_vertices())
        # model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # dgraph has been built, no need to build again
        train(args, path_saver, phase2_train_df, rand_sampler,
              phase2_val_df, val_rand_sampler, sampler, model, None,
              node_feats, edge_feats, creterion, optimizer, False, 5)

with open("online_{}_ap_{}_{}.txt".format(args.model, args.retrain, args.replay_ratio), "a") as f_phase2:
    f_phase2.write("********\n")
    f_phase2.write("\n")
