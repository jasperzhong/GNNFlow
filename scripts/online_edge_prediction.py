import argparse
import logging
import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from dgnn.config import get_default_config
from dgnn.models.dgnn import DGNN
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import (
    EarlyStopMonitor, RandEdgeSampler, get_batch, prepare_input,
    build_dynamic_graph, get_pinned_buffers, get_project_root_dir, load_dataset,
    load_feat, mfgs_to_cuda)

datasets = ['REDDIT', 'GDELT', 'LASTFM', 'MAG', 'MOOC', 'WIKI']
model_names = ['TGN', 'TGAT', 'DySAT']

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=model_names, required=True,
                    help="model architecture" + '|'.join(model_names))
parser.add_argument("--data", choices=datasets, required=True,
                    help="dataset:" + '|'.join(datasets))
parser.add_argument("--epoch", help="maximum training epoch",
                    type=int, default=100)
parser.add_argument("--lr", help='learning rate', type=float, default=0.0001)
parser.add_argument("--num-workers", help="num workers for dataloaders",
                    type=int, default=0)
parser.add_argument("--num-chunks", help="number of chunks for batch sampler",
                    type=int, default=1)
parser.add_argument("--gpu", help="gpu id", type=int, default=0)
parser.add_argument("--profile", help="enable profiling", action="store_true")
parser.add_argument("--val-ratio", type=float, default=0.1,
                    help="validation ratio")
parser.add_argument("--seed", type=int, default=42)

# online training hyper-parameters
parser.add_argument("--skip-phase1", action="store_true",
                    help="skip phase 1")
parser.add_argument("--phase1-ratio", type=float, default=0.3,
                    help="ratio of total data used for phase 1 training")
parser.add_argument("--retrain-interval", type=int, default=1e10)
parser.add_argument("--replay-ratio", type=float, default=0.5)
parser.add_argument("--replay-sampling", type=str, default='random',
                    help="sampling method for replaying edges")


args = parser.parse_args()

if args.profile:
    logging.basicConfig(filename='profile.log',
                        encoding='utf-8', level=logging.DEBUG)

logging.basicConfig(level=logging.INFO)
logging.info(args)

checkpoint_path = os.path.join(get_project_root_dir(),
                               '{}.pt'.format(args.model))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)


def evaluate(data, neg_sampler, batch_size, sampler, model, criterion,
             node_feats, edge_feats, device):
    model.eval()
    val_losses = list()
    aps = list()
    aucs_mrrs = list()

    with torch.no_grad():
        total_loss = 0
        for target_nodes, ts, eid in get_batch(data, batch_size, neg_sampler):
            mfgs = sampler.sample(target_nodes, ts)
            mfgs_to_cuda(mfgs, device)
            mfgs = prepare_input(mfgs, node_feats, edge_feats)
            pred_pos, pred_neg = model(
                mfgs, eid=eid, edge_feats=edge_feats)
            total_loss += criterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pred_pos.size(0)),
                 torch.zeros(pred_neg.size(0))], dim=0)
            aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            aps.append(average_precision_score(y_true, y_pred))

        val_losses.append(float(total_loss))

    ap = float(torch.tensor(aps).mean())
    auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr


def train(train_data, val_data, batch_size, train_neg_sampler, val_neg_sampler,
          sampler, model, optimizer, criterion, node_feats, edge_feats, device):

    best_ap = 0
    best_e = 0
    epoch_time_sum = 0
    early_stopper = EarlyStopMonitor()
    logging.info('Start training...')
    for e in range(args.epoch):
        model.train()
        total_loss = 0

        epoch_time_start = time.time()

        model.reset()
        for (target_nodes, ts, eid) in get_batch(train_data, batch_size, train_neg_sampler):
            # Sample
            mfgs = sampler.sample(target_nodes, ts)
            mfgs_to_cuda(mfgs, device)

            mfgs = prepare_input(
                mfgs, node_feats, edge_feats)

            # Train
            optimizer.zero_grad()
            pred_pos, pred_neg = model(
                mfgs, eid=eid, edge_feats=edge_feats)

            loss = criterion(pred_pos, torch.ones_like(pred_pos))
            loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss) * len(target_nodes)
            loss.backward()
            optimizer.step()

        epoch_time_end = time.time()
        epoch_time = epoch_time_end - epoch_time_start
        epoch_time_sum += epoch_time

        # Validation
        logging.info("***Start validation at epoch {}***".format(e))
        val_start = time.time()
        val_ap, val_auc = evaluate(
            val_data, val_neg_sampler, batch_size, sampler, model, criterion,
            node_feats, edge_feats, device)
        val_end = time.time()
        val_time = val_end - val_start
        logging.info("epoch train time: {} ; val time: {}; val ap:{:4f}; val auc:{:4f}"
                     .format(epoch_time, val_time, val_ap, val_auc))

        if e > 1 and val_ap > best_ap:
            best_e = e
            best_ap = val_ap
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(
                "Best val AP: {:.4f} & val AUC: {:.4f}".format(val_ap, val_auc))

        if early_stopper.early_stop_check(val_ap):
            logging.info("Early stop at epoch {}".format(e))
            break

    logging.info('Avg epoch time: {}'.format(epoch_time_sum / args.epoch))
    return best_e


def weighted_sample(replay_ratio, df, weights, phase1,
                    i, incremental_step, retrain_interval, retrain_count):

    weights = torch.cat((weights, torch.tensor(
        [retrain_count] * (retrain_interval * incremental_step))))
    phase2_new_data_start = phase1 + incremental_step * (i - retrain_interval)
    phase2_new_data_end = phase1 + incremental_step * i
    new_data_index = torch.arange(
        phase2_new_data_start, phase2_new_data_end)
    # first fetch replay samples in old data
    # new data will all be selected to the replay samples
    if replay_ratio != 0:
        num_replay = int(replay_ratio * phase2_new_data_start)
        index_select = torch.multinomial(weights[:phase2_new_data_start],
                                         num_replay).sort().values
        all_index = torch.cat((index_select, new_data_index))
    else:
        all_index = new_data_index
    train_length = int(len(all_index) * 0.9) + 1
    train_index = all_index[:train_length]
    val_index = all_index[train_length:]
    phase2_train_df = df.iloc[train_index.numpy()]
    phase2_val_df = df.iloc[val_index.numpy()]

    return phase2_train_df, phase2_val_df, phase2_new_data_end, weights


def main():
    model_config, data_config = get_default_config(args.model, args.data)

    _, _, _, full_data = load_dataset(args.data)
    phase1_end_idx = int(len(full_data) * args.phase1_ratio)
    phase1_df = full_data.iloc[:phase1_end_idx]
    phase2_df = full_data.iloc[phase1_end_idx:]

    node_feats, edge_feats = load_feat(args.data)
    device = torch.device('cuda:{}'.format(args.gpu))
    batch_size = data_config['batch_size']

    dgraph = build_dynamic_graph(phase1_df, **data_config)
    num_nodes = dgraph.num_vertices()
    num_edges = dgraph.num_edges()

    dim_node = 0 if node_feats is None else node_feats.shape[1]
    dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

    model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                 memory_device=device)
   # Phase 1
    if not args.skip_phase1:
        logging.info("Phase 1 training...")
        phase1_train_end_idx = int(len(phase1_df) * (1-args.val_ratio))
        phase1_train_df = phase1_df.iloc[:phase1_train_end_idx]
        phase1_val_df = phase1_df.iloc[phase1_train_end_idx:]

        train_neg_sampler = RandEdgeSampler(
            phase1_train_df['src'].to_numpy(),
            phase1_train_df['dst'].to_numpy())
        val_neg_sampler = RandEdgeSampler(
            phase1_df['src'].to_numpy(),
            phase1_df['dst'].to_numpy())

        model.to(device)

        sampler = TemporalSampler(dgraph, **model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        train(phase1_train_df, phase1_val_df, batch_size,
              train_neg_sampler, val_neg_sampler, sampler, model,
              optimizer, criterion, node_feats, edge_feats, device)
    else:
        logging.info("Skip phase 1 training...")

    # Phase 2
    logging.info("Phase 2 online training...")
    train_neg_sampler = RandEdgeSampler(
        phase1_df['src'].to_numpy(),
        phase1_df['dst'].to_numpy())
    val_neg_sampler = RandEdgeSampler(
        phase1_df['src'].to_numpy(),
        phase1_df['dst'].to_numpy())

    model.load_state_dict(torch.load(checkpoint_path))
    num_retrain = len(phase2_df) // args.retrain_interval
    incremental_block = np.array_split(phase2_df, num_retrain)





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

# use all the edges to build graph
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
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train(args, path_saver, phase1_train_df, rand_sampler,
      phase1_val_df, val_rand_sampler, sampler, model, None,
      node_feats, edge_feats, creterion, optimizer, True, 5)

# phase1 training done
# update rand_sampler
rand_sampler.add_src_dst_list(phase1_val_df['src'].to_numpy(),
                              phase1_val_df['dst'].to_numpy())
model.load_state_dict(torch.load(path_saver))

print("phase2")
with open("profile_offline_{}_ap.txt".format(args.model), "a") as f_phase2:
    f_phase2.write("*********\n".format(args.retrain))
    f_phase2.write("retrain: {}\n".format(args.retrain))
# Phase2: incremental offline training
phase2_df = df[phase1:]
incremental_step = 1000
retrain_count = 1.0
weights = torch.tensor([1.0] * phase1)
for i, (target_nodes, ts, eid) in enumerate(
        get_batch(phase2_df, None, incremental_step)):
    # add to rand_sampler
    src = target_nodes[:incremental_step]
    dst = target_nodes[incremental_step:incremental_step * 2]
    rand_sampler.add_src_dst_list(src, dst)
    # elimnate add edges because all edges are added at first
    ap, auc = val(
        phase2_df[i * incremental_step: (i + 1) * incremental_step],
        rand_sampler, sampler, model, None, node_feats, edge_feats, creterion)
    print("already add {}k edges".format(i))
    print("test new edges ap: {} auc: {}".format(ap, auc))

    # save the record
    with open("profile_offline_{}_ap.txt".format(args.model), "a") as f_phase2:
        f_phase2.write("val ap: {}\n".format(ap))

    with open("profile_offline_{}_auc.txt".format(args.model), "a") as f_phase2:
        f_phase2.write("val auc: {}\n".format(auc))

    # retrain by using previous data 50k
    if i % args.retrain == 0 and i != 0:
        retrain_count += 1
        phase2_train_df, phase2_val_df, phase2_new_data_end, weights = weighted_sample(
            args.replay_ratio, df, weights, phase1,
            i, incremental_step, args.retrain, retrain_count)
        # reconstruct the rand_sampler again(may not be necessary)
        rand_sampler = RandEdgeSampler(
            phase2_train_df['src'].to_numpy(),
            phase2_train_df['dst'].to_numpy())
        val_rand_sampler = RandEdgeSampler(
            df[: phase2_new_data_end]['src'].to_numpy(),
            df[: phase2_new_data_end]['dst'].to_numpy())

        # dgraph has been built, no need to build again
        train(args, path_saver, phase2_train_df, rand_sampler,
              phase2_val_df, val_rand_sampler, sampler, model, None,
              node_feats, edge_feats, creterion, optimizer, False, 5)

with open("profile_offline_{}_ap.txt".format(args.model), "a") as f_phase2:
    f_phase2.write("********\n")
    f_phase2.write("\n")

with open("profile_offline_{}_auc.txt".format(args.model), "a") as f_phase2:
    f_phase2.write("********\n")
    f_phase2.write("\n")

if __name__ == '__main__':
    if args.model == 'TGN':
        raise NotImplementedError("TGN is not supported yet")
    main()
