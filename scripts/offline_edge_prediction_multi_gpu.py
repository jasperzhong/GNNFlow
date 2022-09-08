import argparse
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed
import torch.nn.parallel
import torch.utils.data
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import BatchSampler, SequentialSampler

import dgnn.cache as caches
from dgnn.config import get_default_config
from dgnn.data import (DistributedBatchSampler, EdgePredictionDataset,
                       default_collate_ndarray)
from dgnn.models.dgnn import DGNN
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import (EarlyStopMonitor, RandEdgeSampler, build_dynamic_graph,
                        prepare_input, get_project_root_dir, load_dataset,
                        load_feat, mfgs_to_cuda)

datasets = ['REDDIT', 'GDELT', 'LASTFM', 'MAG', 'MOOC', 'WIKI']
model_names = ['TGN', 'TGAT', 'DySAT']
cache_names = sorted(name for name in caches.__dict__
                     if not name.startswith("__")
                     and callable(caches.__dict__[name]))

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
parser.add_argument("--profile", help="enable profiling", action="store_true")
parser.add_argument("--seed", type=int, default=42)

# optimization
parser.add_argument("--cache", choices=cache_names, help="feature cache:" +
                    '|'.join(cache_names))
parser.add_argument("--cache-ratio", type=float, default=0,
                    help="cache ratio for feature cache")
args = parser.parse_args()

if args.profile:
    logging.basicConfig(filename='profile.log',
                        encoding='utf-8', level=logging.DEBUG)

logging.basicConfig(level=logging.DEBUG)
logging.info(args)

checkpoint_path = os.path.join(get_project_root_dir(),
                               '{}.pt'.format(args.model))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)


def evaluate(dataloader, sampler, model, criterion, node_feats, edge_feats, device):
    model.eval()
    val_losses = list()
    aps = list()
    aucs_mrrs = list()

    with torch.no_grad():
        total_loss = 0
        for target_nodes, ts, eid in dataloader:
            mfgs = sampler.sample(target_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats)
            mfgs_to_cuda(mfgs, device)
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


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl')
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    model_config, data_config = get_default_config(args.model, args.data)
    data_config["mem_resources_type"] = "shared"

    train_data, val_data, test_data, full_data = load_dataset(args.data)
    train_rand_sampler = RandEdgeSampler(
        train_data['src'].values, train_data['dst'].values)
    val_rand_sampler = RandEdgeSampler(
        full_data['src'].values, full_data['dst'].values)
    test_rand_sampler = RandEdgeSampler(
        full_data['src'].values, full_data['dst'].values)

    train_ds = EdgePredictionDataset(train_data, train_rand_sampler)
    val_ds = EdgePredictionDataset(val_data, val_rand_sampler)
    test_ds = EdgePredictionDataset(test_data, test_rand_sampler)

    batch_size = data_config['batch_size']
    args.lr = args.lr * math.sqrt(world_size)
    logging.info("batch size: {}, lr: {}".format(batch_size, args.lr))
    train_sampler = DistributedBatchSampler(
        SequentialSampler(train_ds), batch_size=batch_size,
        drop_last=False, rank=rank, world_size=world_size)
    val_sampler = DistributedBatchSampler(
        SequentialSampler(val_ds),
        batch_size=batch_size, drop_last=False, rank=rank, world_size=world_size)
    test_sampler = BatchSampler(
        SequentialSampler(test_ds),
        batch_size=batch_size, drop_last=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, sampler=train_sampler,
        collate_fn=default_collate_ndarray, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_ds, sampler=val_sampler,
        collate_fn=default_collate_ndarray, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_ds, sampler=test_sampler,
        collate_fn=default_collate_ndarray, num_workers=args.num_workers)

    dgraph = build_dynamic_graph(
        full_data, **data_config, device=local_rank)

    num_nodes = dgraph.num_vertices()
    num_edges = dgraph.num_edges()
    node_feats, edge_feats = load_feat(
        args.data, shared_memory=True, local_rank=local_rank, local_world_size=local_world_size)

    dim_node = 0 if node_feats is None else node_feats.shape[1]
    dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

    device = torch.device('cuda:{}'.format(local_rank))
    model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                 memory_device=device)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank])
    logging.debug("device: {}".format(device))
    sampler = TemporalSampler(dgraph, **model_config)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_e = train(train_loader, val_loader, train_sampler, sampler,
                   model, optimizer, criterion, node_feats, edge_feats, device)

    if rank == 0:
        logging.info('Loading model at epoch {}...'.format(best_e))
        model.load_state_dict(torch.load(checkpoint_path))

        # To update the memory
        # if model.has_memory():
        #     model.reset()
        #     evaluate(train_loader, sampler, model, criterion, cache, device)
        #     evaluate(val_loader, sampler, model, criterion, cache, device)

        ap, auc = evaluate(test_loader, sampler, model,
                           criterion, node_feats, edge_feats, device)
        logging.info('Test ap:{:4f}  test auc:{:4f}'.format(ap, auc))


def train(train_loader, val_loader, train_sampler, sampler, model,
          optimizer, criterion, node_feats, edge_feats, device):

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    best_ap = 0
    best_e = 0
    epoch_time_sum = 0
    early_stopper = EarlyStopMonitor()
    logging.info('Start training...')
    for e in range(args.epoch):
        model.train()
        total_loss = 0

        epoch_time_start = time.time()
        model.train()

        for i, (target_nodes, ts, eid) in enumerate(train_loader):
            # Sample
            mfgs = sampler.sample(target_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats)
            mfgs_to_cuda(mfgs, device)

            # Train
            optimizer.zero_grad()
            pred_pos, pred_neg = model(
                mfgs, eid=eid, edge_feats=edge_feats)

            loss = criterion(pred_pos, torch.ones_like(pred_pos))
            loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss) * len(target_nodes)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0 and rank == 0:
                logging.info('Epoch {:d}/{:d} | Iter {:d}/{:d} | Throughput {:.2f} samples/s | Loss {:.4f}'.format(
                    e + 1, args.epoch, i + 1, int(len(train_loader)/world_size), (i+1) * len(target_nodes) * world_size / (time.time() - epoch_time_start), total_loss / (i + 1) / world_size))

        epoch_time_end = time.time()
        epoch_time = epoch_time_end - epoch_time_start
        epoch_time_sum += epoch_time

        # Validation
        val_start = time.time()
        val_ap, val_auc = evaluate(
            val_loader, sampler, model, criterion, node_feats, edge_feats, device)

        val_res = torch.tensor([val_ap, val_auc]).to(device)
        torch.distributed.all_reduce(val_res)
        val_res /= world_size

        val_end = time.time()
        val_time = val_end - val_start
        if rank == 0:
            logging.info("epoch train time: {} ; val time: {}; val ap:{:4f}; val auc:{:4f}"
                         .format(epoch_time, val_time, val_res[0], val_res[1]))

        if rank == 0 and e > 1 and val_ap > best_ap:
            best_e = e
            best_ap = val_ap
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(
                "Best val AP: {:.4f} & val AUC: {:.4f}".format(val_ap, val_auc))

        if rank == 0 and early_stopper.early_stop_check(val_ap):
            logging.info("Early stop at epoch {}".format(e))
            break

    if rank == 0:
        logging.info('Avg epoch time: {}'.format(epoch_time_sum / args.epoch))
    return best_e


if __name__ == '__main__':
    if args.model == "TGN":
        raise NotImplementedError("TGN is not supported yet")

    main()
