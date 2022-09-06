import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import BatchSampler, SequentialSampler

import dgnn.cache as caches
from dgnn.config import get_default_config
from dgnn.data import (EdgePredictionDataset, RandomStartBatchSampler,
                       default_collate_ndarray)
from dgnn.models.dgnn import DGNN
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import (RandEdgeSampler, EarlyStopMonitor, build_dynamic_graph,
                        get_pinned_buffers, get_project_root_dir, load_dataset,
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
parser.add_argument("--gpu", help="gpu id", type=int, default=0)
parser.add_argument("--profile", help="enable profiling", action="store_true")
parser.add_argument("--seed", type=int, default=42)

# optimization
parser.add_argument("--cache", choices=cache_names,
                    default='LFUCache', help="feature cache:" +
                    '|'.join(cache_names) + '(default: LFUCache)')
parser.add_argument("--cache-ratio", type=float, default=0.2,
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


def evaluate(dataloader, sampler, model, criterion, cache, device):
    model.eval()
    val_losses = list()
    aps = list()
    aucs_mrrs = list()

    with torch.no_grad():
        total_loss = 0
        for target_nodes, ts, eid in dataloader:
            mfgs = sampler.sample(target_nodes, ts)
            mfgs_to_cuda(mfgs, device)
            mfgs = cache.fetch_feature(mfgs, update_cache=False)
            pred_pos, pred_neg = model(
                mfgs, eid=eid, edge_feats=cache.edge_features)
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
    model_config, data_config = get_default_config(args.model, args.data)

    train_data, val_data, test_data, full_data = load_dataset(args.data)
    train_rand_sampler = RandEdgeSampler(
        train_data['src'].values, train_data['dst'].values)
    val_rand_sampler = RandEdgeSampler(
        full_data['src'].values, full_data['dst'].values, seed=0)
    test_rand_sampler = RandEdgeSampler(
        full_data['src'].values, full_data['dst'].values, seed=2)

    train_ds = EdgePredictionDataset(train_data, train_rand_sampler)
    val_ds = EdgePredictionDataset(val_data, val_rand_sampler)
    test_ds = EdgePredictionDataset(test_data, test_rand_sampler)

    batch_size = data_config['batch_size']
    train_sampler = RandomStartBatchSampler(
        SequentialSampler(train_ds), batch_size=batch_size,
        drop_last=False, num_chunks=args.num_chunks)
    val_sampler = BatchSampler(
        SequentialSampler(val_ds),
        batch_size=batch_size, drop_last=False)
    test_sampler = BatchSampler(
        SequentialSampler(test_ds),
        batch_size=batch_size, drop_last=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, sampler=train_sampler, collate_fn=default_collate_ndarray,
        num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_ds, sampler=val_sampler, collate_fn=default_collate_ndarray,
        num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_ds, sampler=test_sampler, collate_fn=default_collate_ndarray,
        num_workers=args.num_workers)

    dgraph = build_dynamic_graph(full_data, **data_config)

    num_nodes = dgraph.num_vertices()
    num_edges = dgraph.num_edges()
    node_feats, edge_feats = load_feat(args.data)

    dim_node = 0 if node_feats is None else node_feats.shape[1]
    dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

    device = torch.device('cuda:{}'.format(args.gpu))
    model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                 memory_device=device)
    model.to(device)

    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(
        model_config['fanouts'], model_config['num_snapshots'], batch_size,
        node_feats, edge_feats)

    # Cache
    cache = caches.__dict__[args.cache](args.cache_ratio, num_nodes,
                                        num_edges, node_feats,
                                        edge_feats, device,
                                        pinned_nfeat_buffs, pinned_efeat_buffs)

    sampler = TemporalSampler(dgraph, **model_config)

    # only gnnlab static need to pass param
    if args.cache == 'GNNLabStaticCache':
        cache.init_cache(sampler, train_data, 2)
    else:
        cache.init_cache()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_e = train(train_loader, val_loader, train_sampler, sampler,
                   model, optimizer, criterion, cache, device)

    logging.info('Loading model at epoch {}...'.format(best_e))
    model.load_state_dict(torch.load(checkpoint_path))

    # To update the memory
    if model.has_memory():
        model.reset()
        evaluate(train_loader, sampler, model, criterion, cache, device)
        evaluate(val_loader, sampler, model, criterion, cache, device)

    ap, auc = evaluate(test_loader, sampler, model, criterion, cache, device)
    logging.info('Test ap:{:4f}  test auc:{:4f}'.format(ap, auc))


def train(train_loader, val_loader, train_sampler, sampler, model,
          optimizer, criterion, cache, device):

    best_ap = 0
    best_e = 0
    epoch_time_sum = 0
    early_stopper = EarlyStopMonitor()
    logging.info('Start training...')
    for e in range(args.epoch):
        model.train()
        total_loss = 0

        epoch_time_start = time.time()
        sample_time = 0
        feature_time = 0
        train_time = 0
        fetch_all_time = 0
        update_node_time = 0
        update_edge_time = 0
        cache_edge_ratio_sum = 0
        cache_node_ratio_sum = 0
        cuda_time = 0

        # init cache every epoch
        # TODO: maybe a better way to init cache in every epoch
        # only edge feature need to re-init between different epochs
        # cache.init_cache(sampler, train_data, 2)

        train_sampler.reset()
        model.train()
        model.reset()

        time_start = torch.cuda.Event(enable_timing=True)
        sample_end = torch.cuda.Event(enable_timing=True)
        feature_end = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        train_end = torch.cuda.Event(enable_timing=True)
        for i, (target_nodes, ts, eid) in enumerate(train_loader):
            time_start.record()
            mfgs = sampler.sample(target_nodes, ts)

            # Sample
            sample_end.record()
            mfgs_to_cuda(mfgs, device)
            cuda_end.record()

            mfgs = cache.fetch_feature(mfgs)
            feature_end.record()

            # Train
            optimizer.zero_grad()
            pred_pos, pred_neg = model(
                mfgs, eid=eid, edge_feats=cache.edge_features)

            loss = criterion(pred_pos, torch.ones_like(pred_pos))
            loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss) * len(target_nodes)
            loss.backward()
            optimizer.step()
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
        logging.info("Epoch: {}\n".format(e))
        logging.debug('Epoch time:{:.2f}s; dataloader time:{:.2f}s sample time:{:.2f}s; cuda time:{:.2f}s; feature time: {:.2f}s train time:{:.2f}s.; fetch time:{:.2f}s ; update node time:{:.2f}s; cache node ratio: {:.2f}; cache edge ratio: {:.2f}\n'.format(
            epoch_time, epoch_time - sample_time - feature_time - train_time - cuda_time, sample_time, cuda_time, feature_time, train_time, fetch_all_time, update_node_time, cache_node_ratio_sum / (i + 1), cache_edge_ratio_sum / (i + 1)))
        epoch_time_sum += epoch_time

        # Validation
        logging.info("***Start validation at epoch {}***".format(e))
        val_start = time.time()
        val_ap, val_auc = evaluate(
            val_loader, sampler, model, criterion, cache, device)
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


if __name__ == '__main__':
    main()
