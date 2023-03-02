import argparse
import logging
import math
import os
import random
import threading
import time
from collections import OrderedDict

import GPUtil
import numpy as np
import torch
import torch.distributed
import torch.nn
import torch.nn.parallel
import torch.utils.data
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import BatchSampler, SequentialSampler
from tqdm import tqdm

import gnnflow.cache as caches
from gnnflow.config import get_default_config
from gnnflow.data import (DistributedBatchSampler, EdgePredictionDataset,
                          RandomStartBatchSampler, default_collate_ndarray)
from gnnflow.models.dgnn import DGNN
from gnnflow.models.gat import GAT
from gnnflow.models.graphsage import SAGE
from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import (DstRandEdgeSampler, EarlyStopMonitor,
                           build_dynamic_graph, get_pinned_buffers,
                           get_project_root_dir, load_dataset, load_feat,
                           mfgs_to_cuda)

datasets = ['REDDIT', 'GDELT', 'LASTFM', 'MAG', 'MOOC', 'WIKI']
model_names = ['TGN', 'TGAT', 'DySAT', 'GRAPHSAGE', 'GAT']
cache_names = sorted(name for name in caches.__dict__
                     if not name.startswith("__")
                     and callable(caches.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=model_names, required=True,
                    help="model architecture" + '|'.join(model_names))
parser.add_argument("--data", choices=datasets, required=True,
                    help="dataset:" + '|'.join(datasets))
parser.add_argument("--epoch", help="maximum training epoch",
                    type=int, default=50)
parser.add_argument("--lr", help='learning rate', type=float, default=0.0001)
parser.add_argument("--num-workers", help="num workers for dataloaders",
                    type=int, default=8)
parser.add_argument("--num-chunks", help="number of chunks for batch sampler",
                    type=int, default=8)
parser.add_argument("--print-freq", help="print frequency",
                    type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ingestion-batch-size", type=int, default=1000,
                    help="ingestion batch size")
parser.add_argument("--save-node-embeddings-frequency", type=int, default=100,
                    help="save node embedding frequency")
parser.add_argument("--save-node-embeddings-num-last-iter", type=int, default=20,
                    help="save node embedding num last iter")
parser.add_argument("--max-staleness", type=int, default=3600,
                    help="max staleness for node embedding cache")

# optimization
parser.add_argument("--cache", choices=cache_names, help="feature cache:" +
                    '|'.join(cache_names))
parser.add_argument("--edge-cache-ratio", type=float, default=0,
                    help="cache ratio for edge feature cache")
parser.add_argument("--node-cache-ratio", type=float, default=0,
                    help="cache ratio for node feature cache")
parser.add_argument("--snapshot-time-window", type=float, default=0,
                    help="time window for sampling")

args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
logging.info(args)

checkpoint_path = os.path.join(get_project_root_dir(),
                               '{}.pt'.format(args.model))

training_nodes = set()


class TimeBoundedLRU:
    "LRU Cache that invalidates and refreshes old entries."

    def __init__(self, max_size, max_staleness):
        self.cache = OrderedDict()      # { node: (timestamp, result)}
        self.max_size = max_size
        self.max_staleness = max_staleness
        logging.debug("node embedding cache size: {}".format(max_size))

    def get(self, nodes, tss):
        mask, embeds = np.zeros(len(nodes), dtype=np.bool_), []
        for i, node, ts in enumerate(zip(nodes, tss)):
            if node in self.cache:
                self.cache.move_to_end(args)
                prev_ts, result = self.cache[args]
                if ts - prev_ts <= self.max_staleness:
                    mask[i] = True
                    embeds.append(result)
                    continue
            mask[i] = False
        logging.debug("node embedding cache hit rate: {}".format(
            np.sum(mask) / len(nodes)))
        return mask, np.stack(embeds) 

    def update(self, nodes, tss, embeds):
        for node, ts, embed in zip(nodes, tss, embeds):
            self.cache[node] = (ts, embed)
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

training = True


def gpu_load():
    global training
    time.sleep(5)
    while True:
        # stop when training is done
        # use a global variable to stop the thread
        if not training:
            break
        gpus = GPUtil.getGPUs()
        avg_load = sum([gpu.load for gpu in gpus]) / len(gpus)
        logging.info("GPU load: {:.2f}%".format(avg_load * 100))
        time.sleep(1)


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
            mfgs = cache.fetch_feature(
                mfgs, eid)

            if args.use_memory:
                b = mfgs[0][0]  # type: DGLBlock
                if args.distributed:
                    model.module.memory.prepare_input(b)
                    model.module.last_updated = model.module.memory_updater(b)
                else:
                    model.memory.prepare_input(b)
                    model.last_updated = model.memory_updater(b)

            pred_pos, pred_neg = model(mfgs)

            if args.use_memory:
                # NB: no need to do backward here
                # use one function
                if args.distributed:
                    model.module.memory.update_mem_mail(
                        **model.module.last_updated, edge_feats=cache.target_edge_features,
                        neg_sample_ratio=1)
                else:
                    model.memory.update_mem_mail(
                        **model.last_updated, edge_feats=cache.target_edge_features,
                        neg_sample_ratio=1)

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


def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))


def save_node_embeddings(nodes, cur_ts, sampler, model, cache, device, epoch, num_iter):
    model.eval()
    embeds_list = []
    with torch.no_grad():
        if args.use_memory:
            memory_backup = model.memory.backup()

        for target_nodes in split_given_size(nodes, args.batch_size):
            mfgs = sampler.sample(target_nodes, np.array(
                [cur_ts] * len(target_nodes)))
            mfgs_to_cuda(mfgs, device)
            mfgs = cache.fetch_feature(mfgs)

            if args.use_memory:
                b = mfgs[0][0]  # type: DGLBlock
                if args.distributed:
                    model.module.memory.prepare_input(b)
                    model.module.last_updated = model.module.memory_updater(b)
                else:
                    model.memory.prepare_input(b)
                    model.last_updated = model.memory_updater(b)

            embeds = model(mfgs, return_embed=True)
            embeds_list.append([embed.cpu().numpy() for embed in embeds])

        if args.use_memory:
            model.memory.restore(memory_backup)

    # zip
    embeds_list = list(zip(*embeds_list))
    for layer, embeds in enumerate(embeds_list):
        embeds = np.concatenate(embeds, axis=0)
        # save to file
        np.save("node_embeddings_{}_{}_layer{}_{}_{}.npy".format(
            args.model, args.data, layer+1, epoch, num_iter), embeds)


def main():
    args.distributed = int(os.environ.get('WORLD_SIZE', 0)) > 1
    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group('nccl')
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        args.local_rank = args.rank = 0
        args.local_world_size = args.world_size = 1

    logging.info("rank: {}, world_size: {}".format(args.rank, args.world_size))

    model_config, data_config = get_default_config(args.model, args.data)
    model_config["snapshot_time_window"] = args.snapshot_time_window
    args.use_memory = model_config['use_memory']

    if args.distributed:
        # graph is stored in shared memory
        data_config["mem_resource_type"] = "shared"

    train_data, val_data, test_data, full_data = load_dataset(args.data)
    train_rand_sampler = DstRandEdgeSampler(
        train_data['dst'].to_numpy(dtype=np.int32))
    val_rand_sampler = DstRandEdgeSampler(
        full_data['dst'].to_numpy(dtype=np.int32))
    test_rand_sampler = DstRandEdgeSampler(
        full_data['dst'].to_numpy(dtype=np.int32))

    training_nodes.update(train_data['src'].to_numpy(dtype=np.int64))
    training_nodes.update(train_data['dst'].to_numpy(dtype=np.int64))

    train_ds = EdgePredictionDataset(train_data, train_rand_sampler)
    val_ds = EdgePredictionDataset(val_data, val_rand_sampler)
    test_ds = EdgePredictionDataset(test_data, test_rand_sampler)

    batch_size = model_config['batch_size']
    args.batch_size = batch_size
    # NB: learning rate is scaled by the number of workers
    args.lr = args.lr * math.sqrt(args.world_size)
    logging.info("batch size: {}, lr: {}".format(batch_size, args.lr))

    if args.distributed:
        train_sampler = DistributedBatchSampler(
            SequentialSampler(train_ds), batch_size=batch_size,
            drop_last=False, rank=args.rank, world_size=args.world_size,
            num_chunks=args.num_chunks)
        val_sampler = DistributedBatchSampler(
            SequentialSampler(val_ds),
            batch_size=batch_size, drop_last=False, rank=args.rank,
            world_size=args.world_size)
        test_sampler = DistributedBatchSampler(
            SequentialSampler(test_ds),
            batch_size=batch_size, drop_last=False, rank=args.rank,
            world_size=args.world_size)
    else:
        train_sampler = RandomStartBatchSampler(
            SequentialSampler(train_ds), batch_size=batch_size, drop_last=False)
        val_sampler = BatchSampler(
            SequentialSampler(val_ds), batch_size=batch_size, drop_last=False)
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
        **data_config, device=args.local_rank)
    if args.distributed:
        torch.distributed.barrier()
    # insert in batch
    for i in tqdm(range(0, len(full_data), args.ingestion_batch_size)):
        batch = full_data[i:i + args.ingestion_batch_size]
        src_nodes = batch["src"].values.astype(np.int64)
        dst_nodes = batch["dst"].values.astype(np.int64)
        timestamps = batch["time"].values.astype(np.float32)
        eids = batch["eid"].values.astype(np.int64)
        dgraph.add_edges(src_nodes, dst_nodes, timestamps,
                         eids, add_reverse=False)
        if args.distributed:
            torch.distributed.barrier()

    num_nodes = dgraph.max_vertex_id() + 1
    num_edges = dgraph.num_edges()
    # put the features in shared memory when using distributed training
    node_feats, edge_feats = load_feat(
        args.data, shared_memory=args.distributed,
        local_rank=args.local_rank, local_world_size=args.local_world_size)

    dim_node = 0 if node_feats is None else node_feats.shape[1]
    dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

    device = torch.device('cuda:{}'.format(args.local_rank))
    logging.debug("device: {}".format(device))

    if args.model == "GRAPHSAGE":
        model = SAGE(dim_node, model_config['dim_embed'])
    elif args.model == 'GAT':
        model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                     memory_device=device, memory_shared=args.distributed)
    else:
        model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                     memory_device=device, memory_shared=args.distributed)
    model.to(device)

    sampler = TemporalSampler(dgraph, **model_config)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True)

    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(
        model_config['fanouts'], model_config['num_snapshots'], batch_size,
        dim_node, dim_edge)

    # Cache
    cache = caches.__dict__[args.cache](args.edge_cache_ratio, args.node_cache_ratio,
                                        num_nodes, num_edges, device,
                                        node_feats, edge_feats,
                                        dim_node, dim_edge,
                                        pinned_nfeat_buffs,
                                        pinned_efeat_buffs,
                                        None,
                                        False)

    # only gnnlab static need to pass param
    if args.cache == 'GNNLabStaticCache':
        cache.init_cache(sampler=sampler, train_df=train_data,
                         pre_sampling_rounds=2)
    else:
        cache.init_cache()

    node_embed_cache = TimeBoundedLRU(max_size=int(
        args.node_cache_ratio*num_nodes), max_staleness=args.max_staleness)

    logging.info("cache mem size: {:.2f} MB".format(
        cache.get_mem_size() / 1000 / 1000))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_e = train(train_loader, val_loader, sampler,
                   model, optimizer, criterion, cache, node_embed_cache, device)

    logging.info('Loading model at epoch {}...'.format(best_e))
    ckpt = torch.load(checkpoint_path)
    if args.distributed:
        model.module.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt['model'])
    if args.use_memory:
        if args.distributed:
            model.module.memory.restore(ckpt['memory'])
        else:
            model.memory.restore(ckpt['memory'])

    ap, auc = evaluate(test_loader, sampler, model,
                       criterion, cache, device)
    if args.distributed:
        metrics = torch.tensor([ap, auc], device=device)
        torch.distributed.all_reduce(metrics)
        metrics /= args.world_size
        ap, auc = metrics.tolist()
        if args.rank == 0:
            logging.info('Test ap:{:4f}  test auc:{:4f}'.format(ap, auc))

    if args.distributed:
        torch.distributed.barrier()


def train(train_loader, val_loader, sampler, model, optimizer, criterion,
          cache, node_embed_cache, device):
    global training
    best_ap = 0
    best_e = 0
    epoch_time_sum = 0
    early_stopper = EarlyStopMonitor()

    if args.local_rank == 0:
        gpu_load_thread = threading.Thread(target=gpu_load)
        gpu_load_thread.start()

    logging.info('Start training...')
    for e in range(args.epoch):
        model.train()
        cache.reset()
        if e > 0:
            if args.distributed:
                model.module.reset()
            else:
                model.reset()

        total_loss = 0
        cache_edge_ratio_sum = 0
        cache_node_ratio_sum = 0
        total_sampling_time = 0
        total_feature_fetch_time = 0
        total_memory_fetch_time = 0
        total_memory_update_time = 0
        total_memory_write_back_time = 0
        total_model_train_time = 0
        total_samples = 0

        node_embed_cache.clear()

        epoch_time_start = time.time()
        for i, (target_nodes, ts, eid) in enumerate(train_loader):
            mask, cache_embeds = node_embed_cache.get(target_nodes, ts)

            # Sample
            sample_start_time = time.time()
            mfgs = sampler.sample(target_nodes[~mask], ts[~mask])
            total_sampling_time += time.time() - sample_start_time

            # Feature
            mfgs_to_cuda(mfgs, device)
            feature_start_time = time.time()
            mfgs = cache.fetch_feature(
                mfgs, eid[~mask])
            total_feature_fetch_time += time.time() - feature_start_time

            if args.use_memory:
                b = mfgs[0][0]  # type: DGLBlock
                if args.distributed:
                    memory_fetch_start_time = time.time()
                    model.module.memory.prepare_input(b)
                    total_memory_fetch_time += time.time() - memory_fetch_start_time

                    memory_update_start_time = time.time()
                    model.module.last_updated = model.module.memory_updater(b)
                    total_memory_update_time += time.time() - memory_update_start_time
                else:
                    memory_fetch_start_time = time.time()
                    model.memory.prepare_input(b)
                    total_memory_fetch_time += time.time() - memory_fetch_start_time

                    memory_update_start_time = time.time()
                    model.last_updated = model.memory_updater(b)
                    total_memory_update_time += time.time() - memory_update_start_time

            # Train
            model_train_start_time = time.time()
            optimizer.zero_grad()
            # ugly
            embeds = model(mfgs, return_embed=True)
            embeds = torch.cat([cache_embeds, embeds], dim=0)
            pred_pos, pred_neg = model.edge_predictor(embeds)

            total_model_train_time += time.time() - model_train_start_time

            if args.use_memory:
                # NB: no need to do backward here
                with torch.no_grad():
                    # use one function
                    memory_write_back_start_time = time.time()
                    if args.distributed:
                        model.module.memory.update_mem_mail(
                            **model.module.last_updated, edge_feats=cache.target_edge_features,
                            neg_sample_ratio=1)
                    else:
                        model.memory.update_mem_mail(
                            **model.last_updated, edge_feats=cache.target_edge_features,
                            neg_sample_ratio=1)
                    total_memory_write_back_time += time.time() - memory_write_back_start_time

            model_train_start_time = time.time()
            loss = criterion(pred_pos, torch.ones_like(pred_pos))
            loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss) * len(target_nodes)
            loss.backward()
            optimizer.step()
            total_model_train_time += time.time() - model_train_start_time

            cache_edge_ratio_sum += cache.cache_edge_ratio
            cache_node_ratio_sum += cache.cache_node_ratio
            total_samples += len(target_nodes)

            # if args.rank == 0:
            #     times = i // args.save_node_embeddings_frequency * \
            #         args.save_node_embeddings_frequency
            #     if i % args.save_node_embeddings_frequency == 0 or \
            #             i == (times + args.save_node_embeddings_num_last_iter):
            #         # ugly
            #         cur_ts = float(ts[-1])
            #         save_node_embeddings(np.array(list(training_nodes)), cur_ts, sampler,
            #                              model, cache, device, e, i)

            if (i+1) % args.print_freq == 0:
                if args.distributed:
                    metrics = torch.tensor([total_loss, cache_edge_ratio_sum,
                                           cache_node_ratio_sum, total_samples,
                                           total_sampling_time, total_feature_fetch_time,
                                           total_memory_update_time,
                                           total_memory_write_back_time,
                                           total_model_train_time
                                            ]).to(device)
                    torch.distributed.all_reduce(metrics)
                    metrics /= args.world_size
                    total_loss, cache_edge_ratio_sum, cache_node_ratio_sum, \
                        total_samples, total_sampling_time, total_feature_fetch_time, \
                        total_memory_update_time, total_memory_write_back_time, \
                        total_model_train_time = metrics.tolist()

                if args.rank == 0:
                    logging.info('Epoch {:d}/{:d} | Iter {:d}/{:d} | Throughput {:.2f} samples/s | Loss {:.4f} | Cache node ratio {:.4f} | Cache edge ratio {:.4f} | Total Sampling Time {:.2f}s | Total Feature Fetching Time {:.2f}s | Total Memory Fetching Time {:.2f}s | Total Memory Update Time {:.2f}s | Total Memory Write Back Time {:.2f}s | Total Model Train Time {:.2f}s | Total Time {:.2f}s'.format(e + 1, args.epoch, i + 1, int(len(
                        train_loader)/args.world_size), total_samples * args.world_size / (time.time() - epoch_time_start), total_loss / (i + 1), cache_node_ratio_sum / (i + 1), cache_edge_ratio_sum / (i + 1), total_sampling_time, total_feature_fetch_time, total_memory_fetch_time, total_memory_update_time, total_memory_write_back_time, total_model_train_time, time.time() - epoch_time_start))

        epoch_time = time.time() - epoch_time_start
        epoch_time_sum += epoch_time

        # Validation
        val_start = time.time()
        val_ap, val_auc = evaluate(
            val_loader, sampler, model, criterion, cache, device)

        if args.distributed:
            val_res = torch.tensor([val_ap, val_auc]).to(device)
            torch.distributed.all_reduce(val_res)
            val_res /= args.world_size
            val_ap, val_auc = val_res[0].item(), val_res[1].item()

        val_end = time.time()
        val_time = val_end - val_start

        if args.distributed:
            metrics = torch.tensor([val_ap, val_auc, cache_edge_ratio_sum,
                                   cache_node_ratio_sum, total_samples,
                                   total_sampling_time, total_feature_fetch_time,
                                   total_memory_update_time,
                                   total_memory_write_back_time,
                                   total_model_train_time]).to(device)
            torch.distributed.all_reduce(metrics)
            metrics /= args.world_size
            val_ap, val_auc, cache_edge_ratio_sum, cache_node_ratio_sum,
            total_samples, total_sampling_time, total_feature_fetch_time,
            total_memory_update_time, total_memory_write_back_time,
            total_model_train_time = metrics.tolist()

        if args.rank == 0:
            logging.info("Epoch {:d}/{:d} | Validation ap {:.4f} | Validation auc {:.4f} | Train time {:.2f} s | Validation time {:.2f} s | Train Throughput {:.2f} samples/s | Cache node ratio {:.4f} | Cache edge ratio {:.4f} | Total Sampling Time {:.2f}s | Total Feature Fetching Time {:.2f}s | Total Memory Fetching Time {:.2f}s | Total Memory Update Time {:.2f}s | Total Memory Write Back Time {:.2f}s | Total Model Train Time {:.2f}s".format(

                e + 1, args.epoch, val_ap, val_auc, epoch_time, val_time, total_samples * args.world_size / epoch_time, cache_node_ratio_sum / (i + 1), cache_edge_ratio_sum / (i + 1), total_sampling_time, total_feature_fetch_time, total_memory_fetch_time, total_memory_update_time, total_memory_write_back_time, total_model_train_time))

        if args.rank == 0 and val_ap > best_ap:
            best_e = e + 1
            best_ap = val_ap
            if args.distributed:
                model_to_save = model.module
            else:
                model_to_save = model
            torch.save({
                'model': model_to_save.state_dict(),
                'memory': model_to_save.memory.backup() if args.use_memory else None
            }, checkpoint_path)
            logging.info(
                "Best val AP: {:.4f} & val AUC: {:.4f}".format(val_ap, val_auc))

        # if early_stopper.early_stop_check(val_ap):
        #     logging.info("Early stop at epoch {}".format(e))
        #     break

    if args.rank == 0:
        logging.info('Avg epoch time: {}'.format(epoch_time_sum / args.epoch))

    if args.distributed:
        torch.distributed.barrier()

    if args.local_rank == 0:
        training = False
        gpu_load_thread.join()

    return best_e


if __name__ == '__main__':
    main()
