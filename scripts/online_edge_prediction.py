import argparse
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed
import torch.nn
import torch.nn.parallel
import torch.utils.data
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import BatchSampler, SequentialSampler

import gnnflow.cache as caches
from gnnflow.config import get_default_config
from gnnflow.data import (DistributedBatchSampler, EdgePredictionDataset,
                          RandomStartBatchSampler, default_collate_ndarray)
from gnnflow.models.dgnn import DGNN
from gnnflow.models.gat import GAT
from gnnflow.models.graphsage import SAGE
from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import (DstRandEdgeSampler, EarlyStopMonitor, build_dynamic_graph, get_batch,
                           get_pinned_buffers, get_project_root_dir, load_dataset,
                           load_feat, load_partitioned_dataset, mfgs_to_cuda)

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
                    type=int, default=4)
parser.add_argument("--lr", help='learning rate', type=float, default=0.0005)
parser.add_argument("--num-workers", help="num workers for dataloaders",
                    type=int, default=8)
parser.add_argument("--num-chunks", help="number of chunks for batch sampler",
                    type=int, default=8)
parser.add_argument("--print-freq", help="print frequency",
                    type=int, default=100)
parser.add_argument("--seed", type=int, default=42)

# optimization
parser.add_argument("--cache", choices=cache_names, help="feature cache:" +
                    '|'.join(cache_names))
parser.add_argument("--edge-cache-ratio", type=float, default=0,
                    help="edge cache ratio for feature cache")
parser.add_argument("--node-cache-ratio", type=float, default=0,
                    help="node cache ratio for feature cache")
args = parser.parse_args()

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


def evaluate(df, sampler, model, criterion, cache, device, rand_edge_sampler):
    model.eval()
    val_losses = list()
    aps = list()
    aucs_mrrs = list()

    with torch.no_grad():
        total_loss = 0
        for target_nodes, ts, eid in get_batch(df=df,
                                               batch_size=args.batch_size,
                                               num_chunks=args.num_chunks,
                                               rand_edge_sampler=rand_edge_sampler,
                                               world_size=args.world_size):
            mfgs = sampler.sample(target_nodes, ts)
            mfgs_to_cuda(mfgs, device)
            mfgs = cache.fetch_feature(
                mfgs, eid)
            pred_pos, pred_neg = model(mfgs)

            if args.use_memory:
                # NB: no need to do backward here
                # use one function
                model.module.memory.update_mem_mail(
                    **model.module.last_updated, edge_feats=cache.target_edge_features,
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
    args.use_memory = model_config['use_memory']

    build_graph_time = 0
    total_training_time = 0
    # Phase 1
    if args.distributed:
        # graph is stored in shared memory
        data_config["mem_resource_type"] = "shared"
        phase1_build_graph_start = time.time()
        _, _, _, full_data = load_dataset(args.data)
        # deal with phase 1 dataset
        phase1_len = int(len(full_data) * 0.6)
        phase1_train = int(phase1_len * 0.9) + 1
        phase1_train_df = full_data[:phase1_train]
        phase1_val_df = full_data[phase1_train:phase1_len]
        train_rand_sampler = DstRandEdgeSampler(
            phase1_train_df['dst'].to_numpy())
        val_rand_sampler = DstRandEdgeSampler(
            full_data[:phase1_len]['dst'].to_numpy())

        logging.info("world_size: {}".format(args.world_size))
        dgraph = build_dynamic_graph(
            **data_config, device=args.local_rank, dataset_df=full_data[:phase1_len])
        phase1_build_graph_end = time.time()
        phase1_build_graph_time = phase1_build_graph_end - phase1_build_graph_start
        logging.info("phase1 build graph time: {}".format(
            phase1_build_graph_time))
        build_graph_time += phase1_build_graph_time
        # Fetch their own dataset, no redudancy
        train_index = list(range(args.rank, phase1_train, args.world_size))
        phase1_train_df = phase1_train_df.iloc[train_index]
        val_index = list(
            range(args.rank, len(phase1_val_df), args.world_size))
        phase1_val_df = phase1_val_df.iloc[val_index]
    else:
        # TODO: single GPU not implemented
        train_data, val_data, test_data, full_data = load_dataset(args.data)
        train_rand_sampler = DstRandEdgeSampler(train_data['dst'].values)
        val_rand_sampler = DstRandEdgeSampler(full_data['dst'].values)
        test_rand_sampler = DstRandEdgeSampler(full_data['dst'].values)

    batch_size = data_config['batch_size']
    args.batch_size = batch_size
    # NB: learning rate is scaled by the number of workers
    args.lr = args.lr * math.sqrt(args.world_size)
    logging.info("batch size: {}, lr: {}".format(batch_size, args.lr))

    # if not args.distributed:
    #     dgraph = build_dynamic_graph(
    #         **data_config, device=args.local_rank, dataset_df=full_data)

    num_nodes = dgraph.max_vertex_id() + 1
    num_edges = dgraph.num_edges()
    logging.info("graph memory usage: {}".format(
        dgraph.get_graph_memory_usage()))
    logging.info("graph meta data memory usage: {}".format(
        dgraph.get_metadata_memory_usage()))
    # put the features in shared memory when using distributed training
    # phase1 load all the features
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
        model = GAT(dim_node, model_config['dim_embed'])
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
    cache = caches.__dict__[args.cache](args.edge_cache_ratio,
                                        args.node_cache_ratio,
                                        num_nodes,
                                        num_edges, device,
                                        node_feats, edge_feats,
                                        dim_node, dim_edge,
                                        pinned_nfeat_buffs,
                                        pinned_efeat_buffs)

    # only gnnlab static need to pass param
    if args.cache == 'GNNLabStaticCache':
        cache.init_cache(sampler=sampler, train_df=train_data,
                         pre_sampling_rounds=2)
    else:
        cache.init_cache()

    logging.info("cache mem size: {:.2f} MB".format(
        cache.get_mem_size() / 1000 / 1000))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # phase1 training
    with open("online_ap_{}_{}_{}.txt".format(args.model, args.data, 0), "a") as f_phase2:
        f_phase2.write("Phase1\n")
    with open("online_auc_{}_{}_{}.txt".format(args.model, args.data, 0), "a") as f_phase2:
        f_phase2.write("Phase1\n")
    phase1_train_start = time.time()
    best_e = train(phase1_train_df, phase1_val_df, sampler,
                   model, optimizer, criterion, cache, device, train_rand_sampler, val_rand_sampler)
    phase1_train_end = time.time()
    phase1_train_time = phase1_train_end - phase1_train_start
    logging.info("phase1 train time: {}".format(phase1_train_time))
    total_training_time += phase1_train_time
    # phase1 training done
    if args.distributed:
        torch.distributed.barrier()

    # phase2
    # update rand_sampler
    logging.info("Phase2 start")
    with open("online_ap_{}_{}_{}.txt".format(args.model, args.data, 0), "a") as f_phase2:
        f_phase2.write("Phase1\n")
    with open("online_auc_{}_{}_{}.txt".format(args.model, args.data, 0), "a") as f_phase2:
        f_phase2.write("Phase1\n")
    # retrain 100 times
    retrain_num = 100
    phase2_df = full_data[phase1_len:]
    phase2_len = len(phase2_df)
    incremental_step = int(phase2_len / retrain_num)
    replay_ratio = 0
    for i in range(retrain_num):
        phase2_build_graph_start = time.time()
        increment_df = phase2_df[i*incremental_step: (i+1)*incremental_step]
        src = increment_df['src'].to_numpy()
        dst = increment_df['dst'].to_numpy()
        ts = increment_df['time'].to_numpy()
        eid = increment_df['eid'].to_numpy()
        # update graph
        dgraph.add_edges(src, dst, ts, eid, add_reverse=False)

        phase2_build_graph_end = time.time()
        phase2_build_graph_time = phase2_build_graph_end - phase2_build_graph_start
        logging.info("phase2 {}th training build graph time: {}".format(
            i+1, phase2_build_graph_time))
        build_graph_time += phase2_build_graph_time

        phase2_train_start = time.time()
        # random sample phase2 train dataset -> get phase2_train_df & phase2_val_df
        phase2_new_data_start = phase1_len + incremental_step * i
        phase2_new_data_end = phase1_len + incremental_step * (i + 1)
        num_replay = int(replay_ratio * phase2_new_data_start)
        new_data_index = torch.arange(
            phase2_new_data_start, phase2_new_data_end)
        old_data_index = torch.multinomial(torch.ones(
            phase2_new_data_start), num_replay).sort().values
        all_index = torch.cat((old_data_index, new_data_index))
        phase2_train_len = int(len(all_index) * 0.9) + 1
        train_index = all_index[:phase2_train_len]
        val_index = all_index[phase2_train_len:]
        phase2_train_df = full_data.iloc[train_index.numpy()]
        phase2_val_df = full_data.iloc[val_index.numpy()]

        # Rand Sampler
        val_rand_sampler.add_dst_list(dst)
        # train rand sampler, all the data before the first validation data
        train_rand_sampler = DstRandEdgeSampler(
            full_data[:val_index[0]]['dst'].to_numpy())

        # Fetch their own dataset, no redudancy
        train_index = list(
            range(args.rank, len(phase2_train_df), args.world_size))
        phase2_train_df = phase2_train_df.iloc[train_index]
        val_index = list(
            range(args.rank, len(phase2_val_df), args.world_size))
        phase2_val_df = phase2_val_df.iloc[val_index]

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(phase2_train_df, phase2_val_df, sampler,
              model, optimizer, criterion, cache, device, train_rand_sampler, val_rand_sampler)
        phase2_train_end = time.time()
        phase2_train_time = phase2_train_end - phase2_train_start
        logging.info("phase2 {}th train time: {}".format(
            i+1, phase2_train_time))
        total_training_time += phase2_train_time
        if args.distributed:
            torch.distributed.barrier()

    # all end
    logging.info("total build graph time: {}".format(build_graph_time))
    logging.info("total train time {}".format(total_training_time))
    logging.info("total time: {}".format(
        build_graph_time + total_training_time))


def train(train_df, val_df, sampler, model, optimizer, criterion,
          cache, device, train_rand_sampler, test_rand_sampler):
    best_ap = 0
    best_e = 0
    epoch_time_sum = 0
    early_stopper = EarlyStopMonitor()
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
        total_samples = 0
        total_sample_time = 0

        epoch_time_start = time.time()
        for i, (target_nodes, ts, eid) in enumerate(get_batch(df=train_df,
                                                              batch_size=args.batch_size,
                                                              num_chunks=args.num_chunks,
                                                              rand_edge_sampler=train_rand_sampler,
                                                              world_size=args.world_size)):
            # Sample
            start = time.time()
            mfgs = sampler.sample(target_nodes, ts)
            total_sample_time += time.time() - start

            # Feature
            mfgs_to_cuda(mfgs, device)
            mfgs = cache.fetch_feature(
                mfgs, eid)

            # Train
            optimizer.zero_grad()
            pred_pos, pred_neg = model(mfgs)

            if args.use_memory:
                # NB: no need to do backward here
                with torch.no_grad():
                    # use one function
                    model.module.memory.update_mem_mail(
                        **model.module.last_updated, edge_feats=cache.target_edge_features,
                        neg_sample_ratio=1)

            loss = criterion(pred_pos, torch.ones_like(pred_pos))
            loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss) * len(target_nodes)
            loss.backward()
            optimizer.step()

            cache_edge_ratio_sum += cache.cache_edge_ratio
            cache_node_ratio_sum += cache.cache_node_ratio
            total_samples += len(target_nodes)

            if (i+1) % args.print_freq == 0:
                if args.distributed:
                    metrics = torch.tensor([total_loss, cache_edge_ratio_sum,
                                            cache_node_ratio_sum, total_samples, total_sample_time],
                                           device=device)
                    torch.distributed.all_reduce(metrics)
                    metrics /= args.world_size
                    total_loss, cache_edge_ratio_sum, cache_node_ratio_sum, \
                        total_samples, total_sample_time = metrics.tolist()

                if args.rank == 0:
                    logging.info('Epoch {:d}/{:d} | Iter {:d}/{:d} | Throughput {:.2f} samples/s | Loss {:.4f} | Cache node ratio {:.4f} | Cache edge ratio {:.4f} | Total sample time {:.2f}s'.format(e + 1, args.epoch, i + 1, int(len(
                        train_df)/1), total_samples * args.world_size / (time.time() - epoch_time_start), total_loss / (i + 1), cache_node_ratio_sum / (i + 1), cache_edge_ratio_sum / (i + 1), total_sample_time))

        epoch_time = time.time() - epoch_time_start
        epoch_time_sum += epoch_time

        # Validation
        val_start = time.time()
        val_ap, val_auc = evaluate(
            val_df, sampler, model, criterion, cache, device, test_rand_sampler)

        if args.distributed:
            val_res = torch.tensor([val_ap, val_auc]).to(device)
            torch.distributed.all_reduce(val_res)
            val_res /= args.world_size
            val_ap, val_auc = val_res[0].item(), val_res[1].item()

        val_end = time.time()
        val_time = val_end - val_start

        if args.distributed:
            metrics = torch.tensor([val_ap, val_auc, cache_edge_ratio_sum,
                                    cache_node_ratio_sum, total_samples, total_sample_time],
                                   device=device)
            torch.distributed.all_reduce(metrics)
            metrics /= args.world_size
            val_ap, val_auc, cache_edge_ratio_sum, cache_node_ratio_sum, \
                total_samples, total_sample_time = metrics.tolist()

        if args.rank == 0:
            logging.info("Epoch {:d}/{:d} | Validation ap {:.4f} | Validation auc {:.4f} | Train time {:.2f} s | Validation time {:.2f} s | Train Throughput {:.2f} samples/s | Cache node ratio {:.4f} | Cache edge ratio {:.4f} | Total sample time {:.2f}s".format(
                e + 1, args.epoch, val_ap, val_auc, epoch_time, val_time, total_samples * args.world_size / epoch_time, cache_node_ratio_sum / (i + 1), cache_edge_ratio_sum / (i + 1), total_sample_time))
            with open("online_ap_{}_{}_{}.txt".format(args.model, args.data, 0), "a") as f_phase2:
                f_phase2.write("{:.4f}\n".format(val_ap))
            with open("online_auc_{}_{}_{}.txt".format(args.model, args.data, 0), "a") as f_phase2:
                f_phase2.write("{:.4f}\n".format(val_auc))

        if args.rank == 0 and e > 1 and val_ap > best_ap:
            best_e = e + 1
            best_ap = val_ap
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(
                "Best val AP: {:.4f} & val AUC: {:.4f}".format(val_ap, val_auc))

        if early_stopper.early_stop_check(val_ap):
            logging.info("Early stop at epoch {}".format(e))
            break

    if args.rank == 0:
        logging.info('Avg epoch time: {}'.format(epoch_time_sum / args.epoch))

    if args.distributed:
        torch.distributed.barrier()

    return best_e


if __name__ == '__main__':
    main()
