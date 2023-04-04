import argparse
import datetime
import logging
import math
import os
import random
import time

import numpy as np
import psutil
import torch
import torch.distributed
import torch.distributed.rpc
import torch.nn
import torch.nn.parallel
import torch.utils.data
from sklearn.metrics import average_precision_score, roc_auc_score

import gnnflow.cache as caches
import gnnflow.distributed
import gnnflow.distributed.graph_services as graph_services
from gnnflow import DynamicGraph
from gnnflow.config import get_default_config
from gnnflow.distributed.dist_graph import DistributedDynamicGraph
from gnnflow.distributed.kvstore import KVStoreClient
from gnnflow.models.dgnn import DGNN
from gnnflow.models.graphsage import SAGE
from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import (EarlyStopMonitor, build_dynamic_graph, get_batch,
                           get_pinned_buffers, get_project_root_dir, load_feat,
                           load_partitioned_dataset, mfgs_to_cuda)

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
                    type=int, default=100)
parser.add_argument("--lr", help='learning rate', type=float, default=0.00001)
parser.add_argument("--num-workers", help="num workers for dataloaders",
                    type=int, default=0)
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
parser.add_argument("--disable-adaptive-block-size", action="store_true")

# distributed
parser.add_argument("--partition", action="store_true",
                    help="whether to partition the graph")
parser.add_argument("--initial-ingestion-batch-size", type=int, default=100000,
                    help="ingestion batch size")
parser.add_argument("--ingestion-batch-size", type=int, default=1000,
                    help="ingestion batch size")
parser.add_argument("--partition-strategy", type=str, default="roundrobin",
                    help="partition strategy for distributed training")
parser.add_argument("--dynamic-scheduling", action="store_true",
                    help="whether to use dynamic scheduling")
parser.add_argument("--not-partition-train-data", action="store_true",
                    help="whether not to partition the training data")
parser.add_argument("--snapshot-time-window", type=float, default=0,
                    help="time window for sampling")

args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info(args)

checkpoint_path = os.path.join(get_project_root_dir(),
                               '{}.pt'.format(args.model))

MiB = 1 << 20

start = time.time()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(data, sampler, model, criterion, cache, device, rand_sampler):
    model.eval()
    val_losses = list()
    aps = list()
    aucs_mrrs = list()

    with torch.no_grad():
        total_loss = 0
        for target_nodes, ts, eid in get_batch(data, args.batch_size, 0, rand_sampler):
            mfgs = sampler.sample(target_nodes, ts)
            mfgs_to_cuda(mfgs, device)
            mfgs = cache.fetch_feature(
                mfgs, eid, target_edge_features=args.use_memory)

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
        torch.distributed.init_process_group(
            'gloo', timeout=datetime.timedelta(seconds=36000))
        # ddp_pg = torch.distributed.new_group(
        #     ranks=np.arange(int(os.environ.get('WORLD_SIZE', 0))).tolist(),
        #     backend='nccl', timeout=datetime.timedelta(seconds=36000))
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        args.num_nodes = args.world_size // args.local_world_size
        args.partition &= args.num_nodes > 1
        mem = psutil.virtual_memory().percent
        logging.info("memory usage after init process group: {}".format(mem))
    else:
        args.local_rank = args.rank = 0
        args.local_world_size = args.world_size = 1

    logging.info("rank: {}, world_size: {}".format(args.rank, args.world_size))

    set_seed(args.seed + args.rank)

    model_config, data_config = get_default_config(args.model, args.data)
    model_config["snapshot_time_window"] = args.snapshot_time_window
    args.use_memory = model_config['use_memory']

    if args.distributed:
        # graph is stored in shared memory
        data_config["mem_resource_type"] = "shared"

    mem = psutil.virtual_memory().percent
    logging.info("memory usage: {}".format(mem))
    full_data = None
    node_feats = None
    edge_feats = None
    kvstore_client = None
    args.dim_memory = 0 if 'dim_memory' not in model_config else model_config['dim_memory']
    if args.partition:
        dgraph = build_dynamic_graph(
            **data_config, device=args.local_rank,
            adaptive_block_size=not args.disable_adaptive_block_size)
        graph_services.set_dgraph(dgraph)
        dgraph = graph_services.get_dgraph()
        mem = psutil.virtual_memory().percent
        logging.info("memory usage: {}".format(mem))
        gnnflow.distributed.initialize(args.rank, args.world_size,
                                       args.partition_strategy,
                                       args.num_nodes, args.data,
                                       args.dim_memory)
        gnnflow.distributed.dispatch_full_dataset(args.rank, args.data,
                                                  args.initial_ingestion_batch_size, args.ingestion_batch_size)

        # every worker will have a kvstore_client
        dim_node, dim_edge = graph_services.get_dim_node_edge()
        kvstore_client = KVStoreClient(
            dgraph.get_partition_table(),
            dgraph.num_partitions(), args.local_world_size,
            args.local_rank, dim_node, dim_edge, args.dim_memory)

        # print graph edge memory size and metadata
        avg_linked_list_length = dgraph._dgraph.avg_linked_list_length()
        graph_memory_usage = dgraph._dgraph.get_graph_memory_usage() / MiB
        metadata_memory_usage = dgraph._dgraph.get_metadata_memory_usage() / MiB

        # all reduce
        data_list = torch.tensor(
            [avg_linked_list_length, graph_memory_usage, metadata_memory_usage]).cuda()
        torch.distributed.all_reduce(
            data_list, op=torch.distributed.ReduceOp.SUM)
        data_list /= args.world_size
        avg_linked_list_length, graph_memory_usage, metadata_memory_usage = data_list.tolist()
        graph_memory_usage *= args.num_nodes
        logging.info('avg_linked_list_length: {:.2f}, graph mem usage: {:.2f}MiB, metadata (on GPU) mem usage: {:.2f}MiB (adaptive-block-size: {})'.format(
            avg_linked_list_length, graph_memory_usage, metadata_memory_usage, not args.disable_adaptive_block_size))

    else:
        dgraph = build_dynamic_graph(
            **data_config, device=args.local_rank, dataset_df=full_data)
        # put the features in shared memory when using distributed training
        node_feats, edge_feats = load_feat(
            args.data, shared_memory=args.distributed,
            local_rank=args.local_rank, local_world_size=args.local_world_size)

        dim_node = 0 if node_feats is None else node_feats.shape[1]
        dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

    num_nodes = dgraph.max_vertex_id() + 1
    logging.info("max_vertex id {}".format(dgraph.max_vertex_id()))
    num_edges = dgraph.num_edges()

    logging.info("use chunks build graph done")
    train_rand_sampler, eval_rand_sampler = graph_services.get_rand_sampler()
    logging.info("make sampler done")
    mem = psutil.virtual_memory().percent
    logging.info("memory usage: {}".format(mem))

    train_data, val_data, test_data = load_partitioned_dataset(
        args.data, rank=args.rank, world_size=args.world_size,
        partition_train_data=not args.not_partition_train_data)
    if not args.not_partition_train_data:
        train_data = graph_services.get_train_data()

    logging.info("make dataset done")
    mem = psutil.virtual_memory().percent
    logging.info("memory usage: {}".format(mem))
    args.batch_size = model_config['batch_size']
    # NB: learning rate is scaled by the number of workers
    args.lr = args.lr * math.sqrt(args.world_size)
    logging.info("batch size: {}, lr: {}".format(args.batch_size, args.lr))

    if args.distributed and args.data not in ['REDDIT', 'GDELT', 'MAG']:
        raise NotImplementedError("distributed training is not supported for dataset {}".format(
            args.data))

    logging.info("make dataloader done")
    dataset_end = time.time()

    device = torch.device('cuda:{}'.format(args.local_rank))
    logging.debug("device: {}".format(device))
    logging.info("dim_node: {}, dim_edge: {}".format(dim_node, dim_edge))
    mem = psutil.virtual_memory().percent
    logging.info("memory usage: {}".format(mem))
    if args.model == "GRAPHSAGE":
        model = SAGE(
            dim_node, model_config['dim_embed'], num_layers=model_config['num_layers'])
    elif args.model == 'GAT':
        model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                     memory_device=device, memory_shared=args.distributed,
                     kvstore_client=kvstore_client)
    else:
        model = DGNN(dim_node, dim_edge, **model_config, num_nodes=num_nodes,
                     memory_device=device, memory_shared=args.distributed,
                     kvstore_client=kvstore_client)
    model.to(device)

    if args.distributed:
        assert isinstance(dgraph, DistributedDynamicGraph)
        sampler = TemporalSampler(dgraph._dgraph, **model_config)
        graph_services.set_dsampler(sampler, args.dynamic_scheduling)
        sampler = graph_services.get_dsampler()
    else:
        assert isinstance(dgraph, DynamicGraph)
        sampler = TemporalSampler(dgraph, **model_config)
    build_graph_end = time.time()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=False)

    # pinned_nfeat_buffs, pinned_efeat_buffs = None, None
    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(
        model_config['fanouts'], model_config['num_snapshots'], args.batch_size,
        dim_node, dim_edge)

    # Cache
    cache = caches.__dict__[args.cache](args.edge_cache_ratio,
                                        args.node_cache_ratio,
                                        num_nodes,
                                        num_edges, device,
                                        node_feats, edge_feats,
                                        dim_node, dim_edge,
                                        pinned_nfeat_buffs,
                                        pinned_efeat_buffs,
                                        kvstore_client,
                                        args.partition)

    init_start = time.time()
    # only gnnlab static need to pass param
    if args.cache == 'GNNLabStaticCache':
        cache.init_cache(sampler=sampler, train_df=train_data,
                         pre_sampling_rounds=2, batch_size=args.batch_size)
    else:
        cache.init_cache()
    init_time = time.time() - init_start

    # all reduce the init time
    if args.distributed:
        init_time = torch.tensor(init_time, device=device)
        torch.distributed.all_reduce(
            init_time, op=torch.distributed.ReduceOp.SUM)
        init_time /= args.world_size
        init_time = init_time.item()

    logging.info("cache mem size: {:.2f} MiB, init cache time: {:.2f}s".format(
        cache.get_mem_size() / MiB, init_time))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    before_train_end = time.time()
    logging.info("load time: {}".format(dataset_end - start))
    logging.info("build graph time: {}".format(build_graph_end - dataset_end))
    logging.info("other time: {}".format(before_train_end - build_graph_end))
    logging.info("init time: {}".format(build_graph_end - start))
    logging.info("before train time: {}".format(before_train_end - start))
    mem = psutil.virtual_memory().percent
    logging.info("memory usage: {}".format(mem))
    best_e = train(train_data, val_data, sampler,
                   model, optimizer, criterion, cache, device, train_rand_sampler,
                   eval_rand_sampler)

    if args.rank == 0:
        logging.info('Loading model at epoch {}...'.format(best_e))
        model.load_state_dict(torch.load(checkpoint_path))

        ap, auc = evaluate(test_data, sampler, model,
                           criterion, cache, device, eval_rand_sampler)
        logging.info('Test ap:{:4f}  test auc:{:4f}'.format(ap, auc))

    if args.distributed:
        torch.distributed.barrier()
        dgraph.shutdown()
        sampler.shutdown()
        logging.info("Rank {} shutdown".format(args.rank))
        torch.distributed.rpc.shutdown()
        torch.distributed.destroy_process_group()
        logging.info("Rank {} shutdown done".format(args.rank))


def train(train_data, val_data, sampler, model, optimizer, criterion,
          cache, device, train_rand_sampler, val_rand_sampler):
    best_ap = 0
    best_e = 0
    epoch_time_sum = 0
    all_total_samples = 0
    early_stopper = EarlyStopMonitor()
    logging.info('Start training... distributed: {}'.format(args.distributed))
    torch.distributed.barrier()
    for e in range(args.epoch):
        model.train()
        cache.reset()
        if e > 0:
            model.module.reset()
        total_loss = 0
        cache_edge_ratio_sum = 0
        cache_node_ratio_sum = 0
        total_samples = 0
        total_sampling_time = 0
        total_feature_fetch_time = 0
        total_memory_fetch_time = 0
        total_memory_update_time = 0
        total_model_train_time = 0
        cv_sampling_time = 0

        epoch_time_start = time.time()
        for i, (target_nodes, ts, eid) in enumerate(get_batch(train_data, args.batch_size,
                                                              args.num_chunks, train_rand_sampler, args.world_size)):
            # Sample
            sample_start_time = time.time()
            mfgs = sampler.sample(target_nodes, ts)
            total_sampling_time += time.time() - sample_start_time

            # Feature
            mfgs_to_cuda(mfgs, device)
            feature_start_time = time.time()
            mfgs = cache.fetch_feature(
                mfgs, eid, target_edge_features=args.use_memory)
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
            model_start = time.time()
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

            model_end = time.time()
            total_model_train_time += model_end - model_start

            cache_edge_ratio_sum += cache.cache_edge_ratio
            cache_node_ratio_sum += cache.cache_node_ratio
            total_samples += len(target_nodes)

            if (i+1) % args.print_freq == 0:
                if args.distributed:
                    metrics = torch.tensor([total_loss, cache_edge_ratio_sum,
                                            cache_node_ratio_sum, total_samples, total_sampling_time,
                                            total_feature_fetch_time, total_memory_fetch_time, total_memory_update_time, total_model_train_time], device=device)
                    torch.distributed.all_reduce(metrics)
                    metrics /= args.world_size
                    total_loss, cache_edge_ratio_sum, cache_node_ratio_sum, \
                        total_samples, total_sampling_time, total_feature_fetch_time, total_memory_fetch_time, total_memory_update_time, total_model_train_time = metrics.tolist()

                    all_sampling_time = sampler.get_sampling_time()
                    std = all_sampling_time.std(dim=1).mean()
                    mean = all_sampling_time.mean(dim=1).mean()
                    cv_sampling_time += std / mean

                if args.rank == 0:
                    logging.info('Epoch {:d}/{:d} | Iter {:d}/{:d} | Throughput {:.2f} samples/s | Loss {:.4f} | Cache node ratio {:.4f} | Cache edge ratio {:.4f} | avg sampling time CV {:.4f} | Total sampling time: {:.2f}s | Total feature fetch time: {:.2f}s | Total memory fetch time: {:.2f}s | Total memory fetch time: {:.2f}s | Total model train time: {:.2f}s |Total time: {:.2f}s'.format(e + 1, args.epoch, i + 1, math.ceil(len(
                        train_data)/args.batch_size), total_samples * args.world_size / (time.time() - epoch_time_start), total_loss / (i + 1), cache_node_ratio_sum / (i + 1), cache_edge_ratio_sum / (i + 1), cv_sampling_time / ((i+1)/args.print_freq), total_sampling_time, total_feature_fetch_time, total_memory_fetch_time, total_memory_update_time, total_model_train_time, time.time() - epoch_time_start))
                    logging.info('Fetching Communication time: {:.3f}'.format(
                        cache.kvstore_client.comm_time))
        epoch_time = time.time() - epoch_time_start
        epoch_time_sum += epoch_time

        if args.distributed:
            metrics = torch.tensor([total_loss, cache_edge_ratio_sum,
                                    cache_node_ratio_sum, total_samples, total_sampling_time],
                                   device=device)
            torch.distributed.all_reduce(metrics)
            metrics /= args.world_size
            total_loss, cache_edge_ratio_sum, cache_node_ratio_sum, \
                total_samples, total_sampling_time = metrics.tolist()

            all_sampling_time = sampler.get_sampling_time()
            std = all_sampling_time.std(dim=1).mean()
            mean = all_sampling_time.mean(dim=1).mean()
            cv_sampling_time += std / mean

        # Validation
        val_start = time.time()
        val_ap, val_auc = evaluate(
            val_data, sampler, model, criterion, cache, device, val_rand_sampler)

        if args.distributed:
            val_res = torch.tensor([val_ap, val_auc]).to(device)
            torch.distributed.all_reduce(val_res)
            val_res /= args.world_size
            val_ap, val_auc = val_res[0].item(), val_res[1].item()

        val_end = time.time()
        val_time = val_end - val_start

        if args.distributed:
            metrics = torch.tensor([val_ap, val_auc, cache_edge_ratio_sum,
                                    cache_node_ratio_sum, total_samples], device=device)
            torch.distributed.all_reduce(metrics)
            metrics /= args.world_size
            val_ap, val_auc, cache_edge_ratio_sum, cache_node_ratio_sum, \
                total_samples = metrics.tolist()

            all_sampling_time = sampler.get_sampling_time()
            # if args.rank == 0:
            #     print(all_sampling_time)

            std = all_sampling_time.std(dim=1).mean()
            mean = all_sampling_time.mean(dim=1).mean()
            cv_sampling_time += std / mean

        all_total_samples += total_samples

        if args.rank == 0:
            logging.info("Epoch {:d}/{:d} | Validation ap {:.4f} | Validation auc {:.4f} | Train time {:.2f} s | Validation time {:.2f} s | Train Throughput {:.2f} samples/s | Cache node ratio {:.4f} | Cache edge ratio {:.4f} | sampling time CV {:.4f} | Total sampling time {:.2f}s | Total feature fetching time {:.2f}s Total memory fetch time: {:.2f}s | Total memory update time: {:.2f}s | Total model train time: {:.2f}s".format(
                e + 1, args.epoch, val_ap, val_auc, epoch_time, val_time, total_samples * args.world_size / epoch_time, cache_node_ratio_sum / (i + 1), cache_edge_ratio_sum / (i + 1), cv_sampling_time / ((i+1)/args.print_freq), total_sampling_time, total_feature_fetch_time, total_memory_fetch_time, total_memory_update_time, total_model_train_time))
            logging.info('Fetching Communication time: {:.3f}'.format(
                cache.kvstore_client.comm_time))
        if args.rank == 0 and val_ap > best_ap:
            best_e = e + 1
            best_ap = val_ap
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(
                "Best val AP: {:.4f} & val AUC: {:.4f}".format(val_ap, val_auc))

        if early_stopper.early_stop_check(val_ap):
            logging.info("Early stop at epoch {}".format(e))
            break

    if args.rank == 0:
        logging.info('Avg epoch time: {}, Avg train throughput: {}'.format(
            epoch_time_sum / (e + 1), all_total_samples * args.world_size / epoch_time_sum))

    if args.distributed:
        torch.distributed.barrier()

    return best_e


if __name__ == '__main__':
    main()
    logging.info('Training finished')
