import logging
import os
import threading
import time

import numpy as np
import psutil
import torch
import torch.distributed
import torch.distributed.rpc as rpc

import gnnflow.distributed.graph_services as graph_services
from gnnflow.distributed.dispatcher import get_dispatcher
from gnnflow.distributed.kvstore import KVStoreServer
from gnnflow.utils import (DstRandEdgeSampler, load_dataset, get_node_feats,
                           load_dataset_in_chunks, load_feat, load_node_feat,
                           local_world_size)


def initialize(rank: int, world_size: int,
               initial_ingestion_batch_size: int,
               ingestion_batch_size: int, partition_strategy: str,
               num_partitions: int, undirected: bool, data_name: str,
               dim_memory: int, chunksize: int, partition_train_data: bool):
    """
    Initialize the distributed environment.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The number of processes participating in the job.
        initial_ingestion_batch_size (int): The number of edges to ingest in
        ingestion_batch_size (int): The number of samples to ingest in each iteration.
        num_partitions (int): The number of partitions to split the dataset into.
        undirected (bool): Whether the graph is undirected.
        data_name (str): the dataset name of the dataset for loading features.
        dim_memory (int): the dimension of memory
        chunksize (int): the chunksize of the dataset
        partition_train_data (bool): whether to partition the train data
    """
    # NB: disable IB according to https://github.com/pytorch/pytorch/issues/86962
    rpc.init_rpc("worker%d" % rank, rank=rank, world_size=world_size,
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                     rpc_timeout=1800,
                     num_worker_threads=32,
                     _transports=["shm", "uv"],
                     _channels=["cma", "mpt_uv", "basic", "cuda_xth", "cuda_ipc", "cuda_basic"]))
    logging.info("Rank %d: Initialized RPC.", rank)

    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize the KVStore.
    if local_rank == 0:
        graph_services.set_kvstore_server(KVStoreServer())

    train_dst_set = set()
    full_dst_set = set()

    start = time.time()
    if rank == 0:
        dispatcher = get_dispatcher(partition_strategy, num_partitions)
        # load the feature only at rank 0
        node_feats = None
        _, edge_feats = load_feat(data_name, load_node=False)

        # load_node_feat_thread = threading.Thread(
        #     target=load_node_feat, args=(data_name, ))
        # load_node_feat_thread.start()
        load_node_feat(data_name)
        node_feats = get_node_feats()

        logging.info("Rank %d: Loaded features in %f seconds.", rank,
                     time.time() - start)

        # read csv in chunks
        df_iterator = load_dataset_in_chunks(data_name, chunksize=chunksize)
        for i, dataset in enumerate(df_iterator):
            dataset.rename(columns={'Unnamed: 0': 'eid'}, inplace=True)
            train_end = dataset['ext_roll'].values.searchsorted(1)

            train_dst_set.update(dataset['dst'].values[:train_end].tolist())
            full_dst_set.update(dataset['dst'].values.tolist())

            if i > 0:
                initial_ingestion_batch_size = ingestion_batch_size

            dispatcher.partition_graph(dataset, initial_ingestion_batch_size,
                                       ingestion_batch_size,
                                       undirected, node_feats, edge_feats,
                                       dim_memory, partition_train_data)
            del dataset

        # deal with unpartitioned nodes
        partition_table = dispatcher._partitioner._partition_table
        unassigned_nodes_index = (
            partition_table == -1).nonzero().squeeze(dim=1)
        logging.info("len of unassigned nodes: {}".format(
            len(unassigned_nodes_index)))

        if len(unassigned_nodes_index) > 0:
            partition_id = torch.arange(
                len(unassigned_nodes_index), dtype=torch.int8) % dispatcher._num_partitions
            partition_table[unassigned_nodes_index] = partition_id

        # join the thread
        # load_node_feat_thread.join()

        dim_node = 0 if node_feats is None else node_feats.shape[1]
        dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

        del edge_feats
        dispatcher.broadcast_graph_metadata()
        dispatcher.broadcast_partition_table()
        dispatcher.broadcast_node_edge_dim(dim_node, dim_edge)


        # node feature/memory
        partition_table = graph_services.get_partition_table()
        dim_edge = graph_services.get_dim_edge()
        if node_feats is not None:
            futures = []
            for partition_id in reversed(range(dispatcher._num_partitions)):
                # the KVStore server is in local_rank 0
                partition_mask = partition_table == partition_id
                assert partition_mask.sum() > 0  # should not be 0
                vertices = torch.arange(len(partition_table), dtype=torch.long)
                partition_vertices = vertices[partition_mask]
                keys = partition_vertices
                kvstore_rank = partition_id * local_world_size()
                features = node_feats[keys]
                if partition_id == 0:
                    graph_services.push_tensors(keys, features, 'node')
                else:
                    futures.append(rpc.rpc_async("worker%d" % kvstore_rank, graph_services.push_tensors,
                                                 args=(keys, features, 'node')))
                logging.info(
                    "partition: {} dispatch done".format(partition_id))
                mem = psutil.virtual_memory().percent
                logging.info("peak memory usage: {}".format(mem))
            del node_feats

            for future in futures:
                future.wait()
            mem = psutil.virtual_memory().percent
            logging.info("node features done memory usage: {}".format(mem))

        if dim_memory > 0:
            futures = []
            for partition_id in range(dispatcher._num_partitions):
                partition_mask = partition_table == partition_id
                assert partition_mask.sum() > 0  # should not be 0
                vertices = torch.arange(
                    len(partition_table), dtype=torch.long)
                partition_vertices = vertices[partition_mask]
                keys = partition_vertices.contiguous()
                kvstore_rank = partition_id * local_world_size()
                if partition_id == 0:
                    graph_services.init_memory(keys, dim_memory, dim_edge)
                else:
                    futures.append(rpc.rpc_async("worker%d" % kvstore_rank, graph_services.init_memory,
                                                 args=(keys, dim_memory, dim_edge)))
                logging.info(
                    "partition: {} memory dispatch done".format(partition_id))
                mem = psutil.virtual_memory().percent
                logging.info("peak memory usage: {}".format(mem))

            for future in futures:
                future.wait()

            mem = psutil.virtual_memory().percent
            logging.info("memory dispatch done memory usage: {}".format(mem))

        # deal with rand sampler
        train_rand_sampler = DstRandEdgeSampler(list(train_dst_set))
        val_rand_sampler = DstRandEdgeSampler(list(full_dst_set))
        test_rand_sampler = DstRandEdgeSampler(list(full_dst_set))
        dispatcher.broadcast_rand_sampler(
            train_rand_sampler, val_rand_sampler, test_rand_sampler)
        logging.info("make sampler done")

    # check
    torch.distributed.barrier()
    if rank == 0:
        logging.info("Rank %d: Ingested data in %f seconds.", rank,
                     time.time() - start)

    logging.info("Rank %d: Number of vertices: %d, number of edges: %d",
                 rank, graph_services.num_vertices(), graph_services.num_edges())
    logging.info("Rank %d: partition table shape: %s",
                 rank, str(graph_services.get_partition_table().shape))

    # save partition table
    partition_table_numpy = graph_services.get_partition_table().numpy()
    np.savetxt('partition_table.txt', partition_table_numpy, delimiter='\n')
    dgraph = graph_services.get_dgraph()
    logging.info("Rank %d: local number of vertices: %d, number of edges: %d",
                 rank, dgraph._dgraph.num_vertices(), dgraph._dgraph.num_edges())
    mem = psutil.virtual_memory().percent
    logging.info("build graph done memory usage: {}".format(mem))
