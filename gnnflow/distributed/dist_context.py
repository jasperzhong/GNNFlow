from gnnflow.utils import get_project_root_dir, load_dataset, load_feat
from gnnflow.utils import RandEdgeSampler, get_project_root_dir, load_dataset, load_feat
import logging
import os
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.distributed
import torch.distributed.rpc as rpc

import gnnflow.distributed.graph_services as graph_services
from gnnflow.distributed.dispatcher import get_dispatcher
from gnnflow.distributed.kvstore import KVStoreServer
<< << << < HEAD
== == == =
>>>>>> > main


def initialize(rank: int, world_size: int, dataset: pd.DataFrame,
               initial_ingestion_batch_size: int,
               ingestion_batch_size: int, partition_strategy: str,
               num_partitions: int, undirected: bool, data_name: str,
               use_memory: int, chunk: int):
    """
    Initialize the distributed environment.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The number of processes participating in the job.
        dataset (df.DataFrame): The dataset to ingest.
        initial_ingestion_batch_size (int): The number of edges to ingest in
        ingestion_batch_size (int): The number of samples to ingest in each iteration.
        num_partitions (int): The number of partitions to split the dataset into.
        undirected (bool): Whether the graph is undirected.
        data_name (str): the dataset name of the dataset for loading features.
        use_memory (bool): if the kvstore need to initialize the memory.
        chunk (int): the number of chunks of the dataset
    """
    # NB: disable IB according to https://github.com/pytorch/pytorch/issues/86962
    rpc.init_rpc("worker%d" % rank, rank=rank, world_size=world_size,
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                     rpc_timeout=1800,
                     _transports=["shm", "uv"],
                     _channels=["cma", "mpt_uv", "basic", "cuda_xth", "cuda_ipc", "cuda_basic"]))
    logging.info("Rank %d: Initialized RPC.", rank)

    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize the KVStore.
    if local_rank == 0:
        graph_services.set_kvstore_server(KVStoreServer())

    start = time.time()
    if rank == 0:
        dispatcher = get_dispatcher(partition_strategy, num_partitions)
        # load the feature only at rank 0
        node_feats, edge_feats = load_feat(data_name)
        logging.info("Rank %d: Loaded features in %f seconds.", rank,
                     time.time() - start)
        if chunk > 1:
            for i in range(chunk):  # 10 chunks of data
                # train_data, val_data, test_data, full_data = load_dataset(args.data)
                logging.info("{}th chunk add edges".format(i))
                data_dir = os.path.join(get_project_root_dir(), "data")
                path = os.path.join(data_dir, 'MAG', 'edges_{}.csv'.format(i))
                dataset = pd.read_csv(path, engine='pyarrow')
                dispatcher.partition_graph(dataset, initial_ingestion_batch_size,
                                           ingestion_batch_size,
                                           undirected, node_feats, edge_feats,
                                           use_memory)
                del dataset
        else:
            # for those datasets that don't need chunks
            train_data, _, _, dataset = load_dataset(data_name)
            dispatcher.partition_graph(dataset, initial_ingestion_batch_size,
                                       ingestion_batch_size,
                                       undirected, node_feats, edge_feats,
                                       use_memory)
            train_rand_sampler = RandEdgeSampler(
                train_data['src'].values, train_data['dst'].values)
            val_rand_sampler = RandEdgeSampler(
                dataset['src'].values, dataset['dst'].values)
            test_rand_sampler = RandEdgeSampler(
                dataset['src'].values, dataset['dst'].values)
            logging.info("make sampler done")
            dispatcher.broadcast_rand_sampler(
                train_rand_sampler, val_rand_sampler, test_rand_sampler)
            del train_data
            del dataset

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
