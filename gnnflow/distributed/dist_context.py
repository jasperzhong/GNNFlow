import logging
import os

import pandas as pd
import torch
import torch.distributed
import torch.distributed.rpc as rpc
import numpy as np

import gnnflow.distributed.graph_services as graph_services
from gnnflow.distributed.dispatcher import get_dispatcher
from gnnflow.distributed.kvstore import KVStoreServer
from gnnflow.utils import load_feat


def initialize(rank: int, world_size: int, dataset: pd.DataFrame,
               ingestion_batch_size: int, partition_strategy: str,
               num_partitions: int, undirected: bool, data_name: str,
               use_memory: int):
    """
    Initialize the distributed environment.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The number of processes participating in the job.
        dataset (df.DataFrame): The dataset to ingest.
        ingestion_batch_size (int): The number of samples to ingest in each iteration.
        num_partitions (int): The number of partitions to split the dataset into.
        undirected (bool): Whether the graph is undirected.
        data_name (str): the dataset name of the dataset for loading features.
        use_memory (bool): if the kvstore need to initialize the memory.
    """
    # NB: disable IB according to https://github.com/pytorch/pytorch/issues/86962
    rpc.init_rpc("worker%d" % rank, rank=rank, world_size=world_size)
    #  rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
    #      _transports=["shm", "uv"],
    #      _channels=["cma", "mpt_uv", "basic", "cuda_xth", "cuda_ipc", "cuda_basic"]))
    logging.info("Rank %d: Initialized RPC.", rank)

    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize the KVStore.
    if local_rank == 0:
        graph_services.set_kvstore_server(KVStoreServer())

    if rank == 0:
        dispatcher = get_dispatcher(partition_strategy, num_partitions)
        # load the feature only at rank 0
        node_feats, edge_feats = load_feat(data_name)
        node_feats = None
        # if edge_feats != None:
        #     edge_len = len(edge_feats)
        #     edge_feats = edge_feats[:edge_len // 100]
        dispatcher.partition_graph(dataset, ingestion_batch_size,
                                   undirected, node_feats, edge_feats,
                                   use_memory)

    # check
    torch.distributed.barrier()
    logging.info("Rank %d: Number of vertices: %d, number of edges: %d",
                 rank, graph_services.num_vertices(), graph_services.num_edges())
    logging.info("Rank %d: partition table shape: %s",
                 rank, str(graph_services.get_partition_table().shape))

    # save partition table
    partition_table_numpy = graph_services.get_partition_table().numpy()
    np.savetxt('partiton_table.txt', partition_table_numpy, delimiter='\n')
    dgraph = graph_services.get_dgraph()
    logging.info("Rank %d: local number of vertices: %d, number of edges: %d",
                 rank, dgraph._dgraph.num_vertices(), dgraph._dgraph.num_edges())
