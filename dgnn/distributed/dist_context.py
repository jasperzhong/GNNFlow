import logging

import pandas as pd
import torch
import torch.distributed
import torch.distributed.rpc as rpc

import dgnn.distributed.graph_services as graph_services
from dgnn.distributed.dispatcher import get_dispatcher


def initialize(rank: int, world_size: int, dataset: pd.DataFrame,
               ingestion_batch_size: int, partition_strategy: str,
               num_partitions: int, undirected: bool):
    """
    Initialize the distributed environment.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The number of processes participating in the job.
        dataset (df.DataFrame): The dataset to ingest.
        ingestion_batch_size (int): The number of samples to ingest in each iteration.
        num_partitions (int): The number of partitions to split the dataset into.
        undirected (bool): Whether the graph is undirected.
    """
    rpc.init_rpc("worker%d" % rank, rank=rank, world_size=world_size)
    logging.info("Rank %d: Initialized RPC.", rank)

    if rank == 0:
        dispatcher = get_dispatcher(partition_strategy, num_partitions)
        dispatcher.partition_graph(dataset, ingestion_batch_size,
                                   undirected)

    # check
    torch.distributed.barrier()
    logging.info("Rank %d: Number of vertices: %d, number of edges: %d",
                 rank, graph_services.num_vertices(), graph_services.num_edges())
