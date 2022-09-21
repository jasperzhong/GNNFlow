import logging

import numpy as np
import pandas as pd
import torch
import torch.distributed
import torch.distributed.rpc as rpc

import dgnn.distributed.graph_services as graph_services
from dgnn.distributed.partition import get_partitioner


def initialize(rank: int, world_size: int, dataset: pd.DataFrame,
               ingestion_batch_size: int, partition_strategy: str,
               num_partition: int, undirected: bool):
    """
    Initialize the distributed environment.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The number of processes participating in the job.
        dataset (df.DataFrame): The dataset to ingest.
        ingestion_batch_size (int): The number of samples to ingest in each iteration.
        num_partition (int): The number of partitions to split the dataset into.
        undirected (bool): Whether the graph is undirected.
    """
    rpc.init_rpc("worker%d" % rank, rank=rank, world_size=world_size)
    logging.info("Rank %d: Initialized RPC.", rank)

    if rank == 0:
        dispatch(dataset, ingestion_batch_size,
                 partition_strategy, num_partition, undirected)

    torch.distributed.barrier()

    # check
    logging.info("Rank %d: Number of vertices: %d, number of edges: %d",
                 rank, graph_services.num_vertices(), graph_services.num_edges())


def dispatch(dataset: pd.DataFrame, ingestion_batch_size: int,
             partition_strategy: str, num_partition: int, undirected: bool):
    """
    Dispatch and partition the dataset to the workers.

    Args:
        dataset (df.DataFrame): The dataset to ingest.
        ingestion_batch_size (int): The number of samples to ingest in each iteration.
        partition_strategy (str): The partitioning strategy.
        num_partition (int): The number of partitions to split the dataset into.
        undirected (bool): Whether the graph is undirected.
    """
    num_worker_per_machine = torch.cuda.device_count()
    # Partition the dataset.
    partitioner = get_partitioner(partition_strategy, num_partition)
    for i in range(0, len(dataset), ingestion_batch_size):
        batch = dataset[i:i + ingestion_batch_size]
        src_nodes = batch["src"].values.astype(np.int64)
        dst_nodes = batch["dst"].values.astype(np.int64)
        timestamps = batch["time"].values.astype(np.float32)
        eids = batch["eid"].values.astype(np.int64)

        if undirected:
            src_nodes_ext = np.concatenate([src_nodes, dst_nodes])
            dst_nodes_ext = np.concatenate([dst_nodes, src_nodes])
            src_nodes = src_nodes_ext
            dst_nodes = dst_nodes_ext
            timestamps = np.concatenate([timestamps, timestamps])
            eids = np.concatenate([eids, eids])

        src_nodes = torch.from_numpy(src_nodes)
        dst_nodes = torch.from_numpy(dst_nodes)
        timestamps = torch.from_numpy(timestamps)
        eids = torch.from_numpy(eids)

        partitions = partitioner.partition(
            src_nodes, dst_nodes, timestamps, eids)

        # Dispatch the partitions to the workers.
        futures = []
        for partition_id, edges in enumerate(partitions):
            for worker_id in range(num_worker_per_machine):
                worker_rank = partition_id * num_worker_per_machine + worker_id
                future = rpc.rpc_async("worker%d" % worker_rank, graph_services.add_edges,
                                       args=(edges))
                futures.append(future)

        # Wait for the workers to finish.
        for future in futures:
            future.wait()
