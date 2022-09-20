import pandas as pd
import torch
import torch.distributed
import torch.distributed.rpc as rpc

from dgnn.distributed.partition import get_partitioner


def initialize(rank: int, world_size: int, dataset: pd.DataFrame, ingestion_batch_size: int, partition_strategy: str, num_partition: int):
    """
    Initialize the distributed environment.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The number of processes participating in the job.
        dataset (df.DataFrame): The dataset to ingest.
        ingestion_batch_size (int): The number of samples to ingest in each iteration.
        num_partition (int): The number of partitions to split the dataset into.
    """
    rpc.init_rpc("worker%d" % rank, rank=rank, world_size=world_size)

    if rank == 0:
        dispatch(dataset, ingestion_batch_size,
                 partition_strategy, num_partition)

    torch.distributed.barrier()


def dispatch(dataset: pd.DataFrame, ingestion_batch_size: int, partition_strategy: str, num_partition: int):
    """
    Dispatch and partition the dataset to the workers.

    Args:
        dataset (df.DataFrame): The dataset to ingest.
        ingestion_batch_size (int): The number of samples to ingest in each iteration.
        partition_strategy (str): The partitioning strategy.
        num_partition (int): The number of partitions to split the dataset into.
    """
    # Partition the dataset.
    partitioner = get_partitioner(partition_strategy, num_partition)
    for i in range(0, len(dataset), ingestion_batch_size):
        batch = dataset[i:i + ingestion_batch_size]
        src_nodes = batch["src"].values
        dst_nodes = batch["dst"].values
        timestamps = batch["time"].values
        partitions = partitioner.partition(
            src_nodes, dst_nodes, timestamps)

        print(partitions)
