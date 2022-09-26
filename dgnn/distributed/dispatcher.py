from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed
import torch.distributed.rpc as rpc

import dgnn.distributed.graph_services as graph_services
from dgnn.distributed.partition import get_partitioner

global dispatcher
dispatcher = None


class Dispatcher:
    """
    Dispatch the graph data to the workers.
    """

    def __init__(self, partition_strategy: str, num_partitions: int):
        self._rank = torch.distributed.get_rank()
        assert self._rank == 0, "Only rank 0 can initialize the dispatcher."
        self._num_partitions = num_partitions
        self._num_workers_per_machine = torch.cuda.device_count()
        self._partitioner = get_partitioner(partition_strategy, num_partitions)

        self._num_nodes = 0
        self._num_edges = 0

    def dispatch_edges(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                       timestamps: torch.Tensor, eids: torch.Tensor, defer_sync: bool = False):
        partitions = self._partitioner.partition(
            src_nodes, dst_nodes, timestamps, eids)

        # Dispatch the partitions to the workers.
        futures = []
        for partition_id, edges in enumerate(partitions):
            edges = list(edges)
            for worker_id in range(self._num_workers_per_machine):
                worker_rank = partition_id * self._num_workers_per_machine + worker_id
                # TODO: the communication is duplicated for each worker in the remote machine.
                # We can optimize it by sending the data to one of the worker and let it
                # broadcast the data to the other workers.
                future = rpc.rpc_async("worker%d" % worker_rank, graph_services.add_edges,
                                       args=(edges))
                futures.append(future)

        if not defer_sync:
            # Wait for the workers to finish.
            for future in futures:
                future.wait()

            self.broadcast_graph_metadata()
        return futures

    def partition_graph(self, dataset: pd.DataFrame, ingestion_batch_size: int,
                        undirected: bool):
        """
        partition the dataset to the workers.

        Args:
            dataset (df.DataFrame): The dataset to ingest.
            ingestion_batch_size (int): The number of samples to ingest in each iteration.
            partition_strategy (str): The partitioning strategy.
            num_partitions (int): The number of partitions to split the dataset into.
            undirected (bool): Whether the graph is undirected.
        """
        # Partition the dataset.
        futures = []
        for i in range(0, len(dataset), ingestion_batch_size):
            batch = dataset[i:i + ingestion_batch_size]
            src_nodes = batch["src"].values.astype(np.int64)
            dst_nodes = batch["dst"].values.astype(np.int64)
            timestamps = batch["time"].values.astype(np.float32)
            eids = batch["eid"].values.astype(np.int64)

            self._num_nodes = len(
                np.unique(np.concatenate([src_nodes, dst_nodes])))
            self._num_edges += len(eids)

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

            futures.extend(self.dispatch_edges(src_nodes, dst_nodes,
                           timestamps, eids, defer_sync=True))

            # Wait for the workers to finish.
            for future in futures:
                future.wait()
            futures = []
        self.broadcast_graph_metadata()

    def broadcast_graph_metadata(self):
        """
        Broadcast the graph metadata (i.e., num_nodes, num_edges )to all 
        the workers.
        """
        # Broadcast the graph metadata to all the workers.
        for partition_id in range(self._num_partitions):
            for worker_id in range(self._num_workers_per_machine):
                worker_rank = partition_id * self._num_workers_per_machine + worker_id
                rpc.rpc_sync("worker%d" % worker_rank, graph_services.set_graph_metadata,
                             args=(self._num_nodes, self._num_edges))


def get_dispatcher(partition_strategy: Optional[str] = None, num_partitions: Optional[int] = None):
    """
    Get the dispatcher singleton.

    Args:
        partition_strategy (str): The partitioning strategy.
        num_partitions (int): The number of partitions to split the dataset into.

    Returns:
        The dispatcher singleton.
    """
    global dispatcher
    if dispatcher is None:
        assert partition_strategy is not None and num_partitions is not None, \
            "The dispatcher is not initialized. Please specify the partitioning strategy and the number of partitions."
        dispatcher = Dispatcher(partition_strategy, num_partitions)
    return dispatcher
