from typing import Optional

import numpy as np
import pandas as pd
import psutil
import torch
import torch.distributed
import torch.distributed.rpc as rpc
from tqdm import tqdm

import gnnflow.distributed.graph_services as graph_services
from gnnflow.distributed.partition import get_partitioner
from gnnflow.utils import local_world_size

global dispatcher
dispatcher = None


class Dispatcher:
    """
    Dispatch the graph data to the workers.
    """

    def __init__(self, partition_strategy: str, num_partitions: int, dataset_name: str):
        self._rank = torch.distributed.get_rank()
        assert self._rank == 0, "Only rank 0 can initialize the dispatcher."
        self._num_partitions = num_partitions
        self._local_world_size = local_world_size()
        self._partitioner = get_partitioner(
            partition_strategy, num_partitions, self._local_world_size, dataset_name)

        self._num_edges = 0
        self._max_nodex = 0
        self._nodes = set()

    def dispatch_edges(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                       timestamps: torch.Tensor, eids: torch.Tensor,
                       edge_feats: Optional[torch.Tensor] = None,
                       partition_train_data: bool = False,
                       is_initial_ingestion_batch: bool = False):
        """
        Dispatch the edges to the workers.

        Args:
            src_nodes (torch.Tensor): The source nodes of the edges.
            dst_nodes (torch.Tensor): The destination nodes of the edges.
            timestamps (torch.Tensor): The timestamps of the edges.
            eids (torch.Tensor): The edge IDs of the edges.
            edge_feats (torch.Tensor): The edge features of the edges.
            defer_sync (bool): Whether to defer the synchronization.
        """
        self._nodes.update(src_nodes.tolist())
        self._nodes.update(dst_nodes.tolist())
        self._num_edges += torch.unique(eids).size(0)

        partitions, evenly_partitioned_dataset = self._partitioner.partition(
            src_nodes, dst_nodes, timestamps, eids, return_evenly_dataset=partition_train_data, is_initial_ingestion=is_initial_ingestion_batch)

        self._max_node = self._partitioner._max_node

        # Dispatch the partitions to the workers.
        for partition_id, edges in enumerate(partitions):
            edges = list(edges)
            for worker_id in range(self._local_world_size):
                worker_rank = partition_id * self._local_world_size + worker_id
                # TODO: the communication is duplicated for each worker in the remote machine.
                # We can optimize it by sending the data to one of the worker and let it
                # broadcast the data to the other workers.
                if worker_rank == self._rank:
                    graph_services.add_edges(*edges)
                else:
                    rpc.rpc_async("worker%d" % worker_rank, graph_services.add_edges,
                                  args=(*edges, ))

        futures = []
        # Dispatch the training samples to the workers.
        if evenly_partitioned_dataset is not None:
            for partition_id in range(self._num_partitions):
                for worker_id in range(self._local_world_size):
                    worker_rank = partition_id * self._local_world_size + worker_id
                    dataset = evenly_partitioned_dataset[partition_id][worker_id]
                    if worker_rank == self._rank:
                        graph_services.add_train_data(*dataset)
                    else:
                        futures.append(rpc.rpc_async("worker%d" % worker_rank, graph_services.add_train_data,
                                                     args=(*dataset, )))

        for partition_id, edges in enumerate(partitions):
            edges = list(edges)
            # the KVStore server is in local_rank 0
            # TODO: maybe each worker will have a KVStore server
            # node_feats and memory dispatch later using partition table.
            kvstore_rank = partition_id * self._local_world_size
            if edge_feats is not None:
                keys = edges[3]
                features = edge_feats[keys]
                futures.append(rpc.rpc_async("worker%d" % kvstore_rank, graph_services.push_tensors,
                                             args=(keys, features, 'edge')))

        return futures

    def partition_graph(self, dataset: pd.DataFrame, initial_ingestion_batch_size: int,
                        ingestion_batch_size: int, undirected: bool,
                        node_feats: Optional[torch.Tensor] = None,
                        edge_feats: Optional[torch.Tensor] = None,
                        dim_memory: int = 0, partition_train_data: bool = False):
        """
        partition the dataset to the workers.

        Args:
            dataset (df.DataFrame): The dataset to ingest.
            initial_ingestion_batch_size (int): The initial ingestion batch size.
            ingestion_batch_size (int): The number of samples to ingest in each iteration.
            undirected (bool): Whether the graph is undirected.
            node_feats (torch.Tensor): The node features of the dataset.
            edge_feats (torch.Tensor): The edge features of the dataset.
            dim_memory (int): The dimension of the memory.
            partition_train_data (bool): Whether to partition the training data.
        """
        # Partition the dataset.
        futures = []
        range_list = [0] + \
            list(range(initial_ingestion_batch_size,
                       len(dataset), ingestion_batch_size)) + [len(dataset)]
        t = tqdm(total=len(dataset))
        for i in range(len(range_list)-1):
            batch = dataset[range_list[i]:range_list[i+1]]
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

            is_initial_ingestion = False
            if i == 0:
                is_initial_ingestion = True

            futures.extend(self.dispatch_edges(src_nodes, dst_nodes,
                                               timestamps, eids, edge_feats,
                                               partition_train_data,
                                               is_initial_ingestion))

            for future in futures:
                future.wait()
            futures = []
            t.update(len(batch))

        t.close()

    def broadcast_graph_metadata(self):
        """
        Broadcast the graph metadata (i.e., num_nodes, num_edges )to all
        the workers.
        """
        # Broadcast the graph metadata to all the workers.
        for partition_id in range(self._num_partitions):
            for worker_id in range(self._local_world_size):
                worker_rank = partition_id * self._local_world_size + worker_id
                rpc.rpc_sync("worker%d" % worker_rank, graph_services.set_graph_metadata,
                             args=(len(self._nodes), self._num_edges, self._max_node, self._num_partitions))

    def broadcast_partition_table(self):
        """
        Broadcast the partition table to all the workers.
        """
        # Broadcast the partition table to all the workers.
        for partition_id in range(self._num_partitions):
            for worker_id in range(self._local_world_size):
                worker_rank = partition_id * self._local_world_size + worker_id
                rpc.rpc_sync("worker%d" % worker_rank, graph_services.set_partition_table,
                             args=(self._partitioner.get_partition_table(), ))

    def broadcast_node_edge_dim(self, dim_node, dim_edge):
        """
        Broadcast the node_feat/edge_feat dimension to all the workers.
        """
        # Broadcast the dim_node/dim_edge to all the workers.
        for partition_id in range(self._num_partitions):
            for worker_id in range(self._local_world_size):
                worker_rank = partition_id * self._local_world_size + worker_id
                rpc.rpc_sync("worker%d" % worker_rank, graph_services.set_dim_node_edge,
                             args=(dim_node, dim_edge))

    def broadcast_rand_sampler(self, train_rand_sampler, val_rand_sampler, test_rand_sampler):
        for partition_id in range(self._num_partitions):
            for worker_id in range(self._local_world_size):
                worker_rank = partition_id * self._local_world_size + worker_id
                rpc.rpc_sync("worker%d" % worker_rank, graph_services.set_rand_sampler,
                             args=(train_rand_sampler, val_rand_sampler, test_rand_sampler))


def get_dispatcher(partition_strategy: Optional[str] = None, num_partitions: Optional[int] = None, dataset_name: str = None):
    """
    Get the dispatcher singleton.

    Args:
        partition_strategy (str): The partitioning strategy.
        num_partitions (int): The number of partitions to split the dataset into.
        dataset_name (str): the name of the dataset (for loading initial partition)

    Returns:
        The dispatcher singleton.
    """
    global dispatcher
    if dispatcher is None:
        assert partition_strategy is not None and num_partitions is not None, \
            "The dispatcher is not initialized. Please specify the partitioning strategy and the number of partitions."
        dispatcher = Dispatcher(partition_strategy, num_partitions, dataset_name)
    return dispatcher
