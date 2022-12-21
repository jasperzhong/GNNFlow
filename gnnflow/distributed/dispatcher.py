from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed
import torch.distributed.rpc as rpc

import gnnflow.distributed.graph_services as graph_services
from gnnflow.distributed.partition import get_partitioner
from gnnflow.utils import local_world_size

global dispatcher
dispatcher = None


class Dispatcher:
    """
    Dispatch the graph data to the workers.
    """

    def __init__(self, partition_strategy: str, num_partitions: int, node_feat: bool, edge_feat: bool,
                 memory: bool, dataset_name: str):
        self._rank = torch.distributed.get_rank()
        assert self._rank == 0, "Only rank 0 can initialize the dispatcher."
        self._num_partitions = num_partitions
        self._local_world_size = local_world_size()
        self._partitioner = get_partitioner(
            partition_strategy, num_partitions, self._local_world_size, dataset_name)

        self._num_edges = 0
        self._max_nodex = 0

        self._node_feat = node_feat
        self._edge_feat = edge_feat
        self._memory = memory

        self._train_dst_set = set()
        self._nontrain_dst_set = set()

    def dispatch_edges(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                       timestamps: torch.Tensor, eids: torch.Tensor,
                       partition_train_data: bool = False):
        """
        Dispatch the edges to the workers.

        Args:
            src_nodes (torch.Tensor): The source nodes of the edges.
            dst_nodes (torch.Tensor): The destination nodes of the edges.
            timestamps (torch.Tensor): The timestamps of the edges.
            eids (torch.Tensor): The edge IDs of the edges.
            partition_train_data (bool): Whether to partition the training data.
        """
        self._num_edges += torch.unique(eids).size(0)

        partitions, evenly_partitioned_dataset = self._partitioner.partition(
            src_nodes, dst_nodes, timestamps, eids, return_evenly_dataset=partition_train_data)

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

        if self._edge_feat:
            for partition_id, edges in enumerate(partitions):
                edges = list(edges)
                # the KVStore server is in local_rank 0
                # TODO: maybe each worker will have a KVStore server
                # node_feats and memory dispatch later using partition table.
                kvstore_rank = partition_id * self._local_world_size
                keys = edges[3]
                futures.append(rpc.rpc_async("worker%d" % kvstore_rank, graph_services.load_tensors,
                                             args=(keys, 'edge')))

        for future in futures:
            future.wait()

    def partition_graph(self, dataset: pd.DataFrame, dispatch_node_memory: bool = False):
        """
        partition the dataset to the workers.

        Args:
            dataset (df.DataFrame): The dataset to ingest.
            node_feat (bool): Whether to partition the node features.
            edge_feat (bool): Whether to partition the edge features.
            memory (bool): Whether to partition the memory.
        """
        # Partition the dataset.
        src_nodes = torch.from_numpy(dataset["src"].values.astype(np.int64))
        dst_nodes = torch.from_numpy(dataset["dst"].values.astype(np.int64))
        timestamps = torch.from_numpy(dataset["time"].values.astype(np.float32))
        eids = torch.from_numpy(dataset["eid"].values.astype(np.int64))
        train_end = dataset['ext_roll'].values.searchsorted(1)

        # dispatch edges
        src_nodes_train = src_nodes[:train_end]
        dst_nodes_train = dst_nodes[:train_end]
        timestamps_train = timestamps[:train_end]
        eids_train = eids[:train_end]

        if len(src_nodes_train) > 0:
            self._train_dst_set.update(dst_nodes_train.tolist())
            self.dispatch_edges(src_nodes_train, dst_nodes_train,
                                timestamps_train, eids_train, partition_train_data=True)

        src_nodes_nontrain = src_nodes[train_end:]
        dst_nodes_nontrain = dst_nodes[train_end:]
        timestamps_nontrain = timestamps[train_end:]
        eids_nontrain = eids[train_end:]

        if len(src_nodes_nontrain) > 0:
            self._nontrain_dst_set.update(dst_nodes_nontrain.tolist())
            self.dispatch_edges(src_nodes_nontrain, dst_nodes_nontrain,
                                timestamps_nontrain, eids_nontrain, partition_train_data=False)

        # deal with unpartitioned nodes
        partition_table = self._partitioner._partition_table
        unassigned_nodes_index = (
            partition_table == -1).nonzero().squeeze(dim=1)

        if len(unassigned_nodes_index) > 0:
            partition_id = torch.arange(
                len(unassigned_nodes_index), dtype=torch.int8) % self._num_partitions
            partition_table[unassigned_nodes_index] = partition_id

        self.broadcast_graph_metadata()
        self.broadcast_partition_table()

        if dispatch_node_memory:
            self.dispatch_node_memory()

    def dispatch_node_memory(self):
        partition_table = self._partitioner._partition_table
        # dispatch node feature/memory
        futures = []
        if self._node_feat:
            for partition_id in range(self._num_partitions):
                # the KVStore server is in local_rank 0
                partition_mask = partition_table == partition_id
                assert partition_mask.sum() > 0  # should not be 0
                vertices = torch.arange(len(partition_table), dtype=torch.long)
                partition_vertices = vertices[partition_mask]
                keys = partition_vertices
                kvstore_rank = partition_id * local_world_size()
                futures.append(rpc.rpc_async("worker%d" % kvstore_rank, graph_services.load_tensors,
                                             args=(keys, 'node')))

        if self._memory:
            futures = []
            for partition_id in range(self._num_partitions):
                partition_mask = partition_table == partition_id
                assert partition_mask.sum() > 0  # should not be 0
                vertices = torch.arange(
                    len(partition_table), dtype=torch.long)
                partition_vertices = vertices[partition_mask]
                keys = partition_vertices.contiguous()
                kvstore_rank = partition_id * local_world_size()
                futures.append(rpc.rpc_async("worker%d" % kvstore_rank, graph_services.load_tensors,
                                             args=(keys, 'memory')))

        for future in futures:
            future.wait()

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
                             args=(self._max_node, self._num_edges, self._max_node, self._num_partitions))

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

    def broadcast_rand_sampler(self):
        train_dst_set = torch.LongTensor(list(self._train_dst_set))
        nontrain_dst_set = torch.LongTensor(list(self._nontrain_dst_set))
        futures = []
        for partition_id in range(self._num_partitions):
            for worker_id in range(self._local_world_size):
                worker_rank = partition_id * self._local_world_size + worker_id
                if worker_rank == self._rank:
                    graph_services.set_rand_sampler(
                        train_dst_set, nontrain_dst_set)
                else:
                    futures.append(rpc.rpc_async("worker%d" % worker_rank, graph_services.set_rand_sampler,
                                                 args=(train_dst_set, nontrain_dst_set)))

        for future in futures:
            future.wait()


def get_dispatcher(partition_strategy: Optional[str] = None, num_partitions: Optional[int] = None,
                   node_feat: bool = False, edge_feat: bool = False, memory: bool = False, dataset_name: str = ""):
    """
    Get the dispatcher singleton.

    Args:
        partition_strategy (str): The partitioning strategy.
        num_partitions (int): The number of partitions to split the dataset into.
        node_feat (bool): Whether to load node features.
        edge_feat (bool): Whether to load edge features.
        memory (bool): Whether to load memory.

    Returns:
        The dispatcher singleton.
    """
    global dispatcher
    if dispatcher is None:
        dispatcher = Dispatcher(
            partition_strategy, num_partitions, node_feat, edge_feat, memory, dataset_name)
    return dispatcher
