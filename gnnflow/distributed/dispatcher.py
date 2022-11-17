import logging
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

    def __init__(self, partition_strategy: str, num_partitions: int):
        self._rank = torch.distributed.get_rank()
        assert self._rank == 0, "Only rank 0 can initialize the dispatcher."
        self._num_partitions = num_partitions
        self._local_world_size = local_world_size()
        self._partitioner = get_partitioner(partition_strategy, num_partitions)

        self._num_edges = 0
        self._max_nodex = 0
        self._nodes = set()

    def dispatch_edges(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                       timestamps: torch.Tensor, eids: torch.Tensor,
                       edge_feats: Optional[torch.Tensor] = None):
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

        partitions = self._partitioner.partition(
            src_nodes, dst_nodes, timestamps, eids)

        self._max_node = self._partitioner._max_node

        # Dispatch the partitions to the workers.
        futures = []
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
                        dim_memory: int = 0):
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
        """
        # Partition the dataset.
        futures = []
        range_list = [0] + \
            list(range(initial_ingestion_batch_size,
                       len(dataset), ingestion_batch_size)) + [len(dataset)]
        t = tqdm(total=len(dataset))
        logging.info("len dataset: {}".format(len(dataset)))
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
            futures.extend(self.dispatch_edges(src_nodes, dst_nodes,
                                               timestamps, eids, edge_feats))

            for future in futures:
                future.wait()
            futures = []
            t.update(len(batch))

        t.close()

        # deal with unpartitioned nodes
        partition_table = self._partitioner._partition_table
        unassigned_nodes_index = (partition_table == -1).nonzero().squeeze()
        logging.info("len of unassigned nodes: {}".format(
            len(unassigned_nodes_index)))
        partition_id = torch.arange(
            len(unassigned_nodes_index), dtype=torch.int8) % self._num_partitions
        partition_table[unassigned_nodes_index] = partition_id

        dim_node = 0 if node_feats is None else node_feats.shape[1]
        dim_edge = 0 if edge_feats is None else edge_feats.shape[1]

        del edge_feats
        self.broadcast_graph_metadata()
        self.broadcast_partition_table()
        self.broadcast_node_edge_dim(dim_node, dim_edge)

        # node feature/memory
        if node_feats is not None:
            futures = []
            for partition_id in reversed(range(self._num_partitions)):
                # the KVStore server is in local_rank 0
                partition_mask = partition_table == partition_id
                assert partition_mask.sum() > 0  # should not be 0
                vertices = torch.arange(len(partition_table), dtype=torch.long)
                partition_vertices = vertices[partition_mask]
                keys = partition_vertices.contiguous()
                kvstore_rank = partition_id * local_world_size()
                if node_feats is not None:
                    features = node_feats[keys]
                    futures.append(rpc.rpc_async("worker%d" % kvstore_rank, graph_services.push_tensors,
                                                 args=(keys, features, 'node')))
                logging.info(
                    "partition: {} dispatch done".format(partition_id))
            mem = psutil.virtual_memory().percent
            logging.info("peak memory usage: {}".format(mem))
            del node_feats

            for future in futures:
                future.wait()

        if dim_memory > 0:
            for partition_id in range(self._num_partitions):
                partition_mask = partition_table == partition_id
                assert partition_mask.sum() > 0  # should not be 0
                vertices = torch.arange(
                    len(partition_table), dtype=torch.long)
                partition_vertices = vertices[partition_mask]
                keys = partition_vertices.contiguous()
                kvstore_rank = partition_id * local_world_size()
                # use None as value and just init keys here.
                memory = torch.zeros(
                    (len(keys), dim_memory), dtype=torch.float32)
                memory_ts = torch.zeros(len(keys), dtype=torch.float32)
                dim_raw_message = 2 * dim_memory + dim_edge
                mailbox = torch.zeros(
                    (len(keys), dim_raw_message), dtype=torch.float32)
                mailbox_ts = torch.zeros(
                    (len(keys), ), dtype=torch.float32)
                all_mem = torch.cat((memory,
                                    memory_ts.unsqueeze(dim=1),
                                    mailbox,
                                    mailbox_ts.unsqueeze(dim=1),
                                     ), dim=1)
                futures.append(rpc.rpc_async("worker%d" % kvstore_rank, graph_services.push_tensors,
                                             args=(keys, all_mem, 'memory')))

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
