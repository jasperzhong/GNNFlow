from typing import List
from numpy import int32

import torch
import torch.distributed.rpc as rpc

from dgnn.distributed import graph_services


class KVStoreServer:
    """
    Key-value store server.

    NB: we let local root (i.e., local_rank == 0) to be the server.

    The value can be:
    - node feature (key: "N{node_id}", value: node feature)
    - edge feature (key: "E{edge_id}", value: edge feature)
    - memory (key: "M{node_id}", value: memory)
    """

    def __init__(self):
        # keys -> tensors
        self._node_feat_map = {}
        self._edge_feat_map = {}
        self._memory_map = {}

    def push(self, keys: torch.Tensor, tensors: List[torch.Tensor], mode: str):
        """
        Push tensors to the server.

        Args:
            keys (torch.Tensor): The keys.
            tensors (List[torch.Tensor]): The tensors.
        """
        assert len(keys) == len(
            tensors), "The number of keys and tensors must be the same."

        if mode == 'node':
            for key, tensor in zip(keys, tensors):
                self._node_feat_map[key] = tensor

        if mode == 'edge':
            for key, tensor in zip(keys, tensors):
                self._edge_feat_map[key] = tensor

        if mode == 'memory':
            for key, tensor in zip(keys, tensors):
                self._memory_map[key] = tensor

    def pull(self, keys: torch.Tensor, mode: str) -> List[torch.Tensor]:
        """
        Pull tensors from the server.

        Args:
            keys (torch.Tensor): The keys.

        Returns:
            List[torch.Tensor]: The tensors.
        """
        if mode == 'node':
            return [self._node_feat_map[key] for key in keys]

        if mode == 'edge':
            return [self._edge_feat_map[key] for key in keys]

        if mode == 'memory':
            return [self._memory_map[key] for key in keys]


class KVStoreClient:
    """
    Key-value store client.

    It is used by the trainer to push/pull tensors to/from the KVStore servers.
    """

    def __init__(self, partition_table: torch.Tensor,
                 num_partitions: int32,
                 num_workers_per_machine: int32):
        self._partition_table = partition_table
        self._num_partitions = num_partitions
        self._num_workers_per_machine = num_workers_per_machine

    def push(self, keys: torch.Tensor, tensors: List[torch.Tensor], mode: str):
        """
        Push tensors to the corresponding KVStore servers according to the partition table.

        Args:
            keys (torch.Tensor): The keys.
            tensors (List[torch.Tensor]): The tensors.
        """
        # TODO(guangming): rpc call to the corresponding KVStore servers (call graph_services.push_tensors)
        # dispatch different keys to different partitions
        partition_table = self._partition_table
        # TODO: eid may have some problems
        partition_ids = partition_table[keys]

        futures = []
        for partition_id in range(self._num_partitions):
            partition_mask = partition_ids == partition_id
            if partition_mask.sum() == 0:
                continue
            partition_keys = keys[partition_mask]

            # local rank 0 in those partitions
            worker_rank = partition_id * self._num_workers_per_machine

            futures.append(rpc.rpc_async(
                'worker{}'.format(worker_rank)),
                graph_services.push_tensors,
                args=(partition_keys, tensors, mode))

        # TODO: futures need wait?
        for future in futures:
            future.wait()

    def pull(self, keys: torch.Tensor, mode: str) -> List[torch.Tensor]:
        """
        Pull tensors from the corresponding KVStore servers according to the partition table.

        Args:
            keys (torch.Tensor): The keys.

        Returns:
            List[torch.Tensor]: The tensors.
        """
        # TODO(guangming): rpc call to the corresponding KVStore servers (call graph_services.pull_tensors)
        # dispatch different keys to different partitions
        partition_table = self._partition_table
        # TODO: eid may have some problems
        partition_ids = partition_table[keys]

        futures = []
        for partition_id in range(self._num_partitions):
            partition_mask = partition_ids == partition_id
            if partition_mask.sum() == 0:
                continue
            partition_keys = keys[partition_mask]

            # local rank 0 in those partitions
            worker_rank = partition_id * self._num_workers_per_machine

            futures.append(rpc.rpc_async(
                'worker{}'.format(worker_rank)),
                graph_services.pull_tensors,
                args=(partition_keys, mode))

        # collect pull results
        pull_results = []
        for future in futures:
            pull_results.append(future.wait())

        return pull_results
