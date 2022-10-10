import logging
from typing import List, Optional

import torch
import torch.distributed.rpc as rpc

from gnnflow.distributed import graph_services


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
                self._node_feat_map[int(key)] = tensor
        elif mode == 'edge':
            for key, tensor in zip(keys, tensors):
                self._edge_feat_map[int(key)] = tensor
        elif mode == 'memory':
            for key, tensor in zip(keys, tensors):
                self._memory_map[int(key)] = tensor
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def pull(self, keys: torch.Tensor, mode: str) -> List[torch.Tensor]:
        """
        Pull tensors from the server.

        Args:
            keys (torch.Tensor): The keys.

        Returns:
            List[torch.Tensor]: The tensors.
        """
        if mode == 'node':
            return [self._node_feat_map[int(key)] for key in keys]
        elif mode == 'edge':
            return [self._edge_feat_map[int(key)] for key in keys]
        elif mode == 'memory':
            return [self._memory_map[int(key)] for key in keys]
        else:
            raise ValueError(f"Unknown mode: {mode}")


class KVStoreClient:
    """
    Key-value store client.

    It is used by the trainer to push/pull tensors to/from the KVStore servers.
    """

    def __init__(self, partition_table: torch.Tensor,
                 num_partitions: int,
                 num_workers_per_machine: int):
        self._partition_table = partition_table
        self._num_partitions = num_partitions
        self._num_workers_per_machine = num_workers_per_machine

    def push(self, keys: torch.Tensor, tensors: List[torch.Tensor], mode: str, nid: Optional[torch.Tensor] = None):
        """
        Push tensors to the corresponding KVStore servers according to the partition table.

        Args:
            keys (torch.Tensor): The keys.
            tensors (List[torch.Tensor]): The tensors.
            mode (bool): Decide to push node/edge features or memory.
            nid (Optional[torch.Tensor]): If push edge features,
                use nid to get the partition ids

        """
        # dispatch different keys to different partitions
        partition_table = self._partition_table
        if mode == 'edge':
            if nid is None:
                raise ValueError('Nid is None when pushing edge features')
            # get the partition_ids using nid
            partition_ids = partition_table[nid]
        else:
            partition_ids = partition_table[keys]

        futures = []
        for partition_id in range(self._num_partitions):
            partition_mask = partition_ids == partition_id
            if partition_mask.sum() == 0:
                continue
            partition_keys = keys[partition_mask]

            # local rank 0 in those partitions
            worker_rank = partition_id * self._num_workers_per_machine

            futures.append(rpc.rpc_async('worker{}'.format(worker_rank),
                                         graph_services.push_tensors, args=(partition_keys, tensors, mode)))

        for future in futures:
            future.wait()

    def pull(self, keys: torch.Tensor, mode: str, nid: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Pull tensors from the corresponding KVStore servers according to the partition table.

        Args:
            keys (torch.Tensor): The keys.
            mode (bool): Decide to fetch node/edge features or memory.
            nid (Optional[torch.Tensor]): If fetch edge features,
                use nid to get the partition ids

        Returns:
            List[torch.Tensor]: The tensors.
        """
        # dispatch different keys to different partitions
        partition_table = self._partition_table
        if mode == 'edge':
            if nid is None:
                raise ValueError(
                    'Nid is None when fetching edge features'
                )
            # get the partition_ids using nid
            partition_ids = partition_table[nid]
        else:
            partition_ids = partition_table[keys]

        futures = []
        masks = []
        for partition_id in range(self._num_partitions):
            partition_mask = partition_ids == partition_id
            if partition_mask.sum() == 0:
                continue
            # nid and keys are in the same positions
            partition_keys = keys[partition_mask]

            # local rank 0 in those partitions
            worker_rank = partition_id * self._num_workers_per_machine

            futures.append(rpc.rpc_async('worker{}'.format(worker_rank),
                                         graph_services.pull_tensors, args=(partition_keys, mode)))
            masks.append(partition_mask)

        # collect pull results
        pull_results = []
        for future in futures:
            pull_results.append(future.wait())

        return self._merge_pull_results(pull_results, masks)

    def _merge_pull_results(self, pull_results: List[torch.Tensor], masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge pull results from different partitions.

        Args:
            pull_results: pull results from different partitions.
            masks: masks for each partition.

        Returns:
            merged pull result.
        """
        assert len(pull_results) > 0
        assert len(pull_results) == len(masks)

        all_pull_results = 0
        for pull_result in pull_results:
            all_pull_results += len(pull_result)

        logging.info("pull results: {}".format(pull_results))
        all_pull_results = torch.zeros(
            (all_pull_results, pull_results[0].shape[1]), dtype=torch.float32)

        for mask, pull_result in (masks, pull_results):
            idx = mask.nonzero().squeeze()
            all_pull_results[idx] = pull_result

        return all_pull_results
