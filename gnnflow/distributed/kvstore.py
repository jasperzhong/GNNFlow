import logging
import os
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed.rpc as rpc

from gnnflow.distributed import graph_services
from gnnflow.utils import local_world_size, rank
from libgnnflow import KVStore


class KVStoreServer:
    """
    Key-value store server.

    NB: we let local root (i.e., local_rank == 0) to be the server.

    The value can be:
    - node feature 
    - edge feature
    - memory 
    """

    def __init__(self, node_feat_mmap: Optional[np.memmap] = None, edge_feat_mmap: Optional[np.memmap] = None, dim_memory: int = 0, dim_edge: int = 0):
        self._use_cpp_kvstore = os.environ.get("USE_CPP_KVSTORE", "0") == "1"
        # NB: default is not to use map
        self._not_use_map = os.environ.get("NOT_USE_MAP", "1") == "1"

        self._node_feat_mmap = node_feat_mmap
        self._edge_feat_mmap = edge_feat_mmap
        self._dim_memory = dim_memory
        self._dim_edge = dim_edge
        # keys -> tensors
        if self._use_cpp_kvstore:
            self._node_feat_kvstore = KVStore()
            self._edge_feat_kvstore = KVStore()
            self._memory_kvstore = KVStore()
            logging.info("Use C++ KVStore.")
        elif not self._not_use_map:
            self._node_feat_map = {}
            self._edge_feat_map = {}
            self._memory_map = {}

            self._node_feat_lock = threading.Lock()
            self._edge_feat_lock = threading.Lock()
            self._memory_lock = threading.Lock()
            logging.info("Use Python KVStore.")
        else:
            logging.info("No map.")

            self._node_feat_map = {}
            self._edge_feat = None
            # NB: memory would be updated
            self._memory_map = {}
            self._memory_lock = threading.Lock()

            self._eids = None

    def eid_keys(self) -> torch.Tensor:
        if self._use_cpp_kvstore:
            raise NotImplementedError
        elif not self._not_use_map:
            return torch.tensor(list(self._edge_feat_map.keys()))
        else:
            return self._eids

    def push(self, keys: torch.Tensor, tensors: torch.Tensor, mode: str):
        """
        Push tensors to the server.

        Args:
            keys (torch.Tensor): The keys.
            tensors (List[torch.Tensor]): The tensors.
        """
        assert len(keys) == len(
            tensors), "The number of keys {} and tensors {} must be the same.".format(
            len(keys), len(tensors))

        if self._use_cpp_kvstore:
            keys = keys.tolist()
            if mode == 'node':
                self._node_feat_kvstore.set(keys, tensors)
            elif mode == 'edge':
                self._edge_feat_kvstore.set(keys, tensors)
            elif mode == 'memory':
                self._memory_kvstore.set(keys, tensors)
        elif not self._not_use_map:
            keys = keys.tolist()
            if mode == 'node':
                with self._node_feat_lock:
                    for key, tensor in zip(keys, tensors):
                        self._node_feat_map[key] = tensor
            elif mode == 'edge':
                with self._edge_feat_lock:
                    for key, tensor in zip(keys, tensors):
                        self._edge_feat_map[key] = tensor
            elif mode == 'memory':
                with self._memory_lock:
                    for key, tensor in zip(keys, tensors):
                        self._memory_map[key] = tensor
            else:
                raise ValueError(f"Unknown mode: {mode}")
        else:
            if mode == 'node':
                keys = keys.tolist()
                for key, tensor in zip(keys, tensors):
                    self._node_feat_map[key] = tensor
            elif mode == 'edge':
                # assume new keys are larger
                # sort keys and tensors
                keys, indices = torch.sort(keys)
                tensors = tensors[indices]

                if self._edge_feat is None:
                    self._edge_feat = tensors
                else:
                    self._edge_feat = torch.cat(
                        [self._edge_feat, tensors], dim=0)
                if self._eids is None:
                    self._eids = keys
                else:
                    self._eids = torch.cat([self._eids, keys], dim=0)
            elif mode == 'memory':
                keys = keys.tolist()
                with self._memory_lock:
                    for key, tensor in zip(keys, tensors):
                        self._memory_map[key] = tensor
            else:
                raise ValueError(f"Unknown mode: {mode}")

    def load(self, keys: torch.Tensor, mode: str):
        """
        Load tensors from the mmap file.

        Args:
            keys (torch.Tensor): The keys.
        """
        if mode == 'node' and self._node_feat_mmap is not None:
            # remove seen keys
            keys = keys.tolist()
            keys = [key for key in keys if key not in self._node_feat_map]
            if len(keys) == 0:
                return
            keys = np.array(keys)
            node_feat = self._node_feat_mmap[keys]
            node_feat = torch.from_numpy(node_feat)
            self.push(keys, node_feat, mode)
        elif mode == 'edge' and self._edge_feat_mmap is not None:
            # assume that all keys are unseen
            edge_feat = self._edge_feat_mmap[keys.numpy()]
            edge_feat = torch.from_numpy(edge_feat)
            self.push(keys, edge_feat, mode)
        elif mode == 'memory' and self._dim_memory > 0:
            # remove seen keys
            keys = keys.tolist()
            keys = [key for key in keys if key not in self._node_feat_map]
            if len(keys) == 0:
                return
            keys = np.array(keys)
            memory = torch.zeros(
                (len(keys), self._dim_memory), dtype=torch.float32)
            memory_ts = torch.zeros(len(keys), dtype=torch.float32)
            dim_raw_message = 2 * self._dim_memory + self._dim_edge
            mailbox = torch.zeros(
                (len(keys), dim_raw_message), dtype=torch.float32)
            mailbox_ts = torch.zeros(
                (len(keys), ), dtype=torch.float32)
            all_mem = torch.cat((memory,
                                memory_ts.unsqueeze(dim=1),
                                mailbox,
                                mailbox_ts.unsqueeze(dim=1)), dim=1)
            self.push(keys, all_mem, mode)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def pull(self, keys: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Pull tensors from the server.

        Args:
            keys (torch.Tensor): The keys.

        Returns:
            List[torch.Tensor]: The tensors.
        """
        if self._use_cpp_kvstore:
            keys = keys.tolist()
            if mode == 'node':
                return torch.stack(self._node_feat_kvstore.get(keys))
            elif mode == 'edge':
                return torch.stack(self._edge_feat_kvstore.get(keys))
            elif mode == 'memory':
                return torch.stack(self._memory_kvstore.get(keys))
        elif not self._not_use_map:
            keys = keys.tolist()
            if mode == 'node':
                return torch.stack(list(map(self._node_feat_map.get, keys)))
            elif mode == 'edge':
                return torch.stack(list(map(self._edge_feat_map.get, keys)))
            elif mode == 'memory':
                return torch.stack(list(map(self._memory_map.get, keys)))
            else:
                raise ValueError(f"Unknown mode: {mode}")
        else:
            if mode == 'node':
                keys = keys.tolist()
                return torch.stack(list(map(self._node_feat_map.get, keys)))
            elif mode == 'edge':
                indices = torch.searchsorted(self._eids, keys)
                return self._edge_feat.index_select(0, indices)
            elif mode == 'memory':
                keys = keys.tolist()
                return torch.stack(list(map(self._memory_map.get, keys)))

    def reset_memory(self):
        if self._use_cpp_kvstore:
            self._memory_kvstore.fill_zeros()
        else:
            for mem in iter(self._memory_map.values()):
                mem.fill_(0)


class KVStoreClient:
    """
    Key-value store client.

    It is used by the trainer to push/pull tensors to/from the KVStore servers.
    """

    def __init__(self, partition_table: torch.Tensor,
                 num_partitions: int,
                 num_workers_per_machine: int,
                 local_rank: int,
                 dim_node_feat: int = 0,
                 dim_edge_feat: int = 0,
                 dim_memory: int = 0):
        self._partition_table = partition_table

        self._num_partitions = num_partitions
        self._num_workers_per_machine = num_workers_per_machine
        self._local_rank = local_rank

        self._dim_node_feat = dim_node_feat
        self._dim_edge_feat = dim_edge_feat
        self._dim_memory = dim_memory
        self._dim_mail = dim_memory * 2 + self._dim_edge_feat

        self.comm_time = 0

    def push(self, keys: torch.Tensor, tensors: torch.Tensor, mode: str, nid: Optional[torch.Tensor] = None):
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

        for partition_id in range(self._num_partitions):
            partition_mask = partition_ids == partition_id
            if partition_mask.sum() == 0:
                continue
            partition_keys = keys[partition_mask].clone()
            partition_tensors = tensors[partition_mask].clone()
            # local rank 0 in those partitions
            worker_rank = partition_id * self._num_workers_per_machine

            rpc.rpc_async('worker{}'.format(worker_rank),
                          graph_services.push_tensors, args=(partition_keys, partition_tensors, mode))

    def pull_local(self, keys: torch.Tensor, mode: str, nid: Optional[torch.Tensor] = None):
        """
        Pull tensors from the corresponding KVStore servers according to the partition table.

        Args:
            keys (torch.Tensor): The keys.
            mode (bool): Decide to fetch node/edge features or memory.
            nid (Optional[torch.Tensor]): If fetch edge features,
                use nid to get the partition ids

        Returns:
            torch.Tensor: The tensors.
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
            partition_keys = keys[partition_mask].clone()

            # local rank 0 in those partitions
            worker_rank = partition_id * self._num_workers_per_machine
            futures.append(rpc.rpc_async('worker{}'.format(worker_rank),
                                         graph_services.pull_tensors, args=(partition_keys, mode)))
            masks.append(partition_mask)

        return futures, masks

    def pull_collect(self, futures, masks, mode):
        """
        Pull tensors from the corresponding KVStore servers according to the partition table.

        Args:
            keys (torch.Tensor): The keys.
            mode (bool): Decide to fetch node/edge features or memory.
            nid (Optional[torch.Tensor]): If fetch edge features,
                use nid to get the partition ids

        Returns:
            torch.Tensor: The tensors.
        """
        # collect pull results
        pull_results = []
        for future in futures:
            pull_results.append(future.wait())

        return self._merge_pull_results(pull_results, masks, mode)

    def pull(self, keys: torch.Tensor, mode: str, nid: Optional[torch.Tensor] = None):
        """
        Pull tensors from the corresponding KVStore servers according to the partition table.

        Args:
            keys (torch.Tensor): The keys.
            mode (bool): Decide to fetch node/edge features or memory.
            nid (Optional[torch.Tensor]): If fetch edge features,
                use nid to get the partition ids

        Returns:
            torch.Tensor: The tensors.
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
            partition_keys = keys[partition_mask].clone()

            # local rank 0 in those partitions
            worker_rank = partition_id * self._num_workers_per_machine
            futures.append(rpc.rpc_async('worker{}'.format(worker_rank),
                                         graph_services.pull_tensors, args=(partition_keys, mode)))
            masks.append(partition_mask)

        # collect pull results
        pull_results = []
        for future in futures:
            start = time.time()
            pull_results.append(future.wait())
            end = time.time()
            self.comm_time += end - start

        return self._merge_pull_results(pull_results, masks, mode)

    def init_cache(self, capacity: int) -> Tuple[torch.Tensor, torch.Tensor]:
        global_rank = rank()
        world_size = local_world_size()
        kvstore_rank = (global_rank // world_size) * world_size
        if kvstore_rank == global_rank:
            keys, feats = graph_services.init_cache(capacity)
        else:
            future = rpc.rpc_async('worker{}'.format(
                kvstore_rank), graph_services.init_cache, args=(capacity, ))
            keys, feats = future.wait()
        return keys, feats.float()

    def _merge_pull_results(self, pull_results: List[torch.Tensor], masks: List[torch.Tensor], mode: str):
        """
        Merge pull results from different partitions.

        Args:
            pull_results: pull results from different partitions.
            masks: masks for each partition.
            mode: decide to fetch node/edge features or memory.

        Returns:
            merged pull result.
        """
        assert len(pull_results) > 0
        assert len(pull_results) == len(masks)

        all_pull_results = 0
        for pull_result in pull_results:
            all_pull_results += len(pull_result)

        if mode == "memory":

            all_mem = torch.zeros(
                (all_pull_results, self._dim_memory), dtype=torch.float32)
            all_mem_ts = torch.zeros((all_pull_results,), dtype=torch.float32)
            all_mail = torch.zeros(
                (all_pull_results, self._dim_mail), dtype=torch.float32)
            all_mail_ts = torch.zeros((all_pull_results,), dtype=torch.float32)

            for mask, pull_result in zip(masks, pull_results):
                idx = mask.nonzero().squeeze()
                all_mem[idx] = pull_result[:, :self._dim_memory]
                all_mem_ts[idx] = pull_result[:, self._dim_memory]
                all_mail[idx] = pull_result[:,
                                            self._dim_memory + 1:self._dim_memory + 1 + self._dim_mail]
                all_mail_ts[idx] = pull_result[:, -1]

            return (all_mem, all_mem_ts, all_mail, all_mail_ts)
        else:
            if mode == 'edge':
                dim = self._dim_edge_feat
            else:
                dim = self._dim_node_feat

            all_pull_results = torch.zeros(
                (all_pull_results, dim), dtype=torch.float32)

            for mask, pull_result in zip(masks, pull_results):
                idx = mask.nonzero().squeeze()
                all_pull_results[idx] = pull_result.float()

            return all_pull_results

    def _merge_pull_results_memory(self, pull_results: List[torch.Tensor], masks: List[torch.Tensor]):
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

    # only reset the memory on its machine
    def reset_memory(self):
        if self._local_rank == 0:
            graph_services.reset_memory()
