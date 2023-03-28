import logging
from queue import Queue
# import torch.multiprocessing as mp
from typing import List, Optional, Union

import numpy as np
import torch
from dgl.heterograph import DGLBlock

from gnnflow.distributed.kvstore import KVStoreClient


class Cache:
    """
    Feature cache on GPU
    """

    def __init__(self, edge_cache_ratio: int, node_cache_ratio: int,
                 num_nodes: int, num_edges: int,
                 device: Union[str, torch.device],
                 node_feats: Optional[torch.Tensor] = None,
                 edge_feats: Optional[torch.Tensor] = None,
                 dim_node_feat: Optional[int] = 0,
                 dim_edge_feat: Optional[int] = 0,
                 pinned_nfeat_buffs: Optional[torch.Tensor] = None,
                 pinned_efeat_buffs: Optional[torch.Tensor] = None,
                 kvstore_client: Optional[KVStoreClient] = None,
                 distributed: Optional[bool] = False,
                 neg_sample_ratio: Optional[int] = 1):
        """
        Initialize the cache

        Args:
            edge_cache_ratio: The edge ratio of the cache size to the total number of nodes or edges
                    range: [0, 1].
            node_cache_ratio: The node ratio of the cache size to the total number of nodes or edges
                    range: [0, 1].
            num_nodes: The number of nodes in the graph
            num_edges: The number of edges in the graph
            device: The device to use
            node_feats: The node features
            edge_feats: The edge features
            dim_node_feat: The dimension of node features
            dim_edge_feat: The dimension of edge features
            pinned_nfeat_buffs: The pinned memory buffers for node features
            pinned_efeat_buffs: The pinned memory buffers for edge features
            kvstore_client: The KVStore_Client for fetching features when using distributed
                    training
            distributed: Whether to use distributed training
            neg_sample_ratio: The ratio of negative samples to positive samples
        """
        if device == 'cpu' or device == torch.device('cpu'):
            raise ValueError('Cache must be on GPU')

        if node_feats is None and edge_feats is None and not distributed:
            raise ValueError(
                'At least one of node_feats and edge_feats must be provided')

        if node_feats is not None and node_feats.shape[0] != num_nodes:
            raise ValueError(
                'The number of nodes in node_feats {} does not match num_nodes {}'.format(
                    node_feats.shape[0], num_nodes))

        if edge_feats is not None and edge_feats.shape[0] != num_edges:
            raise ValueError(
                'The number of edges in edge_feats {} does not match num_edges {}'.format(
                    edge_feats.shape[0], num_edges))

        if distributed:
            assert kvstore_client is not None, 'kvstore_client must be provided when using ' \
                'distributed training'
            assert neg_sample_ratio > 0, 'neg_sample_ratio must be positive'
        else:
            if node_feats is not None and node_feats.dtype == torch.bool:
                node_feats = node_feats.to(torch.float32)
            if edge_feats is not None and edge_feats.dtype == torch.bool:
                edge_feats = edge_feats.to(torch.float32)

        # NB: cache_ratio == 0 means no cache
        assert edge_cache_ratio >= 0 and edge_cache_ratio <= 1, 'edge_cache_ratio must be in [0, 1]'
        assert edge_cache_ratio >= 0 and edge_cache_ratio <= 1, 'node_cache_ratio must be in [0, 1]'

        self.edge_cache_ratio = edge_cache_ratio
        self.node_cache_ratio = node_cache_ratio
        self.node_capacity = int(node_cache_ratio * num_nodes)
        self.edge_capacity = int(edge_cache_ratio * num_edges)

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.device = device
        self.pinned_nfeat_buffs = pinned_nfeat_buffs
        self.pinned_efeat_buffs = pinned_efeat_buffs

        self.cache_node_ratio = 0
        self.cache_edge_ratio = 0

        self.kvstore_client = kvstore_client
        # we can add a flag to indicate whether it's distributed
        # so that we can only use one fetch feature function
        self.distributed = distributed
        self.target_edge_features = None
        # TODO: use queue
        # self.target_edge_features = Queue(maxsize=10)

        # used to extract src_node id of input eid
        self.neg_sample_ratio = neg_sample_ratio

        # stores node's features
        if self.dim_node_feat != 0:
            self.cache_node_buffer = torch.zeros(
                self.node_capacity, self.dim_node_feat, dtype=torch.float32, device=self.device)

            # flag for indicating those cached nodes
            self.cache_node_flag = torch.zeros(
                num_nodes, dtype=torch.bool, device=self.device)
            # maps node id -> index
            self.cache_node_map = torch.zeros(
                num_nodes, dtype=torch.int64, device=self.device) - 1
            # maps index -> node id
            self.cache_index_to_node_id = torch.zeros(
                self.node_capacity, dtype=torch.int64, device=self.device) - 1

        if self.dim_edge_feat != 0:
            self.cache_edge_buffer = torch.zeros(
                self.edge_capacity, self.dim_edge_feat, dtype=torch.float32, device=self.device)

            # flag for indicating those cached edges
            self.cache_edge_flag = torch.zeros(
                num_edges, dtype=torch.bool, device=self.device)
            # maps edge id -> index
            self.cache_edge_map = torch.zeros(
                num_edges, dtype=torch.int64, device=self.device) - 1
            # maps index -> edge id
            self.cache_index_to_edge_id = torch.zeros(
                self.edge_capacity, dtype=torch.int64, device=self.device) - 1

    def get_mem_size(self) -> int:
        """
        Get the memory size of the cache in bytes
        """
        mem_size = 0
        if self.dim_node_feat != 0:
            mem_size += self.cache_node_buffer.element_size() * self.cache_node_buffer.nelement()
            mem_size += self.cache_node_flag.element_size() * self.cache_node_flag.nelement()
            mem_size += self.cache_node_map.element_size() * self.cache_node_map.nelement()
            mem_size += self.cache_index_to_node_id.element_size() * \
                self.cache_index_to_node_id.nelement()

        if self.dim_edge_feat != 0:
            mem_size += self.cache_edge_buffer.element_size() * self.cache_edge_buffer.nelement()
            mem_size += self.cache_edge_flag.element_size() * self.cache_edge_flag.nelement()
            mem_size += self.cache_edge_map.element_size() * self.cache_edge_map.nelement()
            mem_size += self.cache_index_to_edge_id.element_size() * \
                self.cache_index_to_edge_id.nelement()

        return mem_size

    def init_cache(self, *args, **kwargs):
        """
        Init the cache with features
        """
        if self.distributed:
            # the edge map is ordered my insertion order, which is the order of ts
            if self.dim_edge_feat != 0 and self.edge_capacity > 0:
                keys, feats = self.kvstore_client.init_cache(
                    self.edge_capacity)
                cache_edge_id = torch.arange(
                    len(keys), dtype=torch.int64, device=self.device)
                self.cache_edge_buffer[cache_edge_id] = feats.to(
                    self.device).float()
                self.cache_edge_flag[cache_edge_id] = True
                self.cache_index_to_edge_id[cache_edge_id] = keys.to(
                    self.device)
                self.cache_edge_map[keys] = cache_edge_id
        else:
            if self.dim_node_feat != 0:
                cache_node_id = torch.arange(
                    self.node_capacity, dtype=torch.int64, device=self.device)

                # Init parameters related to feature fetching
                self.cache_node_buffer[cache_node_id] = self.node_feats[:self.node_capacity].to(
                    self.device, non_blocking=True)
                self.cache_node_flag[cache_node_id] = True
                self.cache_index_to_node_id = cache_node_id
                self.cache_node_map[cache_node_id] = cache_node_id

            if self.dim_edge_feat != 0:
                cache_edge_id = torch.arange(
                    self.edge_capacity, dtype=torch.int64, device=self.device)

                # Init parameters related to feature fetching
                self.cache_edge_buffer[cache_edge_id] = self.edge_feats[:self.edge_capacity].to(
                    self.device, non_blocking=True)
                self.cache_edge_flag[cache_edge_id] = True
                self.cache_index_to_edge_id = cache_edge_id
                self.cache_edge_map[cache_edge_id] = cache_edge_id

    def resize(self, new_num_nodes: int, new_num_edges: int):
        """
        Resize the cache

        Args:
            new_num_nodes: The new number of nodes
            new_num_edges: The new number of edges
        """
        if self.dim_node_feat != 0 and new_num_nodes > self.num_nodes:
            self.num_nodes = new_num_nodes
            self.node_capacity = int(self.node_cache_ratio * self.num_nodes)
            self.cache_node_buffer.resize_(
                self.node_capacity, self.dim_node_feat)
            self.cache_node_flag.resize_(self.num_nodes)
            self.cache_node_map.resize_(self.num_nodes)
            self.cache_index_to_node_id.resize_(self.node_capacity)

        if self.dim_edge_feat != 0 and new_num_edges > self.num_edges:
            self.num_edges = new_num_edges
            self.edge_capacity = int(self.edge_cache_ratio * self.num_edges)
            self.cache_edge_buffer.resize_(
                self.edge_capacity, self.dim_edge_feat)
            self.cache_edge_flag.resize_(self.num_edges)
            self.cache_edge_map.resize_(self.num_edges)
            self.cache_index_to_edge_id.resize_(self.edge_capacity)

    def reset(self):
        """
        Reset the cache
        """
        raise NotImplementedError

    def update_node_cache(self, cached_node_index: torch.Tensor,
                          uncached_node_id: torch.Tensor,
                          uncached_node_feature: torch.Tensor):
        """
        Update the node cache

        Args:
            cached_node_index: The index of the cached nodes
            uncached_node_id: The id of the uncached nodes
            uncached_node_feature: The features of the uncached nodes
        """
        raise NotImplementedError

    def update_edge_cache(self, cached_edge_index: torch.Tensor,
                          uncached_edge_id: torch.Tensor,
                          uncached_edge_feature: torch.Tensor):
        """
        Update the edge cache

        Args:
            cached_edge_index: The index of the cached edges
            uncached_edge_id: The id of the uncached edges
            uncached_edge_feature: The features of the uncached edges
        """
        raise NotImplementedError

    def fetch_feature(self, mfgs: List[List[DGLBlock]],
                      eid: Optional[np.ndarray] = None, update_cache: bool = True,
                      target_edge_features: bool = True):
        """Fetching the node/edge features of input_node_ids

        Args:
            mfgs: message-passing flow graphs
            eid: target edge ids
            update_cache: whether to update the cache
            target_edge_features: whether to fetch target edge features for TGN

        Returns:
            mfgs: message-passing flow graphs with node/edge features
        """
        if self.dim_node_feat != 0:
            i = 0
            hit_ratio_sum = 0
            for b in mfgs[0]:
                nodes = b.srcdata['ID']
                assert isinstance(nodes, torch.Tensor)
                cache_mask = self.cache_node_flag[nodes]

                hit_ratio = torch.sum(cache_mask) / len(nodes)
                hit_ratio_sum += hit_ratio

                node_feature = torch.zeros(
                    len(nodes), self.dim_node_feat, dtype=torch.float32, device=self.device)

                # fetch the cached features
                cached_node_index = self.cache_node_map[nodes[cache_mask]]
                node_feature[cache_mask] = self.cache_node_buffer[cached_node_index]
                # fetch the uncached features
                uncached_mask = ~cache_mask
                if uncached_mask.sum() > 0:
                    uncached_node_id = nodes[uncached_mask]
                    uncached_node_id_unique, uncached_node_id_unique_index = torch.unique(
                        uncached_node_id, return_inverse=True)

                    if self.distributed:
                        if self.pinned_nfeat_buffs is not None:
                            # TODO: maybe fetch local and remote features separately
                            self.pinned_nfeat_buffs[
                                i][:uncached_node_id_unique.shape[0]] = self.kvstore_client.pull(
                                uncached_node_id_unique.cpu(), mode='node')
                            uncached_node_feature = self.pinned_nfeat_buffs[i][:uncached_node_id_unique.shape[0]].to(
                                self.device, non_blocking=True).float()
                        else:
                            uncached_node_feature = self.kvstore_client.pull(
                                uncached_node_id_unique.cpu(), mode='node').to(self.device).float()
                    else:
                        if self.pinned_nfeat_buffs is not None:
                            torch.index_select(self.node_feats, 0, uncached_node_id_unique.to('cpu'),
                                               out=self.pinned_nfeat_buffs[i][:uncached_node_id_unique.shape[0]])
                            uncached_node_feature = self.pinned_nfeat_buffs[i][:uncached_node_id_unique.shape[0]].to(
                                self.device, non_blocking=True)
                        else:
                            uncached_node_feature = self.node_feats[uncached_node_id_unique.cpu()].to(
                                self.device, non_blocking=True)
                    node_feature[uncached_mask] = uncached_node_feature[uncached_node_id_unique_index]

                i += 1
                b.srcdata['h'] = node_feature

                if update_cache and uncached_mask.sum() > 0:
                    self.update_node_cache(cached_node_index=cached_node_index,
                                           uncached_node_id=uncached_node_id_unique,
                                           uncached_node_feature=uncached_node_feature)

            self.cache_node_ratio = hit_ratio_sum / i if i > 0 else 0

        # Edge feature
        if self.dim_edge_feat != 0:
            i = 0
            hit_ratio_sum = 0
            for mfg in mfgs:
                for b in mfg:
                    edges = b.edata['ID']
                    assert isinstance(edges, torch.Tensor)
                    if len(edges) == 0:
                        continue

                    cache_mask = self.cache_edge_flag[edges]
                    hit_ratio = torch.sum(cache_mask) / len(edges)
                    hit_ratio_sum += hit_ratio

                    edge_feature = torch.zeros(len(edges), self.dim_edge_feat,
                                               dtype=torch.float32, device=self.device)

                    # fetch the cached features
                    cached_edge_index = self.cache_edge_map[edges[cache_mask]]
                    edge_feature[cache_mask] = self.cache_edge_buffer[cached_edge_index]
                    # fetch the uncached features
                    uncached_mask = ~cache_mask
                    uncached_edge_id = edges[uncached_mask]

                    if len(uncached_edge_id) > 0:
                        if self.distributed:
                            # edge_features need to convert to nid first.
                            # get the first_indices of the origin tensor in unique
                            # pytorch should have this option like numpy !!
                            uncached_edge_id_unique, uncached_edge_id_unique_index, counts = torch.unique(
                                uncached_edge_id, return_inverse=True, return_counts=True)
                            _, ind_sorted = torch.sort(
                                uncached_edge_id_unique_index, stable=True)
                            cum_sum = counts.cumsum(0)
                            cum_sum = torch.cat(
                                (torch.tensor([0]).cuda(), cum_sum[:-1]))
                            first_indicies = ind_sorted[cum_sum]

                            src_nid = b.srcdata['ID'][b.edges()[1]]
                            uncached_eid_to_nid = src_nid[uncached_mask]
                            # use the same indices as eid when unique
                            uncached_eid_to_nid_unique = uncached_eid_to_nid[first_indicies].cpu(
                            )
                            if self.pinned_efeat_buffs is not None:
                                self.pinned_efeat_buffs[
                                    i][:uncached_edge_id_unique.shape[0]] = self.kvstore_client.pull(
                                        uncached_edge_id_unique.cpu(), mode='edge', nid=uncached_eid_to_nid_unique)
                                uncached_edge_feature = self.pinned_efeat_buffs[i][:uncached_edge_id_unique.shape[0]].to(
                                    self.device, non_blocking=True).float()
                            else:
                                uncached_edge_feature = self.kvstore_client.pull(
                                    uncached_edge_id_unique.cpu(), mode='edge', nid=uncached_eid_to_nid_unique).to(self.device).float()
                        else:
                            uncached_edge_id_unique, uncached_edge_id_unique_index = torch.unique(
                                uncached_edge_id, return_inverse=True)
                            if self.pinned_efeat_buffs is not None:
                                torch.index_select(self.edge_feats, 0, uncached_edge_id_unique.to('cpu'),
                                                   out=self.pinned_efeat_buffs[i][:uncached_edge_id_unique.shape[0]])
                                uncached_edge_feature = self.pinned_efeat_buffs[i][:uncached_edge_id_unique.shape[0]].to(
                                    self.device, non_blocking=True)
                            else:

                                uncached_edge_feature = self.edge_feats[uncached_edge_id_unique.cpu()].to(
                                    self.device, non_blocking=True)

                        edge_feature[uncached_mask] = uncached_edge_feature[uncached_edge_id_unique_index]

                    i += 1
                    b.edata['f'] = edge_feature

                    if update_cache and len(uncached_edge_id) > 0:
                        self.update_edge_cache(cached_edge_index=cached_edge_index,
                                               uncached_edge_id=uncached_edge_id_unique,
                                               uncached_edge_feature=uncached_edge_feature)

            self.cache_edge_ratio = hit_ratio_sum / i if i > 0 else 0

            if target_edge_features:
                if self.distributed:
                    # TODO: maybe there are some edge_features is in the memory now
                    num_edges = mfgs[-1][0].num_dst_nodes() // (
                        self.neg_sample_ratio + 2)
                    nid = mfgs[-1][0].srcdata['ID'][:num_edges]
                    # self.target_edge_features.put(self.kvstore_client.pull(
                    #     torch.from_numpy(eid), mode='edge', nid=nid).float())
                    self.target_edge_features = self.kvstore_client.pull(
                        torch.from_numpy(eid), mode='edge', nid=nid).float()
                else:
                    self.target_edge_features = self.edge_feats[eid]
                    # self.target_edge_features.put(self.edge_feats[eid])

        return mfgs

    def fetch_feature_local(self, mfgs: List[List[DGLBlock]],
                            eid: Optional[np.ndarray] = None, update_cache: bool = True,
                            target_edge_features: bool = True):
        """Fetching the node/edge features of input_node_ids

        Args:
            mfgs: message-passing flow graphs
            eid: target edge ids
            update_cache: whether to update the cache
            target_edge_features: whether to fetch target edge features for TGN

        Returns:
            mfgs: message-passing flow graphs with node/edge features
        """
        node_futures = []
        if self.dim_node_feat != 0:
            i = 0
            hit_ratio_sum = 0
            for b in mfgs[0]:
                nodes = b.srcdata['ID']
                assert isinstance(nodes, torch.Tensor)
                cache_mask = self.cache_node_flag[nodes]

                hit_ratio = torch.sum(cache_mask) / len(nodes)
                hit_ratio_sum += hit_ratio

                node_feature = torch.zeros(
                    len(nodes), self.dim_node_feat, dtype=torch.float32, device=self.device)

                # fetch the cached features
                cached_node_index = self.cache_node_map[nodes[cache_mask]]
                node_feature[cache_mask] = self.cache_node_buffer[cached_node_index]
                # fetch the uncached features
                uncached_mask = ~cache_mask
                if uncached_mask.sum() > 0:
                    uncached_node_id = nodes[uncached_mask]
                    uncached_node_id_unique, uncached_node_id_unique_index = torch.unique(
                        uncached_node_id, return_inverse=True)

                    if self.distributed:
                        future = self.kvstore_client.pull_local(
                            uncached_node_id_unique.cpu(), mode='node')
                        node_futures.append(future)

        # Edge feature
        edge_futures = []
        if self.dim_edge_feat != 0:
            i = 0
            hit_ratio_sum = 0
            for mfg in mfgs:
                for b in mfg:
                    edges = b.edata['ID']
                    assert isinstance(edges, torch.Tensor)
                    if len(edges) == 0:
                        continue

                    cache_mask = self.cache_edge_flag[edges]
                    hit_ratio = torch.sum(cache_mask) / len(edges)
                    hit_ratio_sum += hit_ratio

                    edge_feature = torch.zeros(len(edges), self.dim_edge_feat,
                                               dtype=torch.float32, device=self.device)

                    # fetch the cached features
                    cached_edge_index = self.cache_edge_map[edges[cache_mask]]
                    edge_feature[cache_mask] = self.cache_edge_buffer[cached_edge_index]
                    # fetch the uncached features
                    uncached_mask = ~cache_mask
                    uncached_edge_id = edges[uncached_mask]

                    if len(uncached_edge_id) > 0:
                        if self.distributed:
                            # edge_features need to convert to nid first.
                            # get the first_indices of the origin tensor in unique
                            # pytorch should have this option like numpy !!
                            uncached_edge_id_unique, uncached_edge_id_unique_index, counts = torch.unique(
                                uncached_edge_id, return_inverse=True, return_counts=True)
                            _, ind_sorted = torch.sort(
                                uncached_edge_id_unique_index, stable=True)
                            cum_sum = counts.cumsum(0)
                            cum_sum = torch.cat(
                                (torch.tensor([0]).cuda(), cum_sum[:-1]))
                            first_indicies = ind_sorted[cum_sum]

                            src_nid = b.srcdata['ID'][b.edges()[1]]
                            uncached_eid_to_nid = src_nid[uncached_mask]
                            # use the same indices as eid when unique
                            uncached_eid_to_nid_unique = uncached_eid_to_nid[first_indicies].cpu(
                            )

                            future = self.kvstore_client.pull_local(
                                uncached_edge_id_unique.cpu(), mode='edge', nid=uncached_eid_to_nid_unique)
                            edge_futures.append(future)

            target_edge_futures = None
            if target_edge_features:
                if self.distributed:
                    num_edges = mfgs[-1][0].num_dst_nodes() // (
                        self.neg_sample_ratio + 2)
                    nid = mfgs[-1][0].srcdata['ID'][:num_edges]
                    target_edge_futures = self.kvstore_client.pull_local(
                        torch.from_numpy(eid), mode='edge', nid=nid)

        return node_futures, edge_futures, target_edge_futures

    def fetch_feature_collect(self, node_futures, edge_futures, target_egde_futures, mfgs: List[List[DGLBlock]],
                              eid: Optional[np.ndarray] = None, update_cache: bool = True,
                              target_edge_features: bool = True):
        """Fetching the node/edge features of input_node_ids

        Args:
            mfgs: message-passing flow graphs
            eid: target edge ids
            update_cache: whether to update the cache
            target_edge_features: whether to fetch target edge features for TGN

        Returns:
            mfgs: message-passing flow graphs with node/edge features
        """

        if self.dim_node_feat != 0:
            i = 0
            hit_ratio_sum = 0
            for b, (futures, masks) in zip(mfgs[0], node_futures):
                nodes = b.srcdata['ID']
                assert isinstance(nodes, torch.Tensor)
                cache_mask = self.cache_node_flag[nodes]

                hit_ratio = torch.sum(cache_mask) / len(nodes)
                hit_ratio_sum += hit_ratio

                node_feature = torch.zeros(
                    len(nodes), self.dim_node_feat, dtype=torch.float32, device=self.device)

                # fetch the cached features
                cached_node_index = self.cache_node_map[nodes[cache_mask]]
                node_feature[cache_mask] = self.cache_node_buffer[cached_node_index]
                # fetch the uncached features
                uncached_mask = ~cache_mask
                if uncached_mask.sum() > 0:
                    uncached_node_id = nodes[uncached_mask]
                    uncached_node_id_unique, uncached_node_id_unique_index = torch.unique(
                        uncached_node_id, return_inverse=True)

                    if self.distributed:

                        node_feat = self.kvstore_client.pull_collect(
                            futures, masks, 'node')

                        if self.pinned_nfeat_buffs is not None:
                            # TODO: maybe fetch local and remote features separately
                            self.pinned_nfeat_buffs[
                                i][:uncached_node_id_unique.shape[0]] = node_feat
                            uncached_node_feature = self.pinned_nfeat_buffs[i][:uncached_node_id_unique.shape[0]].to(
                                self.device, non_blocking=True).float()
                        else:
                            uncached_node_feature = node_feat.to(
                                self.device).float()
                    else:
                        if self.pinned_nfeat_buffs is not None:
                            torch.index_select(self.node_feats, 0, uncached_node_id_unique.to('cpu'),
                                               out=self.pinned_nfeat_buffs[i][:uncached_node_id_unique.shape[0]])
                            uncached_node_feature = self.pinned_nfeat_buffs[i][:uncached_node_id_unique.shape[0]].to(
                                self.device, non_blocking=True)
                        else:
                            uncached_node_feature = self.node_feats[uncached_node_id_unique].to(
                                self.device, non_blocking=True)
                    node_feature[uncached_mask] = uncached_node_feature[uncached_node_id_unique_index]

                i += 1
                b.srcdata['h'] = node_feature

                if update_cache and uncached_mask.sum() > 0:
                    self.update_node_cache(cached_node_index=cached_node_index,
                                           uncached_node_id=uncached_node_id_unique,
                                           uncached_node_feature=uncached_node_feature)

            self.cache_node_ratio = hit_ratio_sum / i if i > 0 else 0

        # Edge feature
        if self.dim_edge_feat != 0:
            i = 0
            hit_ratio_sum = 0
            for mfg in mfgs:
                for b, (futures, masks) in zip(mfg, edge_futures):  # only have one layer
                    edges = b.edata['ID']
                    assert isinstance(edges, torch.Tensor)
                    if len(edges) == 0:
                        continue

                    cache_mask = self.cache_edge_flag[edges]
                    hit_ratio = torch.sum(cache_mask) / len(edges)
                    hit_ratio_sum += hit_ratio

                    edge_feature = torch.zeros(len(edges), self.dim_edge_feat,
                                               dtype=torch.float32, device=self.device)

                    # fetch the cached features
                    cached_edge_index = self.cache_edge_map[edges[cache_mask]]
                    edge_feature[cache_mask] = self.cache_edge_buffer[cached_edge_index]
                    # fetch the uncached features
                    uncached_mask = ~cache_mask
                    uncached_edge_id = edges[uncached_mask]

                    if len(uncached_edge_id) > 0:
                        if self.distributed:
                            # edge_features need to convert to nid first.
                            # get the first_indices of the origin tensor in unique
                            # pytorch should have this option like numpy !!
                            uncached_edge_id_unique, uncached_edge_id_unique_index, counts = torch.unique(
                                uncached_edge_id, return_inverse=True, return_counts=True)

                            edge_feat = self.kvstore_client.pull_collect(
                                futures, masks, 'edge')

                            if self.pinned_efeat_buffs is not None:
                                self.pinned_efeat_buffs[
                                    i][:uncached_edge_id_unique.shape[0]] = edge_feat
                                uncached_edge_feature = self.pinned_efeat_buffs[i][:uncached_edge_id_unique.shape[0]].to(
                                    self.device, non_blocking=True).float()
                            else:
                                uncached_edge_feature = edge_feat.to(
                                    self.device).float()
                        else:
                            uncached_edge_id_unique, uncached_edge_id_unique_index = torch.unique(
                                uncached_edge_id, return_inverse=True)
                            if self.pinned_efeat_buffs is not None:
                                torch.index_select(self.edge_feats, 0, uncached_edge_id_unique.to('cpu'),
                                                   out=self.pinned_efeat_buffs[i][:uncached_edge_id_unique.shape[0]])
                                uncached_edge_feature = self.pinned_efeat_buffs[i][:uncached_edge_id_unique.shape[0]].to(
                                    self.device, non_blocking=True)
                            else:
                                uncached_edge_feature = self.edge_feats[uncached_edge_id_unique].to(
                                    self.device, non_blocking=True)

                        edge_feature[uncached_mask] = uncached_edge_feature[uncached_edge_id_unique_index]

                    i += 1
                    b.edata['f'] = edge_feature

                    if update_cache and len(uncached_edge_id) > 0:
                        self.update_edge_cache(cached_edge_index=cached_edge_index,
                                               uncached_edge_id=uncached_edge_id_unique,
                                               uncached_edge_feature=uncached_edge_feature)

            self.cache_edge_ratio = hit_ratio_sum / i if i > 0 else 0

            if target_edge_features:
                if self.distributed:

                    futures, masks = target_egde_futures
                    target_edge_feat = self.kvstore_client.pull_collect(
                        futures, masks, 'edge')

                    self.target_edge_features.put(target_edge_feat.float())
                    # self.target_edge_features = self.kvstore_client.pull(
                    #     torch.from_numpy(eid), mode='edge', nid=nid).float()
                else:
                    # self.target_edge_features = self.edge_feats[eid]
                    self.target_edge_features.put(self.edge_feats[eid])

        return mfgs
