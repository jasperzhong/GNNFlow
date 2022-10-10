from typing import List, Optional, Union

import torch
from dgl.heterograph import DGLBlock

from gnnflow.distributed.kvstore import KVStoreClient


class Cache:
    """
    Feature cache on GPU
    """

    def __init__(self, cache_ratio: int, num_nodes: int, num_edges: int,
                 device: Union[str, torch.device],
                 node_feats: Optional[torch.Tensor] = None,
                 edge_feats: Optional[torch.Tensor] = None,
                 dim_node_feat: Optional[int] = 0,
                 dim_edge_feat: Optional[int] = 0,
                 pinned_nfeat_buffs: Optional[torch.Tensor] = None,
                 pinned_efeat_buffs: Optional[torch.Tensor] = None,
                 kvstore_client: Optional[KVStoreClient] = None,
                 distributed: Optional[bool] = False):
        """
        Initialize the cache

        Args:
            cache_ratio: The ratio of the cache size to the total number of nodes or edges
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

        # NB: cache_ratio == 0 means no cache
        assert cache_ratio >= 0 and cache_ratio <= 1, 'cache_ratio must be in [0, 1]'

        self.cache_ratio = cache_ratio
        self.node_capacity = int(cache_ratio * num_nodes)
        self.edge_capacity = int(cache_ratio * num_edges)

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
            return 

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
            self.node_capacity = int(self.cache_ratio * self.num_nodes)
            self.cache_node_buffer.resize_(
                self.node_capacity, self.dim_node_feat)
            self.cache_node_flag.resize_(self.num_nodes)
            self.cache_node_map.resize_(self.num_nodes)
            self.cache_index_to_node_id.resize_(self.node_capacity)

        if self.dim_edge_feat != 0 and new_num_edges > self.num_edges:
            self.num_edges = new_num_edges
            self.edge_capacity = int(self.cache_ratio * self.num_edges)
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
                      eid: Optional[torch.Tensor] = None, update_cache: bool = True,
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
                uncached_node_id = nodes[uncached_mask]
                uncached_node_id_unique, uncached_node_id_unique_index = torch.unique(
                    uncached_node_id, return_inverse=True)

                if self.distributed:
                    if self.pinned_nfeat_buffs is not None:
                        # TODO: maybe fetch local and remote features separately
                        # TODO: may need .to('cpu')
                        self.pinned_nfeat_buffs[
                            i][:uncached_node_id_unique.shape[0]] = self.kvstore_client.pull(
                            uncached_node_id_unique, mode='node')
                        uncached_node_feature = self.pinned_nfeat_buffs[i][:uncached_node_id_unique.shape[0]].to(
                            self.device, non_blocking=True)
                    else:
                        uncached_node_feature = self.kvstore_client.pull(
                            uncached_node_id_unique, mode='node')
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

                if update_cache:
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
                    uncached_edge_id_unique, uncached_edge_id_unique_index = torch.unique(
                        uncached_edge_id, return_inverse=True)

                    if self.distributed:
                        # edge_features need to convert to nid first.
                        src_nid = b.srcdata['ID'][b.edges()[1]]
                        if self.pinned_efeat_buffs is not None:
                            self.pinned_efeat_buffs[
                                i][:uncached_edge_id_unique.shape[0]] = self.kvstore_client.pull(
                                    uncached_edge_id_unique, mode='edge', nid=src_nid)
                            uncached_edge_feature = self.pinned_efeat_buffs[i][:uncached_edge_id_unique.shape[0]].to(
                                self.device, non_blocking=True)
                        else:
                            uncached_edge_feature = self.kvstore_client.pull(
                                uncached_edge_id_unique, mode='edge', nid=src_nid)
                    else:
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

                    if update_cache:
                        self.update_edge_cache(cached_edge_index=cached_edge_index,
                                               uncached_edge_id=uncached_edge_id_unique,
                                               uncached_edge_feature=uncached_edge_feature)

            self.cache_edge_ratio = hit_ratio_sum / i if i > 0 else 0

            if target_edge_features:
                if self.distributed:
                    # TODO: maybe there are some edge_features is in the memory now
                    self.target_edge_features = self.kvstore_client.pull(
                        eid, mode='edge')
                else:
                    self.target_edge_features = self.edge_feats[eid]

        return mfgs

    def fetch_feature_distributed(self, mfgs: List[List[DGLBlock]], update_cache: bool = True):
        """Fetching the node/edge features of input_node_ids in distributed setting

        Args:
            mfgs: message-passing flow graphs
            update_cache: whether to update the cache

        Returns:
            mfgs: message-passing flow graphs with node/edge features
        """
        if self.dim_edge_feat != 0:
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
                uncached_node_id = nodes[uncached_mask]
                uncached_node_id_unique, uncached_node_id_unique_index = torch.unique(
                    uncached_node_id, return_inverse=True)

                if self.pinned_nfeat_buffs is not None:
                    # TODO: maybe fetch local and remote features separately
                    # TODO: may need .to('cpu')
                    self.pinned_nfeat_buffs[
                        i][:uncached_node_id_unique.shape[0]] = self.kvstore_client.pull(
                        uncached_node_id_unique, mode='node')
                    uncached_node_feature = self.pinned_nfeat_buffs[i][:uncached_node_id_unique.shape[0]].to(
                        self.device, non_blocking=True)
                else:
                    uncached_node_feature = self.kvstore_client.pull(
                        uncached_node_id_unique, mode='node')

                node_feature[uncached_mask] = uncached_node_feature[uncached_node_id_unique_index]

                i += 1
                b.srcdata['h'] = node_feature

                if update_cache:
                    # TODO: all update cache need rewrite
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
                    uncached_edge_id_unique, uncached_edge_id_unique_index = torch.unique(
                        uncached_edge_id, return_inverse=True)

                    if self.pinned_efeat_buffs is not None:
                        self.pinned_efeat_buffs[
                            i][:uncached_edge_id_unique.shape[0]] = self.kvstore_client.pull(
                                uncached_edge_id_unique, mode='edge')
                        uncached_edge_feature = self.pinned_efeat_buffs[i][:uncached_edge_id_unique.shape[0]].to(
                            self.device, non_blocking=True)
                    else:
                        uncached_edge_feature = self.kvstore_client.pull(
                            uncached_edge_id_unique, mode='edge')
                    edge_feature[uncached_mask] = uncached_edge_feature[uncached_edge_id_unique_index]

                    i += 1
                    b.edata['f'] = edge_feature

                    if update_cache:
                        self.update_edge_cache(cached_edge_index=cached_edge_index,
                                               uncached_edge_id=uncached_edge_id_unique,
                                               uncached_edge_feature=uncached_edge_feature)

            self.cache_edge_ratio = hit_ratio_sum / i if i > 0 else 0

        return mfgs
