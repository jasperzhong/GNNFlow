from typing import Optional, Union

import torch

from gnnflow.cache.cache import Cache
from gnnflow.distributed.kvstore import KVStoreClient


class LFUCache(Cache):
    """
    Least-frequently-used (LFU) Cache
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
        super(LFUCache, self).__init__(edge_cache_ratio, node_cache_ratio, num_nodes,
                                       num_edges, device, node_feats,
                                       edge_feats, dim_node_feat,
                                       dim_edge_feat, pinned_nfeat_buffs,
                                       pinned_efeat_buffs, kvstore_client,
                                       distributed, neg_sample_ratio)
        self.name = 'lfu'

        if self.dim_node_feat != 0:
            self.cache_node_count = torch.zeros(
                self.node_capacity, dtype=torch.int32, device=self.device)
        if self.dim_edge_feat != 0:
            self.cache_edge_count = torch.zeros(
                self.edge_capacity, dtype=torch.int32, device=self.device)

    def get_mem_size(self) -> int:
        """
        Get the memory size of the cache in bytes
        """
        mem_size = super(LFUCache, self).get_mem_size()
        if self.dim_node_feat != 0:
            mem_size += self.cache_node_count.element_size() * self.cache_node_count.nelement()
        if self.dim_edge_feat != 0:
            mem_size += self.cache_edge_count.element_size() * self.cache_edge_count.nelement()
        return mem_size

    def init_cache(self, *args, **kwargs):
        """
        Init the caching with features
        """
        if self.distributed:
            return
        super(LFUCache, self).init_cache(*args, **kwargs)
        if self.dim_node_feat != 0:
            self.cache_node_count[self.cache_index_to_node_id] += 1

        if self.dim_edge_feat != 0:
            self.cache_edge_count[self.cache_index_to_edge_id] += 1

    def reset(self):
        """
        Reset the cache
        """
        # NB: only edge cache is reset
        if self.distributed:
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

                self.cache_edge_count.zero_()
        else:
            if self.edge_feats is not None:
                cache_edge_id = torch.arange(
                    self.edge_capacity, dtype=torch.int64, device=self.device)

                # Init parameters related to feature fetching
                self.cache_edge_buffer[cache_edge_id] = self.edge_feats[:self.edge_capacity].to(
                    self.device, non_blocking=True)
                self.cache_edge_flag[cache_edge_id] = True
                self.cache_index_to_edge_id = cache_edge_id
                self.cache_edge_map[cache_edge_id] = cache_edge_id

                self.cache_edge_count.zero_()

    def resize(self, new_num_nodes: int, new_num_edges: int):
        """
        Resize the cache

        Args:
            new_num_nodes: The new number of nodes
            new_num_edges: The new number of edges
        """
        super(LFUCache, self).resize(new_num_nodes, new_num_edges)
        if self.dim_node_feat != 0:
            self.cache_node_count.resize_(self.node_capacity)
        if self.dim_edge_feat != 0:
            self.cache_edge_count.resize_(self.edge_capacity)

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
        # If the number of nodes to cache is larger than the cache capacity,
        # we only cache the first self.capacity nodes
        if len(uncached_node_id) > self.node_capacity:
            num_node_to_cache = self.node_capacity
        else:
            num_node_to_cache = len(uncached_node_id)
        node_id_to_cache = uncached_node_id[:num_node_to_cache]
        node_feature_to_cache = uncached_node_feature[:num_node_to_cache]

        # update cached node index first
        self.cache_node_count[cached_node_index] += 1

        # get the k node id with the least water level
        removing_cache_index = torch.topk(
            self.cache_node_count, k=num_node_to_cache, largest=False).indices
        assert len(removing_cache_index) == len(
            node_id_to_cache) == len(node_feature_to_cache)
        removing_node_id = self.cache_index_to_node_id[removing_cache_index]

        # update cache attributes
        self.cache_node_buffer[removing_cache_index] = node_feature_to_cache
        self.cache_node_count[removing_cache_index] = 1
        self.cache_node_flag[removing_node_id] = False
        self.cache_node_flag[node_id_to_cache] = True
        self.cache_node_map[removing_node_id] = -1
        self.cache_node_map[node_id_to_cache] = removing_cache_index
        self.cache_index_to_node_id[removing_cache_index] = node_id_to_cache

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
        # If the number of edges to cache is larger than the cache capacity,
        # we only cache the first self.capacity edges
        if len(uncached_edge_id) > self.edge_capacity:
            num_edge_to_cache = self.edge_capacity
        else:
            num_edge_to_cache = len(uncached_edge_id)
        edge_id_to_cache = uncached_edge_id[:num_edge_to_cache]
        edge_feature_to_cache = uncached_edge_feature[:num_edge_to_cache]

        # update cached edge index first
        self.cache_edge_count[cached_edge_index] += 1

        # get the k edge id with the least water level
        removing_cache_index = torch.topk(
            self.cache_edge_count, k=num_edge_to_cache, largest=False).indices
        assert len(removing_cache_index) == len(
            edge_id_to_cache) == len(edge_feature_to_cache)
        removing_edge_id = self.cache_index_to_edge_id[removing_cache_index]

        # update cache attributes
        self.cache_edge_buffer[removing_cache_index] = edge_feature_to_cache
        self.cache_edge_count[removing_cache_index] = 1
        self.cache_edge_flag[removing_edge_id] = False
        self.cache_edge_flag[edge_id_to_cache] = True
        self.cache_edge_map[removing_edge_id] = -1
        self.cache_edge_map[edge_id_to_cache] = removing_cache_index
        self.cache_index_to_edge_id[removing_cache_index] = edge_id_to_cache
