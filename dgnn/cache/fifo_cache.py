from typing import Optional, Union

import torch

from dgnn.cache.cache import Cache


class FIFOCache(Cache):
    """
    First-in-first-out cache
    """

    def __init__(self, cache_ratio: int, num_nodes: int, num_edges: int,
                 device: Union[str, torch.device],
                 node_feats: Optional[torch.Tensor] = None,
                 edge_feats: Optional[torch.Tensor] = None,
                 pinned_nfeat_buffs: Optional[torch.Tensor] = None,
                 pinned_efeat_buffs: Optional[torch.Tensor] = None):
        """
        Initialize the cache

        Args:
            cache_ratio: The ratio of the cache size to the total number of nodes or edges
            num_nodes: The number of nodes in the graph
            num_edges: The number of edges in the graph
            device: The device to use
            node_feats: The node features
            edge_feats: The edge features
            pinned_nfeat_buffs: The pinned memory buffers for node features
            pinned_efeat_buffs: The pinned memory buffers for edge features
        """
        super(FIFOCache, self).__init__(cache_ratio, num_nodes, num_edges, device,
                                        node_feats, edge_feats, pinned_nfeat_buffs,
                                        pinned_efeat_buffs)
        self.name = 'fifo'
        # pointer to the last entry for the recent cached nodes
        self.cache_node_pointer = 0
        self.cache_edge_pointer = 0

    def init_cache(self, *args, **kwargs):
        """
        Init the cache with features
        """
        super(FIFOCache, self).init_cache(*args, **kwargs)
        if self.node_feats is not None:
            self.cache_node_pointer = self.node_capacity - 1

        if self.edge_feats is not None:
            self.cache_edge_pointer = self.edge_capacity - 1

    def reset(self):
        """
        Reset the cache
        """
        if self.edge_feats is not None:
            self.cache_edge_pointer = self.edge_capacity - 1

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

        if self.cache_node_pointer + num_node_to_cache < self.node_capacity:
            removing_cache_index = torch.arange(
                self.cache_node_pointer + 1, self.cache_node_pointer + num_node_to_cache + 1)
            self.cache_node_pointer = self.cache_node_pointer + num_node_to_cache
        else:
            removing_cache_index = torch.cat([torch.arange(num_node_to_cache - (self.node_capacity - 1 - self.cache_node_pointer)),
                                             torch.arange(self.cache_node_pointer + 1, self.node_capacity)])
            self.cache_node_pointer = num_node_to_cache - \
                (self.node_capacity - 1 - self.cache_node_pointer) - 1
        assert len(removing_cache_index) == len(
            node_id_to_cache) == len(node_feature_to_cache)
        removing_cache_index = removing_cache_index.to(
            device=self.device, non_blocking=True)
        removing_node_id = self.cache_index_to_node_id[removing_cache_index]

        # update cache attributes
        self.cache_node_buffer[removing_cache_index] = node_feature_to_cache
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

        if self.cache_edge_pointer + num_edge_to_cache < self.edge_capacity:
            removing_cache_index = torch.arange(
                self.cache_edge_pointer + 1, self.cache_edge_pointer + num_edge_to_cache + 1)
            self.cache_edge_pointer = self.cache_edge_pointer + num_edge_to_cache
        else:
            removing_cache_index = torch.cat([torch.arange(num_edge_to_cache - (self.edge_capacity - 1 - self.cache_edge_pointer)),
                                             torch.arange(self.cache_edge_pointer + 1, self.edge_capacity)])
            self.cache_edge_pointer = num_edge_to_cache - \
                (self.edge_capacity - 1 - self.cache_edge_pointer) - 1
        assert len(removing_cache_index) == len(
            edge_id_to_cache) == len(edge_feature_to_cache)
        removing_cache_index = removing_cache_index.to(
            device=self.device, non_blocking=True)
        removing_edge_id = self.cache_index_to_edge_id[removing_cache_index]

        # update cache attributes
        self.cache_edge_buffer[removing_cache_index] = edge_feature_to_cache
        self.cache_edge_flag[removing_edge_id] = False
        self.cache_edge_flag[edge_id_to_cache] = True
        self.cache_edge_map[removing_edge_id] = -1
        self.cache_edge_map[edge_id_to_cache] = removing_cache_index
        self.cache_index_to_edge_id[removing_cache_index] = edge_id_to_cache
