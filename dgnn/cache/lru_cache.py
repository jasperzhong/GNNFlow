import time
import numpy as np
import torch
from .cache import Cache


class LRUCache(Cache):
    """The class for caching mechanism and feature fetching
    Caching the node features in GPU
    Fetching features from the caching, CPU mem or remote server automatically
    """

    def __init__(self, capacity, num_nodes, num_edges, node_features=None, edge_features=None, device='cpu', pinned_nfeat_buffs=None, pinned_efeat_buffs=None):
        """
        Args:
            capacity: The capacity of the caching (# nodes cached at most)
            num_nodes: number of nodes in the graph
            feature_dim: feature dimensions
        """
        super(LRUCache, self).__init__(capacity, num_nodes,
                                       num_edges, node_features,
                                       edge_features, device,
                                       pinned_nfeat_buffs,
                                       pinned_efeat_buffs)
        # name
        self.name = 'lru'

    def init_cache(self):
        """
        Init the caching with node features
        """
        _, _ = super(LRUCache, self).init_cache()

    def update_edge_cache(self, cached_edge_index, uncached_edge_id, uncached_edge_feature):
        # If the number of edges to cache is larger than the cache capacity, we only cache the first
        # self.capacity edges
        if len(uncached_edge_id) > self.edge_capacity:
            num_edge_to_cache = self.edge_capacity
        else:
            num_edge_to_cache = len(uncached_edge_id)
        edge_id_to_cache = uncached_edge_id[:num_edge_to_cache]
        edge_feature_to_cache = uncached_edge_feature[:num_edge_to_cache]

        # first all -1
        self.cache_edge_count -= 1
        # update cached edge index to 0 (0 is the highest priority)
        self.cache_edge_count[cached_edge_index] = 0

        # get the k edge id with the least water level
        removing_edge_index = torch.topk(
            self.cache_edge_count, k=num_edge_to_cache, largest=False).indices
        assert len(removing_edge_index) == len(
            edge_id_to_cache) == len(edge_feature_to_cache)
        removing_edge_id = self.cache_index_to_edge_id[removing_edge_index]

        # update cache attributes
        self.cache_edge_buffer[removing_edge_index] = edge_feature_to_cache
        self.cache_edge_count[removing_edge_index] = 0
        self.cache_edge_flag[removing_edge_id] = False
        self.cache_edge_flag[edge_id_to_cache] = True
        self.cache_edge_map[removing_edge_id] = -1
        self.cache_edge_map[edge_id_to_cache] = removing_edge_index
        self.cache_index_to_edge_id[removing_edge_index] = edge_id_to_cache.to(
            self.device, non_blocking=True)

    def update_node_cache(self, cached_node_index, uncached_node_id, uncached_node_feature):
        # If the number of nodes to cache is larger than the cache capacity, we only cache the first
        # self.capacity nodes
        if len(uncached_node_id) > self.node_capacity:
            num_node_to_cache = self.node_capacity
        else:
            num_node_to_cache = len(uncached_node_id)
        node_id_to_cache = uncached_node_id[:num_node_to_cache]
        node_feature_to_cache = uncached_node_feature[:num_node_to_cache]

        # first all -1
        self.cache_node_count -= 1
        # update cached node index to 0 (0 is the highest priority)
        self.cache_node_count[cached_node_index] = 0

        # get the k node id with the least water level
        removing_node_index = torch.topk(
            self.cache_node_count, k=num_node_to_cache, largest=False).indices
        assert len(removing_node_index) == len(
            node_id_to_cache) == len(node_feature_to_cache)
        removing_node_id = self.cache_index_to_node_id[removing_node_index]

        # update cache attributes
        self.cache_node_buffer[removing_node_index] = node_feature_to_cache
        self.cache_node_count[removing_node_index] = 0
        self.cache_node_flag[removing_node_id] = False
        self.cache_node_flag[node_id_to_cache] = True
        self.cache_node_map[removing_node_id] = -1
        self.cache_node_map[node_id_to_cache] = removing_node_index
        self.cache_index_to_node_id[removing_node_index] = node_id_to_cache.to(
            self.device, non_blocking=True)
