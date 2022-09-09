import time
import numpy as np
import torch

from .cache import Cache


class FIFOCache(Cache):
    """The class for caching mechanism and feature fetching
    Caching the node features in GPU
    Fetching features from the caching, CPU mem or remote server automatically
    """

    def __init__(self, capacity, num_nodes, num_edges, node_feats=None, edge_feats=None, device='cpu', pinned_nfeat_buffs=None, pinned_efeat_buffs=None):
        """
        Args:
            capacity: The capacity of the caching (# nodes cached at most)
            num_nodes: number of nodes in the graph
            feature_dim: feature dimensions
        """
        super(FIFOCache, self).__init__(capacity, num_nodes,
                                        num_edges, node_feats,
                                        edge_feats, device,
                                        pinned_nfeat_buffs,
                                        pinned_efeat_buffs)
        # name
        self.name = 'fifo'
        # pointer to the last entry for the recent cached nodes
        self.cache_node_pointer = 0
        self.cache_edge_pointer = 0

        self.cache_node_count = None
        self.cache_edge_count = None

    def init_cache(self, sampler=None, train_df=None, pre_sampling_rounds=2):
        """Init the caching with node features
        """
        _, _ = super(FIFOCache, self).init_cache(
            sampler, train_df, pre_sampling_rounds=2)
        if self.node_feats != None:
            self.cache_node_pointer = self.node_capacity - 1

        if self.edge_feats != None:
            self.cache_edge_pointer = self.edge_capacity - 1

    def update_node_cache(self, cached_node_index, uncached_node_id, uncached_node_feature):
        # If the number of nodes to cache is larger than the cache capacity, we only cache the first
        # self.capacity nodes
        if len(uncached_node_id) > self.node_capacity:
            num_node_to_cache = self.node_capacity
        else:
            num_node_to_cache = len(uncached_node_id)
        node_id_to_cache = uncached_node_id[:num_node_to_cache]
        node_feature_to_cache = uncached_node_feature[:num_node_to_cache]

        if self.cache_node_pointer + num_node_to_cache < self.node_capacity:
            removing_node_index = torch.arange(
                self.cache_node_pointer + 1, self.cache_node_pointer + num_node_to_cache + 1)
            self.cache_node_pointer = self.cache_node_pointer + num_node_to_cache
        else:
            removing_node_index = torch.cat([torch.arange(num_node_to_cache - (self.node_capacity - 1 - self.cache_node_pointer)),
                                             torch.arange(self.cache_node_pointer + 1, self.node_capacity)])
            self.cache_node_pointer = num_node_to_cache - \
                (self.node_capacity - 1 - self.cache_node_pointer) - 1
        assert len(removing_node_index) == len(
            node_id_to_cache) == len(node_feature_to_cache)
        removing_node_index = removing_node_index.to(
            device=self.device, non_blocking=True)
        removing_node_id = self.cache_index_to_node_id[removing_node_index]

        # update cache attributes
        self.cache_node_buffer[removing_node_index] = node_feature_to_cache
        self.cache_node_flag[removing_node_id] = False
        self.cache_node_flag[node_id_to_cache] = True
        self.cache_node_map[removing_node_id] = -1
        self.cache_node_map[node_id_to_cache] = removing_node_index
        self.cache_index_to_node_id[removing_node_index] = node_id_to_cache.to(
            self.device, non_blocking=True)

    def update_edge_cache(self, cached_edge_index, uncached_edge_id, uncached_edge_feature):
        # If the number of edges to cache is larger than the cache capacity, we only cache the first
        # self.capacity edges
        if len(uncached_edge_id) > self.edge_capacity:
            num_edge_to_cache = self.edge_capacity
        else:
            num_edge_to_cache = len(uncached_edge_id)
        edge_id_to_cache = uncached_edge_id[:num_edge_to_cache]
        edge_feature_to_cache = uncached_edge_feature[:num_edge_to_cache]

        if self.cache_edge_pointer + num_edge_to_cache < self.edge_capacity:
            removing_edge_index = torch.arange(
                self.cache_edge_pointer + 1, self.cache_edge_pointer + num_edge_to_cache + 1)
            self.cache_edge_pointer = self.cache_edge_pointer + num_edge_to_cache
        else:
            removing_edge_index = torch.cat([torch.arange(num_edge_to_cache - (self.edge_capacity - 1 - self.cache_edge_pointer)),
                                             torch.arange(self.cache_edge_pointer + 1, self.edge_capacity)])
            self.cache_edge_pointer = num_edge_to_cache - \
                (self.edge_capacity - 1 - self.cache_edge_pointer) - 1
        assert len(removing_edge_index) == len(
            edge_id_to_cache) == len(edge_feature_to_cache)
        removing_edge_index = removing_edge_index.to(
            device=self.device, non_blocking=True)
        removing_edge_id = self.cache_index_to_edge_id[removing_edge_index]

        # update cache attributes
        self.cache_edge_buffer[removing_edge_index] = edge_feature_to_cache
        self.cache_edge_flag[removing_edge_id] = False
        self.cache_edge_flag[edge_id_to_cache] = True
        self.cache_edge_map[removing_edge_id] = -1
        self.cache_edge_map[edge_id_to_cache] = removing_edge_index
        self.cache_index_to_edge_id[removing_edge_index] = edge_id_to_cache.to(
            self.device, non_blocking=True)
