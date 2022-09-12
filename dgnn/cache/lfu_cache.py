from typing import List, Optional, Union

import torch

from .cache import Cache


class LFUCache(Cache):
    """
    Least Frequently Used (LFU) Cache
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
        super(LFUCache, self).__init__(cache_ratio, num_nodes,
                                       num_edges, device, node_feats,
                                       edge_feats, pinned_nfeat_buffs,
                                       pinned_efeat_buffs)
        self.name = 'lfu'

    def init_cache(self, *args, **kwargs):
        """
        Init the caching with node features
        """
        cache_node_id, cache_edge_id = super(LFUCache, self).init_cache(
            sampler, train_df, pre_sampling_rounds=2)
        if self.node_feats != None:
            self.cache_node_count[cache_node_id] += 1

        if self.edge_feats != None:
            self.cache_edge_count[cache_edge_id] += 1

    def update_node_cache(self, cached_node_index, uncached_node_id, uncached_node_feature):
        # If the number of nodes to cache is larger than the cache capacity, we only cache the first
        # self.capacity nodes
        if len(uncached_node_id) > self.node_capacity:
            num_node_to_cache = self.node_capacity
        else:
            num_node_to_cache = len(uncached_node_id)
        node_id_to_cache = uncached_node_id[:num_node_to_cache]
        node_feature_to_cache = uncached_node_feature[:num_node_to_cache]

        # update cached node index first
        self.cache_node_count[cached_node_index] += 1

        # get the k node id with the least water level
        removing_node_index = torch.topk(
            self.cache_node_count, k=num_node_to_cache, largest=False).indices
        assert len(removing_node_index) == len(
            node_id_to_cache) == len(node_feature_to_cache)
        removing_node_id = self.cache_index_to_node_id[removing_node_index]

        # update cache attributes
        self.cache_node_buffer[removing_node_index] = node_feature_to_cache
        self.cache_node_count[removing_node_index] = 1
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

        # update cached edge index first
        self.cache_edge_count[cached_edge_index] += 1

        # get the k edge id with the least water level
        removing_edge_index = torch.topk(
            self.cache_edge_count, k=num_edge_to_cache, largest=False).indices
        # assert len(removing_edge_index) == len(
        #     edge_id_to_cache) == len(edge_feature_to_cache)
        removing_edge_id = self.cache_index_to_edge_id[removing_edge_index]

        # update cache attributes
        self.cache_edge_buffer[removing_edge_index] = edge_feature_to_cache
        self.cache_edge_count[removing_edge_index] = 1
        self.cache_edge_flag[removing_edge_id] = False
        self.cache_edge_flag[edge_id_to_cache] = True
        self.cache_edge_map[removing_edge_id] = -1
        self.cache_edge_map[edge_id_to_cache] = removing_edge_index
        self.cache_index_to_edge_id[removing_edge_index] = edge_id_to_cache.to(
            self.device, non_blocking=True)
