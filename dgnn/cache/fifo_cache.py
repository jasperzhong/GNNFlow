import time
import numpy as np
import torch


class FIFOCache:
    """The class for caching mechanism and feature fetching
    Caching the node features in GPU
    Fetching features from the caching, CPU mem or remote server automatically
    """

    def __init__(self, capacity, num_nodes, feature_dim, g, device):
        """
        Args:
            capacity: The capacity of the caching (# nodes cached at most)
            num_nodes: number of nodes in the graph
            feature_dim: feature dimensions
            g: DGL distributed graph
        """
        # name
        self.name = 'fifo'
        # capacity
        self.capacity = min(capacity, num_nodes)
        # number of nodes in graph
        self.num_nodes = num_nodes
        # feature dimension
        self.feature_dim = feature_dim
        # graph
        self.g = g
        # log_file
        self.log = None

        # storing node in GPU
        self.device = device
        # stores node's features
        self.cache_buffer = torch.zeros(self.capacity, feature_dim, dtype=torch.float32, device=self.device,
                                        requires_grad=False)
        # pointer to the last entry for the recent cached nodes
        self.cache_pointer = 0
        # flag for indicating those cached nodes
        self.cache_flag = torch.zeros(
            num_nodes, dtype=torch.bool, device=self.device, requires_grad=False)
        # maps node id -> index
        self.cache_map = torch.zeros(
            num_nodes, dtype=torch.int64, device=self.device, requires_grad=False) - 1
        # maps index -> node id
        self.cache_index_to_node_id = torch.zeros(self.capacity, dtype=torch.int64, device=self.device,
                                                  requires_grad=False) - 1

    def init_cache(self):
        """Init the caching with node features
        """
        cache_node_id = torch.tensor(
            list(range(self.capacity)), dtype=torch.int64).to(self.device)

        # Init parameters related to feature fetching
        self.cache_buffer[cache_node_id] = self.g.ndata['features'][cache_node_id].to(
            self.device)
        self.cache_flag[cache_node_id] = True
        self.cache_index_to_node_id = torch.tensor(
            cache_node_id, device=self.device)
        node_id = cache_node_id
        index = cache_node_id
        self.cache_map[node_id] = index
        self.cache_pointer = self.capacity - 1

    def fetch_feature(self, input_node_id, update_cache=True):
        """Fetching the node features of input_node_ids
        Args:
            input_node_id: node ids of all the input nodes
            update_cache: update cache or not
        Returns:
            node_feature: tensor
                Tensor stores node features, with a shape of [len(input_node_id), self.feature_dim]
            cache_ratio: (# cached nodes)/(# uncached nodes)
            fetch_time: time to fetch the features
            cache_update_time: time to update the cache
        """
        start_time = time.time()
        cache_mask = self.cache_flag[input_node_id]
        cache_ratio = torch.sum(cache_mask) / len(input_node_id)

        node_feature = torch.zeros(len(input_node_id), self.feature_dim, dtype=torch.float32, device=self.device,
                                   requires_grad=False)

        # fetch the cached features
        cached_node_index = self.cache_map[input_node_id[cache_mask]]
        assert torch.min(cached_node_index) >= 0, "look up non-existing keys"
        node_feature[cache_mask] = self.cache_buffer[cached_node_index]

        # fetch the uncached features
        uncached_mask = ~cache_mask
        node_feature[uncached_mask] = self.g.ndata['features'][input_node_id[uncached_mask]].to(
            self.device)

        fetch_time = time.time() - start_time

        # update the cache buffer
        if update_cache:
            start_time = time.time()
            self.update_cache(cached_node_index=cached_node_index, uncached_node_id=input_node_id[uncached_mask],
                              uncached_node_feature=node_feature[uncached_mask])
            cache_update_time = time.time() - start_time
        else:
            cache_update_time = 0

        return node_feature, cache_ratio, fetch_time, cache_update_time

    def update_cache(self, cached_node_index, uncached_node_id, uncached_node_feature):
        # If the number of nodes to cache is larger than the cache capacity, we only cache the first
        # self.capacity nodes
        if len(uncached_node_id) > self.capacity:
            num_node_to_cache = self.capacity
        else:
            num_node_to_cache = len(uncached_node_id)
        node_id_to_cache = uncached_node_id[:num_node_to_cache]
        node_feature_to_cache = uncached_node_feature[:num_node_to_cache]

        # # update cached node index first
        # self.cache_count[cached_node_index] = 0
        #
        # # get the k node id with the least water level
        # removing_node_index = torch.topk(self.cache_count, k=num_node_to_cache, largest=False).indices
        # assert len(removing_node_index) == len(node_id_to_cache) == len(node_feature_to_cache)
        # removing_node_id = self.cache_index_to_node_id[removing_node_index]

        if self.cache_pointer + num_node_to_cache < self.capacity:
            removing_node_index = torch.arange(
                self.cache_pointer + 1, self.cache_pointer + num_node_to_cache + 1)
            self.cache_pointer = self.cache_pointer + num_node_to_cache
        else:
            removing_node_index = torch.cat([torch.arange(num_node_to_cache - (self.capacity - 1 - self.cache_pointer)),
                                             torch.arange(self.cache_pointer + 1, self.capacity)])
            self.cache_pointer = num_node_to_cache - \
                (self.capacity - 1 - self.cache_pointer) - 1
        assert len(removing_node_index) == len(
            node_id_to_cache) == len(node_feature_to_cache)
        removing_node_index = removing_node_index.to(device=self.device)
        removing_node_id = self.cache_index_to_node_id[removing_node_index]

        # update cache attributes
        self.cache_buffer[removing_node_index] = node_feature_to_cache
        self.cache_flag[removing_node_id] = False
        self.cache_flag[node_id_to_cache] = True
        self.cache_map[removing_node_id] = -1
        self.cache_map[node_id_to_cache] = removing_node_index
        self.cache_index_to_node_id[removing_node_index] = node_id_to_cache.to(
            self.device)
