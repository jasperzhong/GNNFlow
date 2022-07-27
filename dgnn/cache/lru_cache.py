import time
import numpy as np
import torch


class LRUCache:
    """The class for caching mechanism and feature fetching
    Caching the node features in GPU
    Fetching features from the caching, CPU mem or remote server automatically
    """

    def __init__(self, capacity, num_nodes, num_edges, node_features=None, edge_features=None, device='cpu'):
        """
        Args:
            capacity: The capacity of the caching (# nodes cached at most)
            num_nodes: number of nodes in the graph
            feature_dim: feature dimensions
        """
        # name
        self.name = 'lru'
        # capacity
        # TODO
        self.capacity = int(capacity)
        # number of nodes in graph
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.node_features = node_features
        self.edge_features = edge_features
        # feature dimension
        self.node_feature_dim = 0 if node_features is None else node_features.shape[1]
        self.edge_feature_dim = 0 if edge_features is None else edge_features.shape[1]

        # storing node in GPU
        self.device = device
        # stores node's features
        if self.node_feature_dim != 0:
            self.cache_node_buffer = torch.zeros(self.capacity, self.node_feature_dim, dtype=torch.float32, device=self.device,
                                                 requires_grad=False)
            # stores node's water level, used by the LRU logic and initialized as zero
            self.cache_node_count = torch.zeros(
                self.capacity, dtype=torch.int, device=self.device, requires_grad=False)
            # flag for indicating those cached nodes
            self.cache_node_flag = torch.zeros(
                num_nodes, dtype=torch.bool, device=self.device, requires_grad=False)
            # maps node id -> index
            self.cache_node_map = torch.zeros(
                num_nodes, dtype=torch.int64, device=self.device, requires_grad=False) - 1
            # maps index -> node id
            self.cache_index_to_node_id = torch.zeros(self.capacity, dtype=torch.int64, device=self.device,
                                                      requires_grad=False) - 1

        if self.edge_feature_dim != 0:
            self.cache_edge_buffer = torch.zeros(self.capacity, self.edge_feature_dim, dtype=torch.float32, device=self.device,
                                                 requires_grad=False)
            # stores edge's water level, used by the LRU logic and initialized as zero
            self.cache_edge_count = torch.zeros(
                self.capacity, dtype=torch.int, device=self.device, requires_grad=False)
            # flag for indicating those cached edges
            self.cache_edge_flag = torch.zeros(
                num_edges, dtype=torch.bool, device=self.device, requires_grad=False)
            # maps edge id -> index
            self.cache_edge_map = torch.zeros(
                num_edges, dtype=torch.int64, device=self.device, requires_grad=False) - 1
            # maps index -> edge id
            self.cache_index_to_edge_id = torch.zeros(self.capacity, dtype=torch.int64, device=self.device,
                                                      requires_grad=False) - 1

    def init_cache(self):
        """
        Init the caching with node features
        """
        if self.node_features != None:
            cache_node_id = torch.arange(
                self.capacity, dtype=torch.int64).to(self.device)

            # Init parameters related to feature fetching
            self.cache_node_buffer[cache_node_id] = self.node_features[cache_node_id].to(
                self.device)
            self.cache_node_flag[cache_node_id] = True
            self.cache_index_to_node_id = torch.tensor(
                cache_node_id, device=self.device)
            self.cache_node_map[cache_node_id] = cache_node_id

        if self.edge_features != None:
            cache_edge_id = torch.arange(
                self.capacity, dtype=torch.int64).to(self.device)

            # Init parameters related to feature fetching
            self.cache_edge_buffer[cache_edge_id] = self.edge_features[cache_edge_id].to(
                self.device)
            self.cache_edge_flag[cache_edge_id] = True
            self.cache_index_to_edge_id = torch.tensor(
                cache_edge_id, device=self.device)
            self.cache_edge_map[cache_edge_id] = cache_edge_id

    def fetch_feature(self, mfgs, update_cache=True):
        """Fetching the node features of input_node_ids
        Args:
            mfgs: Message-passing Flow Graphs
        Returns:
            node_feature: tensor
                Tensor stores node features, with a shape of [len(input_node_id), self.feature_dim]
            cache_ratio: (# cached nodes)/(# uncached nodes)
            fetch_time: time to fetch the features
            cache_update_time: time to update the cache
        """
        fetch_time = 0
        update_time = 0
        start = time.time()
        if self.node_features is not None:
            for b in mfgs[0]:  # not sure why
                cache_mask = self.cache_node_flag[b.srcdata['ID']]
                cache_ratio = torch.sum(
                    cache_mask) / len(b.srcdata['ID'])

                node_feature = torch.zeros(len(b.srcdata['ID']), self.node_feature_dim, dtype=torch.float32, device=self.device,
                                           requires_grad=False)

                # fetch the cached features
                cached_node_index = self.cache_node_map[b.srcdata['ID'][cache_mask]]
                assert torch.min(
                    cached_node_index) >= 0, "look up non-existing keys"
                node_feature[cache_mask] = self.cache_node_buffer[cached_node_index]

                # fetch the uncached features
                uncached_mask = ~cache_mask
                uncached_node_id = b.srcdata['ID'][uncached_mask]
                node_feature[uncached_mask] = self.node_features[uncached_node_id].to(
                    self.device)

                # save the node feature into the mfgs
                b.srcdata['h'] = node_feature

                # update the cache buffer
                # TODO: if have many snapshots
                if update_cache:
                    start_time = time.time()
                    self.update_node_cache(cached_node_index=cached_node_index, uncached_node_id=uncached_node_id,
                                           uncached_node_feature=node_feature[uncached_mask])
                    cache_update_time = time.time() - start_time
                else:
                    cache_update_time = 0

        if self.edge_features is not None:
            for mfg in mfgs:
                for b in mfg:
                    if b.num_src_nodes() > b.num_dst_nodes():
                        cache_mask = self.cache_edge_flag[b.edata['ID']]
                        cache_ratio = torch.sum(
                            cache_mask) / len(b.edata['ID'])

                        edge_feature = torch.zeros(len(b.edata['ID']), self.edge_feature_dim, dtype=torch.float32, device=self.device,
                                                   requires_grad=False)

                        # fetch the cached features
                        cached_edge_index = self.cache_edge_map[b.edata['ID'][cache_mask]]
                        edge_feature[cache_mask] = self.cache_edge_buffer[cached_edge_index]

                        # fetch the uncached features
                        uncached_mask = ~cache_mask
                        uncached_edge_id = b.edata['ID'][uncached_mask]
                        edge_feature[uncached_mask] = self.edge_features[uncached_edge_id].to(
                            self.device)

                        # save the edge feature into the mfgs
                        b.edata['f'] = edge_feature

                        # TODO
                        if update_cache:
                            start_time = time.time()
                            self.update_edge_cache(cached_edge_index=cached_edge_index, uncached_edge_id=uncached_edge_id,
                                                   uncached_edge_feature=edge_feature[uncached_mask])
                            cache_update_time = time.time() - start_time
                            update_time += cache_update_time
                        else:
                            cache_update_time = 0

        end = time.time()
        fetch_time = end - start - update_time
        return mfgs, fetch_time, update_time, cache_ratio

    def update_edge_cache(self, cached_edge_index, uncached_edge_id, uncached_edge_feature):
        # If the number of edges to cache is larger than the cache capacity, we only cache the first
        # self.capacity edges
        if len(uncached_edge_id) > self.capacity:
            num_edge_to_cache = self.capacity
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
            self.device)

    def update_node_cache(self, cached_node_index, uncached_node_id, uncached_node_feature):
        # If the number of nodes to cache is larger than the cache capacity, we only cache the first
        # self.capacity nodes
        if len(uncached_node_id) > self.capacity:
            num_node_to_cache = self.capacity
        else:
            num_node_to_cache = len(uncached_node_id)
        node_id_to_cache = uncached_node_id[:num_node_to_cache]
        node_feature_to_cache = uncached_node_feature[:num_node_to_cache]

        # first all -1
        self.cache_count -= 1
        # update cached node index to 0 (0 is the highest priority)
        self.cache_count[cached_node_index] = 0

        # get the k node id with the least water level
        removing_node_index = torch.topk(
            self.cache_count, k=num_node_to_cache, largest=False).indices
        assert len(removing_node_index) == len(
            node_id_to_cache) == len(node_feature_to_cache)
        removing_node_id = self.cache_index_to_node_id[removing_node_index]

        # update cache attributes
        self.cache_buffer[removing_node_index] = node_feature_to_cache
        self.cache_count[removing_node_index] = 0
        self.cache_flag[removing_node_id] = False
        self.cache_flag[node_id_to_cache] = True
        self.cache_map[removing_node_id] = -1
        self.cache_map[node_id_to_cache] = removing_node_index
        self.cache_index_to_node_id[removing_node_index] = node_id_to_cache.to(
            self.device)
