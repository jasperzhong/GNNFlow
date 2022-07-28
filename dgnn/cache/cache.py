import torch
import time


class Cache:
    def __init__(self, capacity, num_nodes, num_edges, node_features=None, edge_features=None, device='cpu'):
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
                    if b.num_src_nodes() > b.num_dst_nodes():  # edges > 0
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
