import torch


class Cache:
    def __init__(self, capacity, num_nodes, num_edges, node_features=None, edge_features=None, device='cpu', pinned_nfeat_buffs=None, pinned_efeat_buffs=None):
        self.node_capacity = int(capacity * num_nodes)
        self.edge_capacity = int(capacity * num_edges)
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

        # TODO: maybe use dgl's index_select
        # so that these pin_buffers is unnecessary
        self.pinned_nfeat_buffs = pinned_nfeat_buffs
        self.pinned_efeat_buffs = pinned_efeat_buffs

        self.cache_node_ratio = 0
        self.cache_edge_ratio = 0

        # stores node's features
        if self.node_feature_dim != 0:
            self.cache_node_buffer = torch.zeros(self.node_capacity, self.node_feature_dim, dtype=torch.float32, device=self.device,
                                                 requires_grad=False)
            # stores node's water level, used by the LRU logic and initialized as zero
            self.cache_node_count = torch.zeros(
                self.node_capacity, dtype=torch.int, device=self.device, requires_grad=False)
            # flag for indicating those cached nodes
            self.cache_node_flag = torch.zeros(
                num_nodes, dtype=torch.bool, device=self.device, requires_grad=False)
            # maps node id -> index
            self.cache_node_map = torch.zeros(
                num_nodes, dtype=torch.int64, device=self.device, requires_grad=False) - 1
            # maps index -> node id
            self.cache_index_to_node_id = torch.zeros(self.node_capacity, dtype=torch.int64, device=self.device,
                                                      requires_grad=False) - 1

        if self.edge_feature_dim != 0:
            self.cache_edge_buffer = torch.zeros(self.edge_capacity, self.edge_feature_dim, dtype=torch.float32, device=self.device,
                                                 requires_grad=False)
            # stores edge's water level, used by the LRU logic and initialized as zero
            self.cache_edge_count = torch.zeros(
                self.edge_capacity, dtype=torch.int, device=self.device, requires_grad=False)
            # flag for indicating those cached edges
            self.cache_edge_flag = torch.zeros(
                num_edges, dtype=torch.bool, device=self.device, requires_grad=False)
            # maps edge id -> index
            self.cache_edge_map = torch.zeros(
                num_edges, dtype=torch.int64, device=self.device, requires_grad=False) - 1
            # maps index -> edge id
            self.cache_index_to_edge_id = torch.zeros(self.edge_capacity, dtype=torch.int64, device=self.device,
                                                      requires_grad=False) - 1

    def init_cache(self, sampler, train_df, pre_sampling_rounds=2):
        """
        Init the caching with node features
        """
        cache_node_id = None
        cache_edge_id = None
        if self.node_features != None:
            cache_node_id = torch.arange(
                self.node_capacity, dtype=torch.int64).to(self.device, non_blocking=True)

            # Init parameters related to feature fetching
            self.cache_node_buffer[cache_node_id] = self.node_features[:self.node_capacity].to(
                self.device, non_blocking=True)
            self.cache_node_flag[cache_node_id] = True
            self.cache_index_to_node_id = cache_node_id.clone().detach()
            self.cache_node_map[cache_node_id] = cache_node_id

        if self.edge_features != None:
            cache_edge_id = torch.arange(
                self.edge_capacity, dtype=torch.int64).to(self.device, non_blocking=True)

            # Init parameters related to feature fetching
            self.cache_edge_buffer[cache_edge_id] = self.edge_features[:self.edge_capacity].to(
                self.device, non_blocking=True)
            self.cache_edge_flag[cache_edge_id] = True
            self.cache_index_to_edge_id = cache_edge_id.clone().detach()
            self.cache_edge_map[cache_edge_id] = cache_edge_id

        return cache_node_id, cache_edge_id

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
        update_node_time = 0
        update_edge_time = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        update_node_start = torch.cuda.Event(enable_timing=True)
        update_node_end = torch.cuda.Event(enable_timing=True)
        update_edge_start = torch.cuda.Event(enable_timing=True)
        update_edge_end = torch.cuda.Event(enable_timing=True)
        start.record()
        cache_node_ratio_sum = 0
        i = 0
        self.update_node_time = 0
        if self.node_features is not None:
            for b in mfgs[0]:  # not sure why
                cache_mask = self.cache_node_flag[b.srcdata['ID']]
                cache_node_ratio = torch.sum(
                    cache_mask) / len(b.srcdata['ID'])
                cache_node_ratio_sum += cache_node_ratio

                node_feature = torch.zeros(len(b.srcdata['ID']), self.node_feature_dim, dtype=torch.float32, device=self.device,
                                           requires_grad=False)

                # fetch the cached features
                cached_node_index = self.cache_node_map[b.srcdata['ID'][cache_mask]]
                node_feature[cache_mask] = self.cache_node_buffer[cached_node_index]
                # fetch the uncached features
                uncached_mask = ~cache_mask
                uncached_node_id = b.srcdata['ID'][uncached_mask]
                if self.pinned_nfeat_buffs is not None:
                    torch.index_select(self.node_features, 0, uncached_node_id.to('cpu'),
                                       out=self.pinned_nfeat_buffs[i][:uncached_node_id.shape[0]])
                    node_feature[uncached_mask] = self.pinned_nfeat_buffs[i][:uncached_node_id.shape[0]].to(
                        self.device, non_blocking=True)
                else:
                    node_feature[uncached_mask] = self.node_features[uncached_node_id].to(
                        self.device, non_blocking=True)
                i += 1
                # save the node feature into the mfgs
                b.srcdata['h'] = node_feature
                # update the cache buffer
                # TODO: if have many snapshots
                if update_cache:
                    update_node_start.record()
                    cached_node_index_unique = cached_node_index.unique()
                    uncached_node_id_unique = uncached_node_id.unique()
                    # TODO: need optimize
                    uncached_node_feature = self.node_features[uncached_node_id_unique].to(
                        self.device)
                    self.update_node_cache(cached_node_index=cached_node_index_unique, uncached_node_id=uncached_node_id_unique,
                                           uncached_node_feature=uncached_node_feature)
                    update_node_end.record()
                    update_node_end.synchronize()
                    cache_update_node_time = update_node_start.elapsed_time(
                        update_node_end)
                    update_node_time += cache_update_node_time
                else:
                    cache_update_node_time = 0

            self.update_node_time = update_node_time / 1000
            self.cache_node_ratio = cache_node_ratio_sum / i if i > 0 else 0

        # Edge feature
        i = 0
        cache_edge_ratio_sum = 0
        if self.edge_features is not None:
            for mfg in mfgs:
                for b in mfg:
                    if b.num_src_nodes() > b.num_dst_nodes():  # edges > 0
                        cache_mask = self.cache_edge_flag[b.edata['ID']]
                        cache_edge_ratio = torch.sum(
                            cache_mask) / len(b.edata['ID'])
                        cache_edge_ratio_sum += cache_edge_ratio

                        edge_feature = torch.zeros(len(b.edata['ID']), self.edge_feature_dim, dtype=torch.float32, device=self.device,
                                                   requires_grad=False)

                        # fetch the cached features
                        cached_edge_index = self.cache_edge_map[b.edata['ID'][cache_mask]]
                        edge_feature[cache_mask] = self.cache_edge_buffer[cached_edge_index]
                        # fetch the uncached features
                        uncached_mask = ~cache_mask
                        uncached_edge_id = b.edata['ID'][uncached_mask]
                        if self.pinned_efeat_buffs is not None:
                            torch.index_select(self.edge_features, 0, uncached_edge_id.to('cpu'),
                                               out=self.pinned_efeat_buffs[i][:uncached_edge_id.shape[0]])

                            edge_feature[uncached_mask] = self.pinned_efeat_buffs[i][:uncached_edge_id.shape[0]].to(
                                self.device, non_blocking=True)
                        else:
                            edge_feature[uncached_mask] = self.edge_features[uncached_edge_id].to(
                                self.device, non_blocking=True)
                        i += 1

                        b.edata['f'] = edge_feature

                        # TODO
                        if update_cache:
                            update_edge_start.record()
                            cached_edge_index_unique = cached_edge_index.unique()
                            uncached_edge_id_unique = uncached_edge_id.unique()
                            # TODO: need optimize
                            uncached_edge_feature = self.edge_features[uncached_edge_id_unique].to(
                                self.device)
                            self.update_edge_cache(cached_edge_index=cached_edge_index_unique, uncached_edge_id=uncached_edge_id_unique,
                                                   uncached_edge_feature=uncached_edge_feature)
                            update_edge_end.record()
                            update_edge_end.synchronize()
                            cache_update_time = update_edge_start.elapsed_time(
                                update_edge_end)
                            update_edge_time += cache_update_time
                        else:
                            cache_update_time = 0
            self.cache_edge_ratio = cache_edge_ratio_sum / i if i > 0 else 0

        end.record()
        end.synchronize()
        fetch_time = start.elapsed_time(
            end) - update_edge_time - update_node_time

        self.fetch_time = fetch_time / 1000
        self.update_edge_time = update_edge_time / 1000

        return mfgs
