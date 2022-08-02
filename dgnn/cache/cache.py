import torch
import time


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
        fetch_node_cache_time = 0
        fetch_node_uncache_time = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        update_node_start = torch.cuda.Event(enable_timing=True)
        update_node_end = torch.cuda.Event(enable_timing=True)
        fetch_node_cache_start = torch.cuda.Event(enable_timing=True)
        fetch_node_cache_end = torch.cuda.Event(enable_timing=True)
        fetch_node_uncache_end = torch.cuda.Event(enable_timing=True)
        apply_end = torch.cuda.Event(enable_timing=True)
        update_start = torch.cuda.Event(enable_timing=True)
        update_end = torch.cuda.Event(enable_timing=True)
        start.record()
        cache_node_ratio_sum = 0
        i = 0
        if self.node_features is not None:
            for b in mfgs[0]:  # not sure why
                fetch_node_cache_start.record()
                cache_mask = self.cache_node_flag[b.srcdata['ID']]
                cache_node_ratio = torch.sum(
                    cache_mask) / len(b.srcdata['ID'])
                cache_node_ratio_sum += cache_node_ratio

                node_feature = torch.zeros(len(b.srcdata['ID']), self.node_feature_dim, dtype=torch.float32, device=self.device,
                                           requires_grad=False)

                # fetch the cached features
                cached_node_index = self.cache_node_map[b.srcdata['ID'][cache_mask]]
                node_feature[cache_mask] = self.cache_node_buffer[cached_node_index]
                fetch_node_cache_end.record()
                # fetch the uncached features
                uncached_mask = ~cache_mask
                uncached_node_id = b.srcdata['ID'][uncached_mask]
                torch.index_select(self.node_features, 0, uncached_node_id.to('cpu'),
                                   out=self.pinned_nfeat_buffs[i][:uncached_node_id.shape[0]])
                node_feature[uncached_mask] = self.pinned_nfeat_buffs[i][:uncached_node_id.shape[0]].to(
                    self.device, non_blocking=True)
                i += 1
                # save the node feature into the mfgs
                b.srcdata['h'] = node_feature
                fetch_node_uncache_end.record()
                # update the cache buffer
                # TODO: if have many snapshots
                if update_cache:
                    update_node_start.record()
                    self.update_node_cache(cached_node_index=cached_node_index, uncached_node_id=uncached_node_id,
                                           uncached_node_feature=node_feature[uncached_mask])
                    update_node_end.record()
                    update_node_end.synchronize()
                    cache_update_node_time = update_node_start.elapsed_time(
                        update_node_end)
                    update_node_time += cache_update_node_time
                else:
                    cache_update_node_time = 0

                fetch_node_cache_time += fetch_node_cache_start.elapsed_time(
                    fetch_node_cache_end)
                fetch_node_uncache_time += fetch_node_cache_end.elapsed_time(
                    fetch_node_uncache_end)
            self.update_node_time = update_node_time
            self.cache_node_ratio = cache_node_ratio_sum / i

        self.fetch_node_cache_time = fetch_node_cache_time / 1000
        self.fetch_node_uncache_time = fetch_node_uncache_time

        # Edge feature
        fetch_cache = 0
        fetch_uncache = 0
        uncache_get_id = 0
        uncache_to_cuda = 0
        apply = 0
        fetch_cache_start = torch.cuda.Event(enable_timing=True)
        fetch_cache_end = torch.cuda.Event(enable_timing=True)
        uncached_get_id_end = torch.cuda.Event(enable_timing=True)
        fetch_uncache_end = torch.cuda.Event(enable_timing=True)
        apply_end = torch.cuda.Event(enable_timing=True)
        update_start = torch.cuda.Event(enable_timing=True)
        update_end = torch.cuda.Event(enable_timing=True)
        i = 0
        cache_edge_ratio_sum = 0
        if self.edge_features is not None:
            for mfg in mfgs:
                for b in mfg:
                    if b.num_src_nodes() > b.num_dst_nodes():  # edges > 0
                        fetch_cache_start.record()
                        cache_mask = self.cache_edge_flag[b.edata['ID']]
                        cache_edge_ratio = torch.sum(
                            cache_mask) / len(b.edata['ID'])
                        cache_edge_ratio_sum += cache_edge_ratio

                        edge_feature = torch.zeros(len(b.edata['ID']), self.edge_feature_dim, dtype=torch.float32, device=self.device,
                                                   requires_grad=False)

                        # fetch the cached features
                        cached_edge_index = self.cache_edge_map[b.edata['ID'][cache_mask]]
                        edge_feature[cache_mask] = self.cache_edge_buffer[cached_edge_index]
                        fetch_cache_end.record()
                        # fetch the uncached features
                        uncached_mask = ~cache_mask
                        uncached_edge_id = b.edata['ID'][uncached_mask]
                        uncached_get_id_end.record()
                        torch.index_select(self.edge_features, 0, uncached_edge_id.to('cpu'),
                                           out=self.pinned_efeat_buffs[i][:uncached_edge_id.shape[0]])

                        edge_feature[uncached_mask] = self.pinned_efeat_buffs[i][:uncached_edge_id.shape[0]].to(
                            self.device, non_blocking=True)
                        i += 1
                        fetch_uncache_end.record()

                        b.edata['f'] = edge_feature
                        apply_end.record()
                        apply_end.synchronize()
                        fetch_cache += fetch_cache_start.elapsed_time(
                            fetch_cache_end)
                        fetch_uncache += fetch_cache_end.elapsed_time(
                            fetch_uncache_end)
                        uncache_get_id += fetch_cache_end.elapsed_time(
                            uncached_get_id_end)
                        uncache_to_cuda += uncached_get_id_end.elapsed_time(
                            fetch_uncache_end)
                        apply += fetch_uncache_end.elapsed_time(apply_end)

                        # TODO
                        if update_cache:
                            update_start.record()
                            self.update_edge_cache(cached_edge_index=cached_edge_index, uncached_edge_id=uncached_edge_id,
                                                   uncached_edge_feature=edge_feature[uncached_mask])
                            update_end.record()
                            update_end.synchronize()
                            cache_update_time = update_start.elapsed_time(
                                update_end)
                            update_edge_time += cache_update_time
                        else:
                            cache_update_time = 0
            self.cache_edge_ratio = cache_edge_ratio_sum / i

        end.record()
        end.synchronize()
        fetch_time = start.elapsed_time(
            end) - update_edge_time - update_node_time

        self.fetch_time = fetch_time / 1000
        self.update_edge_time = update_edge_time / 1000
        self.fetch_cache = fetch_cache / 1000
        self.fetch_uncache = fetch_uncache / 1000
        self.apply = apply / 1000
        self.uncache_get_id = uncache_get_id / 1000
        self.uncache_to_cuda = uncache_to_cuda / 1000

        return mfgs
