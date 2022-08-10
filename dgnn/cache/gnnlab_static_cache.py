import time
import numpy as np
import torch

from dgnn.cache.cache import Cache
from dgnn.utils import get_batch


class GNNLabStaticCache(Cache):
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
            g: DGL distributed graph
            local_nid: local node ids
            cost_ratio: remote/local cost ratio
        """
        super(GNNLabStaticCache, self).__init__(capacity, num_nodes,
                                        num_edges, node_features,
                                        edge_features, device,
                                        pinned_nfeat_buffs,
                                        pinned_efeat_buffs)
        # name
        self.name = 'gnnlab'

        self.cache_index_to_edge_id = None
        self.cache_index_to_node_id = None

    def init_cache(self, sampler, train_df, pre_sampling_rounds=5):
        """Init the caching with node features
        """
        node_sampled_count = torch.zeros(self.num_nodes, dtype=torch.int32, device=self.device,
                                            requires_grad=False)
        edge_sampled_count = torch.zeros(self.num_edges, dtype=torch.int32, device=self.device,
                                            requires_grad=False)
        # Do sampling for multiple rounds
        for epoch in range(pre_sampling_rounds):
            for target_nodes, ts, eid in get_batch(train_df):
                mfgs = sampler.sample(target_nodes, ts)
                if self.node_features != None:
                    for b in mfgs[0]:
                        node_sampled_count[b.srcdata['ID']] += 1
                if self.edge_features != None:
                    for mfg in mfgs:
                        for b in mfg:
                            if b.num_src_nodes() > b.num_dst_nodes():
                                edge_sampled_count[b.edata['ID']] += 1

        if self.node_features != None:
            # Get the top-k nodes with the highest sampling count
            cache_node_id = torch.topk(node_sampled_count, k=self.node_capacity, largest=True).indices.to(self.device)

            # Init parameters related to feature fetching
            cache_node_index = torch.arange(self.node_capacity, dtype=torch.int64).to(self.device)
            torch.index_select(self.node_features, 0, cache_node_id.to('cpu'),
                               out=self.pinned_nfeat_buffs[0][:cache_node_id.shape[0]])
            self.cache_node_buffer[cache_node_index] = self.pinned_nfeat_buffs[0][:cache_node_id.shape[0]].to(self.device, non_blocking=True)
            self.cache_node_flag[cache_node_id] = True
            self.cache_node_map[cache_node_id] = cache_node_index

        if self.edge_features != None:
            # Get the top-k edges with the highest sampling count
            cache_edge_id = torch.topk(edge_sampled_count, k=self.edge_capacity, largest=True).indices.to(self.device)

            # Init parameters related to feature fetching
            cache_edge_index = torch.arange(self.edge_capacity, dtype=torch.int64).to(self.device)
            self.cache_edge_buffer[cache_edge_index] = self.edge_features[cache_edge_id].to(self.device)
            # torch.index_select(self.edge_features, 0, cache_edge_id.to('cpu'),
            #                    out=self.pinned_efeat_buffs[0][:cache_edge_id.shape[0]])
            # self.cache_edge_buffer[cache_edge_index] = self.pinned_efeat_buffs[0][:cache_edge_id.shape[0]].to(self.device, non_blocking=True)
            self.cache_edge_flag[cache_edge_id] = True
            self.cache_edge_map[cache_edge_id] = cache_edge_index


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
        self.update_node_time = 0
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
                cache_update_node_time = 0
                fetch_node_uncache_end.synchronize()
                fetch_node_cache_time += fetch_node_cache_start.elapsed_time(
                    fetch_node_cache_end)
                fetch_node_uncache_time += fetch_node_cache_end.elapsed_time(
                    fetch_node_uncache_end)
            self.update_node_time = update_node_time / 1000
            self.cache_node_ratio = cache_node_ratio_sum / i

        self.fetch_node_cache_time = fetch_node_cache_time
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
