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
            cache_node_id = torch.topk(
                node_sampled_count, k=self.node_capacity, largest=True).indices.to(self.device)

            # Init parameters related to feature fetching
            cache_node_index = torch.arange(
                self.node_capacity, dtype=torch.int64).to(self.device)
            if self.pinned_nfeat_buffs is not None:
                torch.index_select(self.node_features, 0, cache_node_id.to('cpu'),
                                   out=self.pinned_nfeat_buffs[0][:cache_node_id.shape[0]])
                self.cache_node_buffer[cache_node_index] = self.pinned_nfeat_buffs[0][:cache_node_id.shape[0]].to(
                    self.device, non_blocking=True)
            else:
                self.cache_node_buffer[cache_node_index] = self.node_features[cache_node_id].to(
                    self.device, non_blocking=True)
            self.cache_node_flag[cache_node_id] = True
            self.cache_node_map[cache_node_id] = cache_node_index

        if self.edge_features != None:
            # Get the top-k edges with the highest sampling count
            cache_edge_id = torch.topk(
                edge_sampled_count, k=self.edge_capacity, largest=True).indices.to(self.device)

            # Init parameters related to feature fetching
            cache_edge_index = torch.arange(
                self.edge_capacity, dtype=torch.int64).to(self.device)
            self.cache_edge_buffer[cache_edge_index] = self.edge_features[cache_edge_id].to(
                self.device)
            # torch.index_select(self.edge_features, 0, cache_edge_id.to('cpu'),
            #                    out=self.pinned_efeat_buffs[0][:cache_edge_id.shape[0]])
            # self.cache_edge_buffer[cache_edge_index] = self.pinned_efeat_buffs[0][:cache_edge_id.shape[0]].to(self.device, non_blocking=True)
            self.cache_edge_flag[cache_edge_id] = True
            self.cache_edge_map[cache_edge_id] = cache_edge_index

    def fetch_feature(self, mfgs, update_cache=False):
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
        mfgs = super(GNNLabStaticCache, self).fetch_feature(mfgs, False)
        return mfgs
