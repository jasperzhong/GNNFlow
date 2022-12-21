from typing import List, Optional, Union

import numpy as np
import torch
from dgl.heterograph import DGLBlock

from gnnflow.cache.cache import Cache
from gnnflow.distributed.kvstore import KVStoreClient
from gnnflow.temporal_sampler import TemporalSampler
from gnnflow.utils import get_batch_no_neg


class GNNLabStaticCache(Cache):
    """
    GNNLab static cache

    paper: https://dl.acm.org/doi/abs/10.1145/3492321.3519557
    """

    def __init__(self, cache_ratio: int, num_nodes: int, num_edges: int,
                 device: Union[str, torch.device],
                 node_feats: Optional[torch.Tensor] = None,
                 edge_feats: Optional[torch.Tensor] = None,
                 dim_node_feat: Optional[int] = 0,
                 dim_edge_feat: Optional[int] = 0,
                 pinned_nfeat_buffs: Optional[torch.Tensor] = None,
                 pinned_efeat_buffs: Optional[torch.Tensor] = None,
                 kvstore_client: Optional[KVStoreClient] = None,
                 distributed: Optional[bool] = False,
                 neg_sample_ratio: Optional[int] = 1):
        """
        Initialize the cache

        Args:
            cache_ratio: The ratio of the cache size to the total number of nodes or edges
                    range: [0, 1].
            num_nodes: The number of nodes in the graph
            num_edges: The number of edges in the graph
            device: The device to use
            node_feats: The node features
            edge_feats: The edge features
            dim_node_feat: The dimension of node features
            dim_edge_feat: The dimension of edge features
            pinned_nfeat_buffs: The pinned memory buffers for node features
            pinned_efeat_buffs: The pinned memory buffers for edge features
            kvstore_client: The KVStore_Client for fetching features when using distributed
                    training
            distributed: Whether to use distributed training
            neg_sample_ratio: The ratio of negative samples to positive samples
        """
        super(GNNLabStaticCache, self).__init__(cache_ratio, num_nodes,
                                                num_edges, device,
                                                node_feats, edge_feats,
                                                dim_node_feat, dim_edge_feat,
                                                pinned_nfeat_buffs,
                                                pinned_efeat_buffs,
                                                kvstore_client, distributed,
                                                neg_sample_ratio)
        # name
        self.name = 'gnnlab'

        self.cache_index_to_node_id = None
        self.cache_index_to_edge_id = None

    def reset(self):
        """Reset the cache"""
        # do nothing
        return

    def get_mem_size(self) -> int:
        """
        Get the memory size of the cache in bytes
        """
        mem_size = 0
        if self.dim_node_feat != 0:
            mem_size += self.cache_node_buffer.element_size() * self.cache_node_buffer.nelement()
            mem_size += self.cache_node_flag.element_size() * self.cache_node_flag.nelement()
            mem_size += self.cache_node_map.element_size() * self.cache_node_map.nelement()

        if self.dim_edge_feat != 0:
            mem_size += self.cache_edge_buffer.element_size() * self.cache_edge_buffer.nelement()
            mem_size += self.cache_edge_flag.element_size() * self.cache_edge_flag.nelement()
            mem_size += self.cache_edge_map.element_size() * self.cache_edge_map.nelement()

        return mem_size

    def init_cache(self, *args, **kwargs):
        """
        Init the caching with features
        """
        node_sampled_count = torch.zeros(self.num_nodes, dtype=torch.int32)
        edge_sampled_count = torch.zeros(self.num_edges, dtype=torch.int32)
        eid_to_nid = torch.zeros(self.num_edges, dtype=torch.int64)

        sampler = kwargs['sampler']
        train_df = kwargs['train_df']
        pre_sampling_rounds = kwargs.get('pre_sampling_rounds', 2)
        batch_size = kwargs.get('batch_size', 600)

        # Do sampling for multiple rounds
        for _ in range(pre_sampling_rounds):
            for target_nodes, ts, _ in get_batch_no_neg(train_df, batch_size):
                mfgs = sampler.sample(target_nodes, ts)
                if self.node_feats is not None or self.dim_node_feat != 0:
                    for b in mfgs[0]:
                        node_sampled_count[b.srcdata['ID']] += 1
                if self.edge_feats is not None or self.dim_edge_feat != 0:
                    for mfg in mfgs:
                        for b in mfg:
                            if b.num_src_nodes() > b.num_dst_nodes():
                                edge_sampled_count[b.edata['ID']] += 1
                                eid_to_nid[b.edata['ID']
                                           ] = b.srcdata['ID'][b.num_dst_nodes():]

        if self.distributed:
            if self.dim_node_feat != 0 and self.node_capacity > 0:
                # Get the top-k nodes with the highest sampling count
                cache_node_id = torch.topk(
                    node_sampled_count, k=self.node_capacity, largest=True).indices.to(self.device)

                # Init parameters related to feature fetching
                cache_node_index = torch.arange(
                    self.node_capacity, dtype=torch.int64).to(self.device)
                self.cache_node_buffer[cache_node_index] = self.kvstore_client.pull(
                    cache_node_id.cpu(), mode='node').to(self.device)
                self.cache_node_flag[cache_node_id] = True
                self.cache_node_map[cache_node_id] = cache_node_index
        else:
            if self.node_feats is not None:
                # Get the top-k nodes with the highest sampling count
                cache_node_id = torch.topk(
                    node_sampled_count, k=self.node_capacity, largest=True).indices.to(self.device)

                # Init parameters related to feature fetching
                cache_node_index = torch.arange(
                    self.node_capacity, dtype=torch.int64).to(self.device)
                self.cache_node_buffer[cache_node_index] = self.node_feats[cache_node_id].to(
                    self.device, non_blocking=True)
                self.cache_node_flag[cache_node_id] = True
                self.cache_node_map[cache_node_id] = cache_node_index

        if self.distributed:
            if self.dim_edge_feat != 0 and self.edge_capacity > 0:
                # Get the top-k edges with the highest sampling count
                cache_edge_id = torch.topk(
                    edge_sampled_count, k=self.edge_capacity, largest=True).indices.to(self.device)

                # Init parameters related to feature fetching
                cache_edge_index = torch.arange(
                    self.edge_capacity, dtype=torch.int64).to(self.device)
                self.cache_edge_buffer[cache_edge_index] = self.kvstore_client.pull(
                    cache_edge_id.cpu(), mode='edge', nid=eid_to_nid[cache_edge_id.cpu()]).to(self.device)
                self.cache_edge_flag[cache_edge_id] = True
                self.cache_edge_map[cache_edge_id] = cache_edge_index

        else:
            if self.edge_feats is not None:
                # Get the top-k edges with the highest sampling count
                cache_edge_id = torch.topk(
                    edge_sampled_count, k=self.edge_capacity, largest=True).indices.to(self.device)

                # Init parameters related to feature fetching
                cache_edge_index = torch.arange(
                    self.edge_capacity, dtype=torch.int64).to(self.device)
                self.cache_edge_buffer[cache_edge_index] = self.edge_feats[cache_edge_id].to(
                    self.device, non_blocking=True)
                self.cache_edge_flag[cache_edge_id] = True
                self.cache_edge_map[cache_edge_id] = cache_edge_index

    def fetch_feature(self, mfgs: List[List[DGLBlock]],
                      eid: Optional[np.ndarray] = None, update_cache: bool = True,
                      target_edge_features: bool = True):
        """Fetching the node features of input_node_ids

        Args:
            mfgs: message-passing flow graphs
            update_cache: whether to update the cache

        Returns:
            mfgs: message-passing flow graphs with node/edge features
        """
        return super(GNNLabStaticCache, self).fetch_feature(mfgs, eid=eid, update_cache=False, target_edge_features=target_edge_features)
