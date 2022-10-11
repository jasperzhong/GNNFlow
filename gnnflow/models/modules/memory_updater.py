"""
This code is based on the implementation of TGL's memory module.

Implementation at:
    https://github.com/amazon-research/tgl/blob/main/memorys.py
"""
import torch
from dgl.heterograph import DGLBlock

from gnnflow.models.modules.layers import TimeEncode


class GRUMemeoryUpdater(torch.nn.Module):
    """
    GRU memory updater proposed by TGN
    """

    def __init__(self, dim_node: int, dim_edge: int, dim_time: int,
                 dim_memory: int):
        """
        Args:
            dim_node: dimension of node features/embeddings
            dim_edge: dimension of edge features
            dim_time: dimension of time features
            dim_embed: dimension of output embeddings
            dim_memory: dimension of memory
        """
        super(GRUMemeoryUpdater, self).__init__()
        self.dim_message = 2 * dim_memory + dim_edge
        self.dim_node = dim_node
        self.dim_time = dim_time
        self.updater = torch.nn.GRUCell(
            self.dim_message + self.dim_time, dim_memory)

        self.use_time_enc = dim_time > 0
        if self.use_time_enc:
            self.time_enc = TimeEncode(dim_time)

        if dim_node > 0 and dim_node != dim_memory:
            self.node_feat_proj = torch.nn.Linear(dim_node, dim_memory)

    def forward(self, b: DGLBlock):
        """
        Update the memory of nodes

        Args:
           b: sampled message flow graph (mfg), where
                `b.num_dst_nodes()` is the number of target nodes to sample,
                `b.srcdata['ID']` is the node IDs of all nodes, and
                `b.srcdata['ts']` is the timestamp of all nodes.

        Return:
            last_updated: {
                "last_updated_nid": node IDs of the target nodes
                "last_updated_memory": updated memory of the target nodes
                "last_updated_ts": timestamp of the target nodes
            }
        """
        device = b.device

        if self.use_time_enc:
            time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
            b.srcdata['mem_input'] = torch.cat(
                [b.srcdata['mem_input'], time_feat], dim=1)

        updated_memory = self.updater(
            b.srcdata['mem_input'], b.srcdata['mem'])

        num_dst_nodes = b.num_dst_nodes()
        last_updated_nid = b.srcdata['ID'][:num_dst_nodes].clone(
        ).detach().to(device)
        last_updated_memory = updated_memory[:num_dst_nodes].clone(
        ).detach().to(device)
        last_updated_ts = b.srcdata['ts'][:num_dst_nodes].clone(
        ).detach().to(device)

        if self.dim_node > 0:
            if self.dim_node == self.dim_embed:
                b.srcdata['h'] += updated_memory
            else:
                b.srcdata['h'] = updated_memory + \
                    self.node_feat_proj(b.srcdata['h'])
        else:
            b.srcdata['h'] = updated_memory

        return {
            "last_updated_nid": last_updated_nid,
            "last_updated_memory": last_updated_memory,
            "last_updated_ts": last_updated_ts
        }
