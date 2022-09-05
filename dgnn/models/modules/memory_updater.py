import torch
from dgl.heterograph import DGLBlock


class GRUMemeoryUpdater(torch.nn.Module):
    """
    GRU memory updater proposed by TGN
    """

    def __init__(self, dim_node: int, dim_edge: int, dim_time: int,
                 dim_embed: int, dim_memory: int):
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
        self.dim_embed = dim_embed
        self.dim_node = dim_node
        self.dim_time = dim_time
        self.updater = torch.nn.GRUCell(
            self.dim_message + self.dim_time, dim_embed)

        if dim_node > 0 and dim_node != dim_embed:
            self.node_feat_proj = torch.nn.Linear(dim_node, dim_embed)

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
        num_dst_nodes = b.num_dst_nodes()
        device = b.device

        target_node_memory_input = b.dstdata['mem_input']
        target_node_memory = b.srcdata['mem'][:num_dst_nodes]
        updated_memory = self.updater(
            target_node_memory_input, target_node_memory)

        last_updated_nid = b.srcdata['ID'][:num_dst_nodes].to(device)
        last_updated_memory = updated_memory.detach().clone().to(device)
        last_updated_ts = b.srcdata['ts'][:num_dst_nodes].to(device)

        new_memory = torch.cat(
            (updated_memory, b.srcdata['mem'][num_dst_nodes:]), dim=0)

        if self.dim_node > 0:
            if self.dim_node == self.dim_embed:
                b.srcdata['h'] += new_memory
            else:
                b.srcdata['h'] = new_memory + \
                    self.node_feat_proj(b.srcdata['h'])
        else:
            b.srcdata['h'] = new_memory

        return {
            "last_updated_nid": last_updated_nid,
            "last_updated_memory": last_updated_memory,
            "last_updated_ts": last_updated_ts
        }
