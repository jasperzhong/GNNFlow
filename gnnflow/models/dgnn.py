"""
This code is based on the implementation of TGL's model module.

Implementation at:
    https://github.com/amazon-research/tgl/blob/main/modules.py
"""
from typing import Dict, List, Optional, Union

import torch
from dgl.heterograph import DGLBlock
from gnnflow.distributed.kvstore import KVStoreClient
from gnnflow.models.modules.layers import EdgePredictor, TransfomerAttentionLayer
from gnnflow.models.modules.memory import Memory
from gnnflow.models.modules.memory_updater import GRUMemeoryUpdater


class DGNN(torch.nn.Module):
    """
    Dynamic Graph Neural Model (DGNN)
    """

    def __init__(self, dim_node: int, dim_edge: int, dim_time: int,
                 dim_embed: int, num_layers: int, num_snapshots: int,
                 att_head: int, dropout: float, att_dropout: float,
                 use_memory: bool, dim_memory: Optional[int] = None,
                 num_nodes: Optional[int] = None,
                 memory_device: Union[torch.device, str] = 'cpu',
                 memory_shared: bool = False,
                 kvstore_client: Optional[KVStoreClient] = None,
                 *args, **kwargs):
        """
        Args:
            dim_node: dimension of node features/embeddings
            dim_edge: dimension of edge features
            dim_time: dimension of time features
            dim_embed: dimension of output embeddings
            num_layers: number of layers
            num_snapshots: number of snapshots
            att_head: number of heads for attention
            dropout: dropout rate
            att_dropout: dropout rate for attention
            use_memory: whether to use memory
            dim_memory: dimension of memory
            num_nodes: number of nodes in the graph
            memory_device: device of the memory
            memory_shared: whether to share memory across local workers
            kvstore_client: The KVStore_Client for fetching memorys when using partition
        """
        super(DGNN, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.dim_time = dim_time
        self.dim_embed = dim_embed
        self.num_layers = num_layers
        self.num_snapshots = num_snapshots
        self.att_head = att_head
        self.dropout = dropout
        self.att_dropout = att_dropout
        self.use_memory = use_memory

        if self.use_memory:
            assert num_snapshots == 1, 'memory is not supported for multiple snapshots'
            assert dim_memory is not None, 'dim_memory should be specified'
            assert num_nodes is not None, 'num_nodes is required when using memory'

            self.memory = Memory(num_nodes, dim_edge, dim_memory,
                                 memory_device, memory_shared,
                                 kvstore_client)

            self.memory_updater = GRUMemeoryUpdater(
                dim_node, dim_edge, dim_time, dim_embed, dim_memory)
            dim_node = dim_memory

        self.layers = torch.nn.ModuleDict()
        for l in range(num_layers):
            for h in range(num_snapshots):
                if l == 0:
                    dim_node_input = dim_node
                else:
                    dim_node_input = dim_embed

                key = 'l' + str(l) + 'h' + str(h)
                self.layers[key] = TransfomerAttentionLayer(dim_node_input,
                                                            dim_edge,
                                                            dim_time,
                                                            dim_embed,
                                                            att_head,
                                                            dropout,
                                                            att_dropout)

        if self.num_snapshots > 1:
            self.combiner = torch.nn.RNN(
                dim_embed, dim_embed)

        self.last_updated = None
        self.edge_predictor = EdgePredictor(dim_embed)

    def reset(self):
        if self.use_memory:
            self.memory.reset()

    def resize(self, num_nodes: int):
        if self.use_memory:
            self.memory.resize(num_nodes)

    def has_memory(self):
        return self.use_memory

    def backup_memory(self) -> Dict:
        if self.use_memory:
            return self.memory.backup()
        return {}

    def restore_memory(self, backup: Dict):
        if self.use_memory:
            self.memory.restore(backup)

    def forward(self, mfgs: List[List[DGLBlock]], return_embed: bool =False):
        """
        Args:
            mfgs: list of list of DGLBlocks
            neg_sample_ratio: negative sampling ratio
        """
        out = list()
        for l in range(self.num_layers):
            for h in range(self.num_snapshots):
                key = 'l' + str(l) + 'h' + str(h)
                rst = self.layers[key](mfgs[l][h])
                if l != self.num_layers - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)

        if self.num_snapshots == 1:
            embed = out[0]
        else:
            embed = torch.stack(out, dim=0)
            embed = self.combiner(embed)[0][-1, :, :]

        if return_embed:
            return embed
        return self.edge_predictor(embed)
