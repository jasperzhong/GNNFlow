from typing import Optional, Union

import torch
from dgl.heterograph import DGLBlock

from dgnn.models.modules.layers import TimeEncode


class Memory:
    """
    Memory module proposed by TGN
    """

    def __init__(self, num_nodes: int, dim_edge: int, dim_time: int,
                 dim_memory: int, device: Union[torch.device, str] = 'cpu',
                 pin_memory: bool = False, shared_memory: bool = False):
        """
        Args:
            num_nodes: number of nodes in the graph
            dim_edge: dimension of the edge features
            dim_time: dimension of the time encoding
            dim_memory: dimension of the output of the memory
            device: device to store the memory
            pin_memory: whether to pin the memory
            shared_memory: whether to store in shared memory (for multi-GPU training)
        """
        self.num_nodes = num_nodes
        self.dim_edge = dim_edge
        self.dim_memory = dim_memory
        # raw message: (src_memory, dst_memory, edge_feat)
        self.dim_raw_message = 2 * dim_memory + dim_edge

        self.use_time_enc = dim_time > 0
        if self.use_time_enc:
            self.time_enc = TimeEncode(dim_time)
            self.time_enc.to(device)


        # memory data structure (default on CPU)
        self.node_memory = torch.zeros(
            (num_nodes, dim_memory), dtype=torch.float32, device=device)
        self.node_memory_ts = torch.zeros(
            num_nodes, dtype=torch.float32, device=device)
        self.mailbox = torch.zeros(
            (num_nodes, self.dim_raw_message),
            dtype=torch.float32, device=device)
        self.mailbox_ts = torch.zeros(
            (num_nodes,), dtype=torch.float32, device=device)
        self.device = device

        # TODO: implement this
        self.pin_memory = pin_memory
        self.shared_memory = shared_memory

    def reset(self):
        """
        Reset the memory and the mailbox.
        """
        self.node_memory.fill_(0)
        self.node_memory_ts.fill_(0)
        self.mailbox.fill_(0)
        self.mailbox_ts.fill_(0)

    def resize(self, num_nodes):
        """
        Resize the memory and the mailbox.

        Args:
            num_nodes: number of nodes in the graph
        """
        if num_nodes <= self.num_nodes:
            return

        self.node_memory.resize_(num_nodes, self.dim_memory)
        self.node_memory_ts.resize_(num_nodes)
        self.mailbox.resize_(num_nodes, self.dim_raw_message)
        self.mailbox_ts.resize_(num_nodes,)

        # fill zeros for the new nodes
        self.node_memory[self.num_nodes:].fill_(0)
        self.node_memory_ts[self.num_nodes:].fill_(0)
        self.mailbox[self.num_nodes:].fill_(0)
        self.mailbox_ts[self.num_nodes:].fill_(0)

        self.num_nodes = num_nodes

    def prepare_input(self, b: DGLBlock):
        """
        Prepare the input for the memory module.

        Args:
          b: sampled message flow graph (mfg), where
                `b.num_dst_nodes()` is the number of target nodes to sample, 
                `b.srcdata['ID']` is the node IDs of all nodes, and 
                `b.srcdata['ts']` is the time stamps of all nodes.
        """
        device = b.device
        num_dst_nodes = b.num_dst_nodes()
        target_nodes = b.srcdata['ID'][:num_dst_nodes]
        all_nodes = b.srcdata['ID']
        assert isinstance(all_nodes, torch.Tensor)

        b.srcdata['mem'] = self.node_memory[all_nodes].to(device)
        b.dstdata['mem_ts'] = self.node_memory_ts[target_nodes].to(device)
        b.dstdata['mail_ts'] = self.mailbox_ts[target_nodes].to(device)

        if self.use_time_enc:
            target_node_ts = b.srcdata['ts'][:num_dst_nodes]
            time_feat = self.time_enc(target_node_ts - b.dstdata['mem_ts'])
        else:
            # dummy time features
            time_feat = torch.zeros(num_dst_nodes, 0, device=device)

        mem_input = self.mailbox[target_nodes].to(device)
        b.dstdata['mem_input'] = torch.cat([mem_input, time_feat], dim=1)

    def update_memory(self, last_updated_nid: torch.Tensor,
                      last_updated_memory: torch.Tensor,
                      last_updated_ts: torch.Tensor):
        """
        Update the memory of last updated nodes.

        Args:
            last_updated_nid: node IDs of the nodes to update
            last_updated_memory: new memory of the nodes
            last_updated_ts: new timestamp of the nodes
        """
        last_updated_nid = last_updated_nid.to(self.device)
        last_updated_memory = last_updated_memory.to(self.device)
        last_updated_ts = last_updated_ts.to(self.device)
        self.node_memory[last_updated_nid] = last_updated_memory
        self.node_memory_ts[last_updated_nid] = last_updated_ts

    def update_mailbox(self, last_updated_nid: torch.Tensor,
                       last_updated_memory: torch.Tensor,
                       last_updated_ts: torch.Tensor,
                       edge_feats: Optional[torch.Tensor] = None):
        """
        Update the mailbox of last updated nodes.

        Args:
            last_updated_nid: node IDs of the nodes to update
            last_updated_memory: new memory of the nodes
            last_updated_ts: new timestamp of the nodes
            edge_feats: edge features of the nodes
        """
        last_updated_nid = last_updated_nid.to(self.device)
        last_updated_memory = last_updated_memory.to(self.device)
        last_updated_ts = last_updated_ts.to(self.device)

        if edge_feats is None:
            # dummy edge features
            edge_feats = torch.zeros(
                last_updated_nid.shape[0]//3, 0, device=self.device)

        edge_feats = edge_feats.to(self.device)

        src, dst, _ = last_updated_nid.tensor_split(3)
        mem_src, mem_dst, _ = last_updated_memory.tensor_split(3)

        src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
        dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
        mail = torch.cat([src_mail, dst_mail],
                         dim=1).reshape(-1, src_mail.shape[1])
        nid = torch.cat(
            [src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
        mail_ts = last_updated_ts[:len(nid)]

        # find unique nid to update mailbox
        uni, inv = torch.unique(nid, return_inverse=True)
        perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
        nid = nid[perm]
        mail = mail[perm]
        mail_ts = mail_ts[perm]

        # update mailbox
        self.mailbox[nid] = mail
        self.mailbox_ts[nid] = mail_ts
