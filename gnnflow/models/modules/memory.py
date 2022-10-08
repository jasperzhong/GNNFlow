"""
This code is based on the implementation of TGL's memory module.

Implementation at:
    https://github.com/amazon-research/tgl/blob/main/memorys.py
"""
from typing import Dict, Optional, Union

import torch
import torch.distributed
from dgl.heterograph import DGLBlock

from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array


class Memory:
    """
    Memory module proposed by TGN
    """

    def __init__(self, num_nodes: int, dim_edge: int, dim_memory: int,
                 device: Union[torch.device, str] = 'cpu',
                 shared_memory: bool = False):
        """
        Args:
            num_nodes: number of nodes in the graph
            dim_edge: dimension of the edge features
            dim_time: dimension of the time encoding
            dim_memory: dimension of the output of the memory
            device: device to store the memory
            shared_memory: whether to store in shared memory (for multi-GPU training)
        """
        if shared_memory:
            device = 'cpu'

        self.num_nodes = num_nodes
        self.dim_edge = dim_edge
        self.dim_memory = dim_memory
        # raw message: (src_memory, dst_memory, edge_feat)
        self.dim_raw_message = 2 * dim_memory + dim_edge
        self.device = device

        if shared_memory:
            local_world_size = torch.cuda.device_count()
            local_rank = torch.distributed.get_rank() % local_world_size
        else:
            local_world_size = 1
            local_rank = 0

        if not shared_memory:
            self.node_memory = torch.zeros(
                (num_nodes, dim_memory), dtype=torch.float32, device=device)
            self.node_memory_ts = torch.zeros(
                num_nodes, dtype=torch.float32, device=device)
            self.mailbox = torch.zeros(
                (num_nodes, self.dim_raw_message),
                dtype=torch.float32, device=device)
            self.mailbox_ts = torch.zeros(
                (num_nodes,), dtype=torch.float32, device=device)
        else:
            if local_rank == 0:
                self.node_memory = create_shared_mem_array(
                    'node_memory', (num_nodes, dim_memory), dtype=torch.float32)
                self.node_memory_ts = create_shared_mem_array(
                    'node_memory_ts', (num_nodes,), dtype=torch.float32)
                self.mailbox = create_shared_mem_array(
                    'mailbox', (num_nodes, self.dim_raw_message),
                    dtype=torch.float32)
                self.mailbox_ts = create_shared_mem_array(
                    'mailbox_ts', (num_nodes,), dtype=torch.float32)

                self.node_memory.zero_()
                self.node_memory_ts.zero_()
                self.mailbox.zero_()
                self.mailbox_ts.zero_()

            torch.distributed.barrier()

            if local_rank != 0:
                # NB: `num_nodes` should be same for all local processes because
                # they share the same local graph
                self.node_memory = get_shared_mem_array(
                    'node_memory', (num_nodes, dim_memory), torch.float32)
                self.node_memory_ts = get_shared_mem_array(
                    'node_memory_ts', (num_nodes,), torch.float32)
                self.mailbox = get_shared_mem_array(
                    'mailbox', (num_nodes, self.dim_raw_message), torch.float32)
                self.mailbox_ts = get_shared_mem_array(
                    'mailbox_ts', (num_nodes,), torch.float32)

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

    def backup(self) -> Dict:
        """
        Backup the current memory and mailbox.
        """
        return {
            'node_memory': self.node_memory.clone(),
            'node_memory_ts': self.node_memory_ts.clone(),
            'mailbox': self.mailbox.clone(),
            'mailbox_ts': self.mailbox_ts.clone(),
        }

    def restore(self, backup: Dict):
        """
        Restore the memory and mailbox from the backup.

        Args:
            backup: backup of the memory and mailbox
        """
        self.node_memory.copy_(backup['node_memory'])
        self.node_memory_ts.copy_(backup['node_memory_ts'])
        self.mailbox.copy_(backup['mailbox'])
        self.mailbox_ts.copy_(backup['mailbox_ts'])

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
        b.dstdata['mem_input'] = self.mailbox[target_nodes].to(device)

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
                       edge_feats: Optional[torch.Tensor] = None,
                       neg_sample_ratio: int = 1):
        """
        Update the mailbox of last updated nodes.

        Args:
            last_updated_nid: node IDs of the nodes to update
            last_updated_memory: new memory of the nodes
            last_updated_ts: new timestamp of the nodes
            edge_feats: edge features of the nodes
            neg_sample_ratio: negative sampling ratio
        """
        last_updated_nid = last_updated_nid.to(self.device)
        last_updated_memory = last_updated_memory.to(self.device)
        last_updated_ts = last_updated_ts.to(self.device)

        split_chunks = 2 + neg_sample_ratio
        if edge_feats is None:
            # dummy edge features
            edge_feats = torch.zeros(
                last_updated_nid.shape[0] // split_chunks, self.dim_edge,
                device=self.device)

        edge_feats = edge_feats.to(self.device)

        src, dst, *_ = last_updated_nid.tensor_split(split_chunks)
        mem_src, mem_dst, *_ = last_updated_memory.tensor_split(split_chunks)

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
