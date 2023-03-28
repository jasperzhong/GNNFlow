"""
This code is based on the implementation of TGL's memory module.

Implementation at:
    https://github.com/amazon-research/tgl/blob/main/memorys.py
"""
import logging
import os
from threading import Lock
from typing import Dict, Optional, Union

import torch
import torch.distributed
from dgl.heterograph import DGLBlock
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array

from gnnflow.distributed.kvstore import KVStoreClient
import gnnflow.utils as utils


class Memory:
    """
    Memory module proposed by TGN
    """

    def __init__(self, num_nodes: int, dim_edge: int, dim_memory: int,
                 device: Union[torch.device, str] = 'cpu',
                 shared_memory: bool = False,
                 kvstore_client: Optional[KVStoreClient] = None):
        """
        Args:
            num_nodes: number of nodes in the graph
            dim_edge: dimension of the edge features
            dim_time: dimension of the time encoding
            dim_memory: dimension of the output of the memory
            device: device to store the memory
            shared_memory: whether to store in shared memory (for multi-GPU training)
            kvstore_client: The KVStore_Client for fetching memorys when using partition
        """
        if shared_memory:
            device = 'cpu'

        self.num_nodes = num_nodes
        self.dim_edge = dim_edge
        self.dim_memory = dim_memory
        # raw message: (src_memory, dst_memory, edge_feat)
        self.dim_raw_message = 2 * dim_memory + dim_edge
        self.device = device

        self.kvstore_client = kvstore_client
        self.partition = self.kvstore_client != None
        self.distributed = shared_memory

        self.lock = Lock()
        self.i = 0

        # if not partition, not need to use kvstore_client
        if not self.partition:
            if shared_memory:
                local_rank = utils.local_rank()
            else:
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
        if self.partition:
            self.kvstore_client.reset_memory()
        else:
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
        all_nodes = b.srcdata['ID']
        assert isinstance(all_nodes, torch.Tensor)

        all_nodes_unique, inv = torch.unique(
            all_nodes.cpu(), return_inverse=True)

        # with self.lock:
        if self.partition:
            # unique all nodes
            pulled_memory = self.kvstore_client.pull(
                all_nodes_unique, mode='memory')
            mem = pulled_memory[0].to(device)
            mem_ts = pulled_memory[1].to(device)
            mail = pulled_memory[2].to(device)
            mail_ts = pulled_memory[3].to(device)
        else:
            mem = self.node_memory[all_nodes_unique].to(device)
            mem_ts = self.node_memory_ts[all_nodes_unique].to(device)
            mail = self.mailbox[all_nodes_unique].to(device)
            mail_ts = self.mailbox_ts[all_nodes_unique].to(device)

        b.srcdata['mem'] = mem[inv]
        b.srcdata['mem_ts'] = mem_ts[inv]
        b.srcdata['mail_ts'] = mail_ts[inv]
        b.srcdata['mem_input'] = mail[inv]
        # if int(os.environ['LOCAL_RANK']) == 0:
        # logging.info('fetch {}th mem at iter: {}'.format(self.i, i))
        # if int(os.environ['LOCAL_RANK']) == 0:
        # logging.info(
        #     'fetch b.srcdata[mem]: {}'.format(b.srcdata['mem']))
        # logging.info('b.srcdata[mem_input]'.format(b.srcdata['mem_input']))

    def update_mem_mail(self, last_updated_nid: torch.Tensor,
                        last_updated_memory: torch.Tensor,
                        last_updated_ts: torch.Tensor,
                        edge_feats: Optional[torch.Tensor] = None,
                        neg_sample_ratio: int = 1):
        """
        Update the mem and mailbox of last updated nodes.

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
        # logging.info('update last updated memory: {}'.format(
        #     last_updated_memory))

        # genereate mail
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

        # nid, mail, mail_ts = self.sync_all_mail(nid, mail, mail_ts)
        # find unique nid to update mailbox
        uni, inv = torch.unique(nid, return_inverse=True)
        perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
        nid_mail = nid[perm]
        mail = mail[perm]
        mail_ts = mail_ts[perm]

        # prepare mem
        num_true_src_dst = last_updated_nid.shape[0] // split_chunks * 2
        nid = last_updated_nid[:num_true_src_dst].to(self.device)
        memory = last_updated_memory[:num_true_src_dst].to(self.device)
        ts = last_updated_ts[:num_true_src_dst].to(self.device)
        # nid, memory, ts = self.sync_all_mail(nid, memory, ts)
        # the nid of mem and mail is different
        # after unique they are the same but perm is still different
        uni, inv = torch.unique(nid, return_inverse=True)
        perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
        nid = nid[perm]
        mem = memory[perm]
        mem_ts = ts[perm]

        # find the global recent
        # Use all_gather to gather the tensors from all the processes
        if self.distributed:
            nid_mails = self.sync(nid_mail)
            mails = self.sync(mail)
            mail_tss = self.sync(mail_ts)
            uni, inv = torch.unique(nid_mails, return_inverse=True)
            perm = torch.arange(
                inv.size(0), dtype=inv.dtype, device=inv.device)
            perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
            nid_mail = nid_mails[perm]
            mail = mails[perm]
            mail_ts = mail_tss[perm]

            nids = self.sync(nid)
            mems = self.sync(mem)
            mem_tss = self.sync(mem_ts)
            uni, inv = torch.unique(nids, return_inverse=True)
            perm = torch.arange(
                inv.size(0), dtype=inv.dtype, device=inv.device)
            perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
            nid = nids[perm]
            mem = mems[perm]
            mem_ts = mem_tss[perm]

        if self.partition:
            # cat the memory first
            all_mem = torch.cat((mem,
                                 mem_ts.unsqueeze(dim=1),
                                 mail,
                                 mail_ts.unsqueeze(dim=1)),
                                dim=1)
            # TODO: mailbox nid is different
            self.kvstore_client.push(nid, all_mem, mode='memory')
        else:
            # update mailbox first
            if not self.distributed or torch.distributed.get_rank() == 0:
                with self.lock:
                    self.mailbox[nid_mail] = mail
                    self.mailbox_ts[nid_mail] = mail_ts
                    # update mem
                    self.node_memory[nid] = mem
                    self.node_memory_ts[nid] = mem_ts
                    # self.i = i
                    # if int(os.environ['LOCAL_RANK']) == 0:
                    # logging.info('update mem at iter: {}'.format(self.i))
                    # if int(os.environ['LOCAL_RANK']) == 0:
                    #     logging.info('fetch self.node_memory_nid {}'.format(
                    #         self.node_memory[nid]))

    def sync(self, tensor: torch.Tensor) -> torch.Tensor:
        world_size = torch.distributed.get_world_size()

        gather_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(
            gather_list, tensor)
        output_tensor = torch.cat(gather_list, dim=0)
        return output_tensor

    def sync_all_mail(self, nid, mail, mail_ts):
        world_size = torch.distributed.get_world_size()

        gather_list = [None for _ in range(world_size)]

        mails = [nid, mail, mail_ts]
        torch.distributed.all_gather_object(
            gather_list, mails)
        nids = torch.cat([gather_list[i][0] for i in range(world_size)], dim=0)
        mails = torch.cat([gather_list[i][1]
                          for i in range(world_size)], dim=0)
        mail_tss = torch.cat([gather_list[i][2]
                             for i in range(world_size)], dim=0)
        # nid_gather_list = [torch.zeros(nid.shape).to(rank)
        #                    for _ in range(world_size)]
        # torch.distributed.all_gather(nid_gather_list, nid)
        # nids = torch.cat(nid_gather_list, dim=0)

        # mail_gather_list = [torch.zeros(mail.shape).to(rank)
        #                     for _ in range(world_size)]
        # torch.distributed.all_gather(mail_gather_list, mail)
        # mails = torch.cat(mail_gather_list, dim=0)

        # mail_ts_gather_list = [torch.zeros(mail_ts.shape).to(rank)
        #                        for _ in range(world_size)]
        # torch.distributed.all_gather(mail_ts_gather_list, mail)
        # mail_tss = torch.cat(mail_gather_list, dim=0)

        return nids, mails, mail_tss
