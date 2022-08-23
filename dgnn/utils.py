import os
import random
from typing import Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import torch

from .dynamic_graph import DynamicGraph


def get_project_root_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset(dataset: str, data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                        pd.DataFrame, pd.DataFrame]:
    """
    Loads the dataset and returns the dataframes for the train, validation, test and


    Args:
        dataset: the name of the dataset.
        data_dir: the directory where the dataset is stored.

    Returns:
        train_df: the dataframe for the train set.
        val_df: the dataframe for the validation set.
        test_df: the dataframe for the test set.
        df: the dataframe for the whole dataset.
    """
    if data_dir is None:
        data_dir = os.path.join(get_project_root_dir(), "data")

    path = os.path.join(data_dir, dataset, 'edges.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        raise ValueError('{} does not exist'.format(path))

    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]
    train_df = df[:train_edge_end]
    val_df = df[train_edge_end:val_edge_end]
    test_df = df[val_edge_end:]

    return train_df, val_df, test_df, df


def load_feat(dataset: str, data_dir: Optional[str] = None, rand_de=0, rand_dn=0, edge_count=0, node_count=0) -> Tuple[torch.Tensor, torch.Tensor]:

    if data_dir is None:
        data_dir = os.path.join(get_project_root_dir(), "data")

    dataset_path = os.path.join(data_dir, dataset)

    node_feat_path = os.path.join(dataset_path, 'node_features.pt')
    node_feats = None
    if os.path.exists(node_feat_path):
        node_feats = torch.load(node_feat_path)
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)

    edge_feat_path = os.path.join(dataset_path, 'edge_features.pt')

    edge_feats = None
    if os.path.exists(edge_feat_path):
        edge_feats = torch.load(edge_feat_path)
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)

    if rand_de > 0 and edge_feats is None:
        edge_feats = torch.randn(edge_count, rand_de)
        # if dataset == 'LASTFM':
        #     edge_feats = torch.randn(1293103, rand_de)
        # elif dataset == 'MOOC':
        #     edge_feats = torch.randn(411749, rand_de)
    if rand_dn > 0 and node_feats is None:
        node_feats = torch.randn(node_count, rand_dn)
        # if dataset == 'LASTFM':
        #     node_feats = torch.randn(1980, rand_dn)
        # elif dataset == 'MOOC':
        #     node_feats = torch.randn(7144, rand_dn)

    if node_feats is not None:
        node_feats = node_feats.pin_memory()
    if edge_feats is not None:
        edge_feats = edge_feats.pin_memory()
    return node_feats, edge_feats


def get_batch(df: pd.DataFrame, batch_size: int = 600):
    group_indexes = list()

    group_indexes.append(np.array(df.index // batch_size))
    for _, rows in df.groupby(
            group_indexes[random.randint(0, len(group_indexes) - 1)]):
        # np.random.randint(self.num_nodes, size=n)
        # TODO: wrap a neglink sampler
        length = np.max(np.array(df['dst'], dtype=int))

        target_nodes = np.concatenate(
            [rows.src.values, rows.dst.values]).astype(
            np.int64)
        ts = np.concatenate(
            [rows.time.values, rows.time.values]).astype(
            np.float32)
        # TODO: align with our edge id
        eid = rows['Unnamed: 0'].values

        yield target_nodes, ts, eid


def build_dynamic_graph(
        dataset_df: pd.DataFrame,
        initial_pool_size: int,
        maximum_pool_size: int,
        mem_resource_type: str,
        minimum_block_size: int,
        blocks_to_preallocate: int,
        insertion_policy: str,
        add_reverse: bool = False) -> DynamicGraph:
    """
    Builds a dynamic graph from the given dataframe.

    Args:
        dataset_df: the dataframe for the whole dataset.
        initial_pool_size: optional, int, the initial pool size of the graph.
        maximum_pool_size: optional, int, the maximum pool size of the graph.
        mem_resource_type: optional, str, the memory resource type. 
            valid options: ("cuda", "unified", or "pinned") (case insensitive).
        minimum_block_size: optional, int, the minimum block size of the graph.
        blocks_to_preallocate: optional, int, the number of blocks to preallocate.
        insertion_policy: the insertion policy to use 
            valid options: ("insert" or "replace") (case insensitive).
        add_reverse: optional, bool, whether to add reverse edges.
    """
    src = dataset_df['src'].to_numpy(dtype=np.int64)
    dst = dataset_df['dst'].to_numpy(dtype=np.int64)
    ts = dataset_df['time'].to_numpy(dtype=np.float32)

    dgraph = DynamicGraph(
        initial_pool_size,
        maximum_pool_size,
        mem_resource_type,
        minimum_block_size,
        blocks_to_preallocate,
        insertion_policy,
        src, dst, ts,
        add_reverse)

    return dgraph


def prepare_input(
        mfgs, node_feats, edge_feats, combine_first=False, pinned=False,
        nfeat_buffs=None, efeat_buffs=None, nids=None, eids=None):
    if combine_first:
        for i in range(len(mfgs[0])):
            if mfgs[0][i].num_src_nodes() > mfgs[0][i].num_dst_nodes():
                num_dst = mfgs[0][i].num_dst_nodes()
                ts = mfgs[0][i].srcdata['ts'][num_dst:]
                nid = mfgs[0][i].srcdata['ID'][num_dst:].float()
                nts = torch.stack([ts, nid], dim=1)
                unts, idx = torch.unique(nts, dim=0, return_inverse=True)
                uts = unts[:, 0]
                unid = unts[:, 1]
                # import pdb; pdb.set_trace()
                b = dgl.create_block(
                    (idx + num_dst, mfgs[0][i].edges()[1]),
                    num_src_nodes=unts.shape[0] + num_dst, num_dst_nodes=num_dst,
                    device=torch.device('cuda:0'))
                b.srcdata['ts'] = torch.cat(
                    [mfgs[0][i].srcdata['ts'][:num_dst], uts], dim=0)
                b.srcdata['ID'] = torch.cat(
                    [mfgs[0][i].srcdata['ID'][:num_dst], unid], dim=0)
                b.edata['dt'] = mfgs[0][i].edata['dt']
                b.edata['ID'] = mfgs[0][i].edata['ID']
                mfgs[0][i] = b
    t_idx = 0
    t_cuda = 0
    i = 0
    if node_feats is not None:
        for b in mfgs[0]:
            if pinned:
                if nids is not None:
                    idx = nids[i]
                else:
                    idx = b.srcdata['ID'].cpu().long()
                torch.index_select(node_feats, 0, idx,
                                   out=nfeat_buffs[i][:idx.shape[0]])
                b.srcdata['h'] = nfeat_buffs[i][
                    : idx.shape[0]].cuda(
                    non_blocking=True)
                i += 1
            else:
                srch = node_feats[b.srcdata['ID'].long()].float()
                b.srcdata['h'] = srch.cuda()
    i = 0
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                if b.num_src_nodes() > b.num_dst_nodes():
                    if pinned:
                        if eids is not None:
                            idx = eids[i]
                        else:
                            idx = b.edata['ID'].cpu().long()
                        torch.index_select(
                            edge_feats, 0, idx,
                            out=efeat_buffs[i][: idx.shape[0]])
                        b.edata['f'] = efeat_buffs[i][
                            : idx.shape[0]].cuda(
                            non_blocking=True)
                        i += 1
                    else:
                        b.edata['f'] = edge_feats[b.edata['ID'].long()].float()
    return mfgs


def mfgs_to_cuda(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to('cuda:0')
    return mfgs


def node_to_dgl_blocks(target_nodes, ts, cuda=True):
    target_nodes = torch.tensor(target_nodes)
    ts = torch.tensor(ts)
    mfgs = list()
    b = dgl.create_block(
        ([],
         []),
        num_src_nodes=target_nodes.shape[0],
        num_dst_nodes=target_nodes.shape[0])
    b.srcdata['ID'] = target_nodes
    b.srcdata['ts'] = ts
    if cuda:
        mfgs.insert(0, [b.to('cuda:0')])
    else:
        mfgs.insert(0, [b])
    return mfgs


class NegLinkSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)


def get_pinned_buffers(fanouts, sample_history, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    for i in fanouts:
        limit *= i
        if edge_feats is not None:
            for _ in range(sample_history):
                pinned_efeat_buffs.insert(0, torch.zeros(
                    (limit, edge_feats.shape[1]), pin_memory=True))

    if node_feats is not None:
        for _ in range(sample_history):
            pinned_nfeat_buffs.insert(0, torch.zeros(
                (limit, node_feats.shape[1]), pin_memory=True))

    return pinned_nfeat_buffs, pinned_efeat_buffs

class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)