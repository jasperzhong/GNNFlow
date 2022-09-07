import os
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from dgl.heterograph import DGLBlock

from .dynamic_graph import DynamicGraph


def get_project_root_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset(dataset: str, data_dir: Optional[str] = None) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the dataset and returns the dataframes for the train, validation, test and
    whole dataset.


    Args:
        dataset: the name of the dataset.
        data_dir: the directory where the dataset is stored.

    Returns:
        train_data: the dataframe for the train dataset.
        val_data: the dataframe for the validation dataset.
        test_data: the dataframe for the test dataset.
        full_data: the dataframe for the whole dataset.
    """
    if data_dir is None:
        data_dir = os.path.join(get_project_root_dir(), "data")

    path = os.path.join(data_dir, dataset, 'edges.csv')
    if not os.path.exists(path):
        raise ValueError('{} does not exist'.format(path))

    full_data = pd.read_csv(path)
    assert isinstance(full_data, pd.DataFrame)

    full_data.rename(columns={'Unnamed: 0': 'eid'}, inplace=True)
    train_end = full_data['ext_roll'].values.searchsorted(1)
    val_end = full_data['ext_roll'].values.searchsorted(2)
    train_data = full_data[:train_end]
    val_data = full_data[train_end:val_end]
    test_data = full_data[val_end:]
    return train_data, val_data, test_data, full_data


def load_feat(dataset: str, data_dir: Optional[str] = None):
    """
    Loads the node and edge features for the given dataset.

    NB: either node_feats or edge_feats can be None, but not both.

    Args:
        dataset: the name of the dataset.
        data_dir: the directory where the dataset is stored.

    Returns:
        node_feats: the node features. (None if not available)
        edge_feats: the edge features. (None if not available)
    """
    if data_dir is None:
        data_dir = os.path.join(get_project_root_dir(), "data")

    dataset_path = os.path.join(data_dir, dataset)
    node_feat_path = os.path.join(dataset_path, 'node_features.pt')
    edge_feat_path = os.path.join(dataset_path, 'edge_features.pt')

    if not os.path.exists(node_feat_path) and \
            not os.path.exists(edge_feat_path):
        raise ValueError("Both {} and {} do not exist".format(
            node_feat_path, edge_feat_path))

    node_feats = None
    if os.path.exists(node_feat_path):
        node_feats = torch.load(node_feat_path)
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)

    edge_feats = None
    if os.path.exists(edge_feat_path):
        edge_feats = torch.load(edge_feat_path)
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)

    return node_feats, edge_feats


def get_batch(df: pd.DataFrame, batch_size: int):
    group_indexes = list()

    group_indexes.append(np.array(df.index // batch_size))
    for _, rows in df.groupby(
            group_indexes[random.randint(0, len(group_indexes) - 1)]):

        target_nodes = np.concatenate(
            [rows.src.values, rows.dst.values]).astype(
            np.int64)
        ts = np.concatenate(
            [rows.time.values, rows.time.values]).astype(
            np.float32)

        eid = rows['eid'].values

        yield target_nodes, ts, eid


def build_dynamic_graph(
        dataset_df: pd.DataFrame,
        initial_pool_size: int,
        maximum_pool_size: int,
        mem_resource_type: str,
        minimum_block_size: int,
        blocks_to_preallocate: int,
        insertion_policy: str,
        undirected: bool,
        *args, **kwargs) -> DynamicGraph:
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
        undirected: whether the graph is undirected.
    """
    src = dataset_df['src'].values.astype(np.int64)
    dst = dataset_df['dst'].values.astype(np.int64)
    ts = dataset_df['time'].values.astype(np.float32)

    dgraph = DynamicGraph(
        initial_pool_size,
        maximum_pool_size,
        mem_resource_type,
        minimum_block_size,
        blocks_to_preallocate,
        insertion_policy,
        src, dst, ts,
        undirected)

    return dgraph


def prepare_input(mfgs, node_feats, edge_feats):
    if node_feats is not None:
        for b in mfgs[0]:
            srch = node_feats[b.srcdata['ID']].float()
            b.srcdata['h'] = srch.cuda()

    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                b.edata['f'] = edge_feats[b.edata['ID']].float()
    return mfgs


def mfgs_to_cuda(mfgs: List[List[DGLBlock]], device: Union[str, torch.device]):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to(device)
    return mfgs


def get_pinned_buffers(
        fanouts, sample_history, batch_size, node_feats, edge_feats):
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


class RandEdgeSampler:
    """
    Samples random edges from the graph.
    """

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


class EarlyStopMonitor:
    """
    Monitor the early stopping criteria.
    """

    def __init__(self, max_round=5, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round
