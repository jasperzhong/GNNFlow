from typing import Tuple
import pandas as pd
import numpy as np
import random
import torch
from .dynamic_graph import DynamicGraph
import os


def load_graph(data_dir: str = None, dataset: str = 'REDDIT') -> pd.DataFrame:
    if data_dir is not None:
        if dataset is not None:
            path = os.path.join(data_dir, dataset, 'edges.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
    else:
        data_dir = os.path.dirname(__file__)
        path = data_dir + '/data/{}/edges.csv'.format(dataset)
        if os.path.exists(path):
            df = pd.read_csv(path)
    return df

def load_feat(data_dir: str = None, dataset: str = 'REDDIT', rand_de=0, rand_dn=0) -> Tuple[torch.Tensor, torch.Tensor]:
    node_feats = None
    if data_dir is None:
        data_dir = os.path.dirname(__file__)
        
    dataset_path = os.path.join(data_dir, dataset)

    node_feat_path = os.path.join(dataset_path, 'node_features.pt')
    if os.path.exists(node_feat_path):
        node_feats = torch.load(node_feat_path)
        if node_feats.dtyep == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feat_path = os.path.join(dataset_path, 'node_features.pt')
    if os.path.exists(edge_feat_path):
        edge_feats = torch.load(edge_feat_path)
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    
    if rand_de > 0:
        if dataset == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif dataset == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
    if rand_dn > 0:
        if dataset == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif dataset == 'MOOC':
            edge_feats = torch.randn(7144, rand_dn)
            
    return node_feats, edge_feats

def get_batch(df: pd.DataFrame, batch_size: int = 600) -> Tuple[torch.Tensor, torch.Tensor]:
    group_indexes = list()
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    
    group_indexes.append(np.array(df[:train_edge_end].index // batch_size))
    for _, rows in df[:train_edge_end].groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
        # np.random.randint(self.num_nodes, size=n)
        # TODO: wrap a neglink sampler
        length = np.max(np.array(df[:train_edge_end]['dst'], dtype=int))
        # TODO: eliminate np to tensor
        target_nodes = np.concatenate([rows.src.values, rows.dst.values, np.random.randint(length, size=batch_size)]).astype(int)
        ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
        yield torch.tensor(target_nodes, dtype=torch.long), torch.tensor(ts, dtype=torch.float32)
        
    
def build_dynamic_graph(df: pd.DataFrame, block_size: int = 1024) -> DynamicGraph:

    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]
   
    train_src = torch.tensor(df['src'][:train_edge_end], dtype=torch.long)
    train_dst = torch.tensor(df['dst'][:train_edge_end], dtype=torch.long)
    train_ts = torch.tensor(df['time'][:train_edge_end], dtype=torch.float32)

    dgraph = DynamicGraph(block_size=block_size)
    dgraph.add_edges(train_src, train_dst, train_ts)

    return dgraph
