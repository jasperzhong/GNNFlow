import pandas as pd
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

def build_dynamic_graph(data_dir: str = None, dataset: str = "REDDIT", block_size: int = 1024) -> DynamicGraph:
    df = load_graph(data_dir, dataset)
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]
   
    train_src = torch.tensor(df['src'][:train_edge_end], dtype=torch.long)
    train_dst = torch.tensor(df['dst'][:train_edge_end], dtype=torch.long)
    train_ts = torch.tensor(df['time'][:train_edge_end], dtype=torch.float32)

    dgraph = DynamicGraph(block_size=block_size)
    dgraph.add_edges(train_src, train_dst, train_ts)

    return dgraph
