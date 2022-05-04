import pandas as pd
import numpy as np
import torch
from .dynamic_graph import DynamicGraph

def load_graph(d: str) -> pd.DataFrame:
    df = pd.read_csv('/home/gmsheng/repos/dynamic-graph-neural-network/dgnn/data/{}/edges.csv'.format(d))
    return df

def build_dynamic_graph(dataset: str = "REDDIT", block_size: int = 1024) -> DynamicGraph:
    block_size = 1024
    batch_size = 600
    df = load_graph(dataset)
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]
    group_indexes = list()
    group_indexes.append(np.array(df[:train_edge_end].index // batch_size))

    train_src = torch.tensor(df['src'][:train_edge_end], dtype=int)
    train_dst = torch.tensor(df['dst'][:train_edge_end], dtype=int)
    train_ts = torch.tensor(df['time'][:train_edge_end]) # float64

    dgraph = DynamicGraph(block_size=block_size)
    dgraph.add_edges(train_src, train_dst, train_ts)

    return dgraph
