import pandas as pd
import numpy as np
import torch
from .dynamic_graph import DynamicGraph
import os

def load_graph(dataset: str) -> pd.DataFrame:
    dir = os.path.dirname(__file__)
    df = pd.read_csv(dir + '/data/{}/edges.csv'.format(dataset))
    return df

def build_dynamic_graph(dataset: str = "REDDIT", block_size: int = 1024) -> DynamicGraph:
    block_size = 1024
    df = load_graph(dataset)
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]
   
    train_src = torch.tensor(df['src'][:train_edge_end], dtype=torch.long)
    train_dst = torch.tensor(df['dst'][:train_edge_end], dtype=torch.long)
    train_ts = torch.tensor(df['time'][:train_edge_end], dtype=torch.float32)

    dgraph = DynamicGraph(block_size=block_size)
    dgraph.add_edges(train_src, train_dst, train_ts)

    return dgraph
