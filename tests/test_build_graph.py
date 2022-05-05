from distutils.command.build import build
import unittest
import numpy as np
import torch

from dgnn.build_graph import build_dynamic_graph, load_graph

class TestBuildGraph(unittest.TestCase):
    def test_build_graph(self):
        """
        Test building a dynamic graph from edges.csv(REDDIT)
        Only use training data to build a graph
        """
        dgraph = build_dynamic_graph("REDDIT")
        df = load_graph("REDDIT")
        train_edge_end = df[df['ext_roll'].gt(0)].index[0]
        srcs = np.array(df['src'][:train_edge_end], dtype=int)
        srcs = np.unique(srcs)
        
        dsts = np.array(df['dst'][:train_edge_end], dtype=int)
        dsts = np.unique(dsts)
        
        self.assertEqual(dgraph.num_vertices - 1, max(np.max(srcs), np.max(dsts)))
        
        # Test edges
        for src in srcs:
            df = df[:train_edge_end]
            out_edges = np.array(df[df['src'] == src]['dst'], dtype=int)
            ts = np.array(df[df['src'] == src]['time'])
            ts = np.flip(ts)
            
            graph_out_edges, graph_ts, _ = dgraph.get_temporal_neighbors(src)
            self.assertEqual(len(out_edges), len(graph_out_edges))
            self.assertEqual(len(graph_out_edges), dgraph.out_degree(src))
            self.assertEqual(np.allclose(ts, graph_ts), True)
        