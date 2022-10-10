import itertools
import unittest

import numpy as np
from parameterized import parameterized

from gnnflow.config import get_default_config
from gnnflow.utils import build_dynamic_graph, load_dataset

MB = 1 << 20
GB = 1 << 30

_, default_config = get_default_config("TGN", "REDDIT")


class TestBuildGraph(unittest.TestCase):

    @parameterized.expand(itertools.product(["cuda", "unified", "pinned", "shared"]))
    def test_build_graph(self, mem_resource_type):
        """
        Test building a dynamic graph from edges.csv(REDDIT)
        Only use training data to build a graph
        """
        train_df, _, _, df = load_dataset(dataset="REDDIT")
        config = default_config.copy()
        config["mem_resource_type"] = mem_resource_type
        config["undirected"] = False
        dgraph = build_dynamic_graph(**config, dataset_df=train_df)

        train_edge_end = df[df['ext_roll'].gt(0)].index[0]
        srcs = np.array(df['src'][:train_edge_end], dtype=int)
        srcs = np.unique(srcs)

        dsts = np.array(df['dst'][:train_edge_end], dtype=int)
        dsts = np.unique(dsts)

        self.assertEqual(dgraph.max_vertex_id(),
                         max(np.max(srcs), np.max(dsts)))

        # Test edges
        for src in srcs:
            df = df[:train_edge_end]
            out_edges = np.array(df[df['src'] == src]['dst'], dtype=int)
            ts = np.array(df[df['src'] == src]['time'])
            ts = np.flip(ts)

            graph_out_edges, graph_ts, _ = dgraph.get_temporal_neighbors(src)
            self.assertEqual(len(out_edges), len(graph_out_edges))
            self.assertEqual(len(graph_out_edges), dgraph.out_degree([src]))
            self.assertTrue(np.allclose(ts, graph_ts))

    @parameterized.expand(itertools.product(["cuda", "unified", "pinned", "shared"]))
    def test_build_graph_undirected(self, mem_resource_type):
        """
        Test building a dynamic graph from edges.csv(REDDIT)
        Only use training data to build a graph
        """
        train_df, _, _, df = load_dataset(dataset="REDDIT")
        config = default_config.copy()
        config["mem_resource_type"] = mem_resource_type
        config["undirected"] = True
        dgraph = build_dynamic_graph(**config, dataset_df=train_df)

        train_edge_end = df[df['ext_roll'].gt(0)].index[0]
        srcs = np.array(df['src'][:train_edge_end], dtype=int)
        srcs = np.unique(srcs)

        dsts = np.array(df['dst'][:train_edge_end], dtype=int)
        dsts = np.unique(dsts)

        srcs = np.concatenate((srcs, dsts))

        self.assertEqual(dgraph.max_vertex_id(),
                         max(np.max(srcs), np.max(dsts)))

        # Test edges
        for src in srcs:
            df = df[:train_edge_end]
            out_edges = np.array(df[df['src'] == src]['dst'], dtype=int)
            out_edges_reverse = np.array(
                df[df['dst'] == src]['src'], dtype=int)
            out_edges = np.concatenate((out_edges, out_edges_reverse))
            ts = np.array(df[df['src'] == src]['time'])
            ts_reverse = np.array(df[df['dst'] == src]['time'])
            ts = np.concatenate((ts, ts_reverse))
            ts = np.flip(ts)

            graph_out_edges, graph_ts, _ = dgraph.get_temporal_neighbors(src)
            self.assertEqual(len(out_edges), len(graph_out_edges))
            self.assertEqual(len(graph_out_edges), dgraph.out_degree([src]))
            self.assertTrue(np.allclose(ts, graph_ts))
