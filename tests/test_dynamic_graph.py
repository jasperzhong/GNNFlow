import unittest

import numpy as np

import dgnn
from dgnn import DynamicGraph


class TestDynamicGraph(unittest.TestCase):
    def test_add_edges_sorted_by_timestamp(self):
        """
        Test that adding edges sorted by timestamps works.
        """
        dgraph = DynamicGraph()

        source_vertices = np.array(
            [0, 0, 0, 1, 1, 1, 2, 2, 2]).astype(np.int64)
        target_vertices = np.array(
            [1, 2, 3, 1, 2, 3, 1, 2, 3]).astype(np.int64)
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]).astype(np.float32)
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)
        self.assertEqual(dgraph.num_edges(), 9)
        self.assertEqual(dgraph.num_vertices(), 4)
        self.assertEqual(dgraph.out_degree(0), 3)
        self.assertEqual(dgraph.out_degree(1), 3)
        self.assertEqual(dgraph.out_degree(2), 3)
        self.assertEqual(dgraph.out_degree(3), 0)

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            0)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [2, 1, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            1)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [5, 4, 3])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            2)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [8, 7, 6])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])
        self.assertEqual(edge_ids.tolist(), [])
        print("Test add edges sorted by timestamps passed.")

    def test_add_edges_sorted_by_timestamps_add_reverse(self):
        """
        Test that adding edges sorted by timestamps works.
        """
        dgraph = DynamicGraph()
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]).astype(np.int64)
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]).astype(np.int64)
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]).astype(np.float32)
        dgraph.add_edges(source_vertices, target_vertices, timestamps)
        self.assertEqual(dgraph.num_edges(), 18)
        self.assertEqual(dgraph.num_vertices(), 4)
        self.assertEqual(dgraph.out_degree(0), 3)
        self.assertEqual(dgraph.out_degree(1), 6)
        self.assertEqual(dgraph.out_degree(2), 6)
        self.assertEqual(dgraph.out_degree(3), 3)

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            0)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [2, 1, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            1)
        self.assertEqual(target_vertices.tolist(), [3, 2, 2, 1, 0, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0, 0, 0, 0])
        self.assertEqual(edge_ids.tolist(), [5, 4, 6, 3, 0, 3])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            2)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 0, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 1, 1, 1, 0])
        self.assertEqual(edge_ids.tolist(), [8, 7, 4, 1, 7, 6])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            3)
        self.assertEqual(target_vertices.tolist(), [2, 1, 0])
        self.assertEqual(timestamps.tolist(), [2, 2, 2])
        self.assertEqual(edge_ids.tolist(), [8, 5, 2])
        print("Test add edges sorted by timestamps passed (add reverse).")
