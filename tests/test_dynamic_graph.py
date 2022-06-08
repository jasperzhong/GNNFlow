import unittest

import numpy as np

import dgnn
from dgnn import DynamicGraph
from dgnn import InsertionPolicy
from parameterized import parameterized
import itertools

class TestDynamicGraph(unittest.TestCase):
    # def test_add_edges_sorted_by_timestamp(self):
    #     """
    #     Test that adding edges sorted by timestamps works.
    #     """
    #     dgraph = DynamicGraph()

    #     source_vertices = np.array(
    #         [0, 0, 0, 1, 1, 1, 2, 2, 2]).astype(np.int64)
    #     target_vertices = np.array(
    #         [1, 2, 3, 1, 2, 3, 1, 2, 3]).astype(np.int64)
    #     timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]).astype(np.float32)
    #     dgraph.add_edges(source_vertices, target_vertices,
    #                      timestamps, add_reverse=False)
    #     self.assertEqual(dgraph.num_edges(), 9)
    #     self.assertEqual(dgraph.num_vertices(), 4)
    #     self.assertEqual(dgraph.out_degree(0), 3)
    #     self.assertEqual(dgraph.out_degree(1), 3)
    #     self.assertEqual(dgraph.out_degree(2), 3)
    #     self.assertEqual(dgraph.out_degree(3), 0)

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
    #         0)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [2, 1, 0])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
    #         1)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [5, 4, 3])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
    #         2)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [8, 7, 6])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
    #         3)
    #     self.assertEqual(target_vertices.tolist(), [])
    #     self.assertEqual(timestamps.tolist(), [])
    #     self.assertEqual(edge_ids.tolist(), [])
    #     print("Test add edges sorted by timestamps passed.")

    # def test_add_edges_sorted_by_timestamps_add_reverse(self):
    #     """
    #     Test that adding edges sorted by timestamps works.
    #     """
    #     dgraph = DynamicGraph()
    #     source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]).astype(np.int64)
    #     target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]).astype(np.int64)
    #     timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]).astype(np.float32)
    #     dgraph.add_edges(source_vertices, target_vertices, timestamps)
    #     self.assertEqual(dgraph.num_edges(), 18)
    #     self.assertEqual(dgraph.num_vertices(), 4)
    #     self.assertEqual(dgraph.out_degree(0), 3)
    #     self.assertEqual(dgraph.out_degree(1), 6)
    #     self.assertEqual(dgraph.out_degree(2), 6)
    #     self.assertEqual(dgraph.out_degree(3), 3)

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
    #         0)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [2, 1, 0])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
    #         1)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 2, 1, 0, 1])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 0, 0, 0, 0])
    #     self.assertEqual(edge_ids.tolist(), [5, 4, 6, 3, 0, 3])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
    #         2)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1, 0, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 1, 1, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [8, 7, 4, 1, 7, 6])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
    #         3)
    #     self.assertEqual(target_vertices.tolist(), [2, 1, 0])
    #     self.assertEqual(timestamps.tolist(), [2, 2, 2])
    #     self.assertEqual(edge_ids.tolist(), [8, 5, 2])
    #     print("Test add edges sorted by timestamps passed (add reverse).")

    # def test_add_edges_unsorted(self):
    #     """
    #     Test that adding edges unsorted works.
    #     """
    #     dgraph = DynamicGraph()
    #     source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    #     target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    #     timestamps = np.array([2, 1, 0, 2, 1, 0, 2, 1, 0])
    #     dgraph.add_edges(source_vertices, target_vertices, timestamps, False)
    #     self.assertEqual(dgraph.num_edges(), 9)
    #     self.assertEqual(dgraph.num_vertices(), 4)
    #     self.assertEqual(dgraph.out_degree(0), 3)
    #     self.assertEqual(dgraph.out_degree(1), 3)
    #     self.assertEqual(dgraph.out_degree(2), 3)
    #     self.assertEqual(dgraph.out_degree(3), 0)

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(0)
    #     self.assertEqual(target_vertices.tolist(), [1, 2, 3])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [0, 1, 2]) 

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(1)
    #     self.assertEqual(target_vertices.tolist(), [1, 2, 3])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [3, 4, 5])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(2)
    #     self.assertEqual(target_vertices.tolist(), [1, 2, 3])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [6, 7, 8])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(3)
    #     self.assertEqual(target_vertices.tolist(), [])
    #     self.assertEqual(timestamps.tolist(), [])
    #     self.assertEqual(edge_ids.tolist(), [])
    #     print("Test add edges unsorted passed.")

    # def test_add_edges_multiple_times_insert(self):
    #     """
    #     Test that adding edges multiple times works.
    #     """
    #     dgraph = DynamicGraph()
    #     source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    #     target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    #     timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    #     dgraph.add_edges(source_vertices, target_vertices, timestamps, False)
    #     self.assertEqual(dgraph.num_edges(), 9)
    #     self.assertEqual(dgraph.num_vertices(), 4)
    #     self.assertEqual(dgraph.out_degree(0), 3)
    #     self.assertEqual(dgraph.out_degree(1), 3)
    #     self.assertEqual(dgraph.out_degree(2), 3)
    #     self.assertEqual(dgraph.out_degree(3), 0)

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(0)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [2, 1, 0])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(1)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [5, 4, 3])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(2)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [8, 7, 6])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(3)
    #     self.assertEqual(target_vertices.tolist(), [])
    #     self.assertEqual(timestamps.tolist(), [])
    #     self.assertEqual(edge_ids.tolist(), [])

    #     # edges with newer timestamps should be added
    #     source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    #     target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    #     timestamps = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5])
    #     dgraph.add_edges(source_vertices, target_vertices, timestamps, False)
    #     self.assertEqual(dgraph.num_edges(), 18)
    #     self.assertEqual(dgraph.num_vertices(), 4)
    #     self.assertEqual(dgraph.out_degree(0), 6)
    #     self.assertEqual(dgraph.out_degree(1), 6)
    #     self.assertEqual(dgraph.out_degree(2), 6)
    #     self.assertEqual(dgraph.out_degree(3), 0)

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(0)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [11, 10, 9, 2, 1, 0])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(1)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [14, 13, 12, 5, 4, 3])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(2)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [17, 16, 15, 8, 7, 6])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(3)
    #     self.assertEqual(target_vertices.tolist(), [])
    #     self.assertEqual(timestamps.tolist(), [])
    #     self.assertEqual(edge_ids.tolist(), [])
    #     print("Test add edges multiple times passed. (insert policy)")

    # def test_add_edges_multiple_times_replace(self):
    #     """
    #     Test that adding edges multiple times works.
    #     """
    #     dgraph = DynamicGraph(insertion_policy=InsertionPolicy.replace)
    #     source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    #     target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    #     timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    #     dgraph.add_edges(source_vertices, target_vertices, timestamps, False)
    #     self.assertEqual(dgraph.num_edges(), 9)
    #     self.assertEqual(dgraph.num_vertices(), 4)
    #     self.assertEqual(dgraph.out_degree(0), 3)
    #     self.assertEqual(dgraph.out_degree(1), 3)
    #     self.assertEqual(dgraph.out_degree(2), 3)
    #     self.assertEqual(dgraph.out_degree(3), 0)

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(0)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [2, 1, 0])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(1)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [5, 4, 3])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(2)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [8, 7, 6])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(3)
    #     self.assertEqual(target_vertices.tolist(), [])
    #     self.assertEqual(timestamps.tolist(), [])
    #     self.assertEqual(edge_ids.tolist(), [])

    #     # edges with newer timestamps should be added
    #     source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    #     target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    #     timestamps = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5])
    #     dgraph.add_edges(source_vertices, target_vertices, timestamps, False)
    #     self.assertEqual(dgraph.num_edges(), 18)
    #     self.assertEqual(dgraph.num_vertices(), 4)
    #     self.assertEqual(dgraph.out_degree(0), 6)
    #     self.assertEqual(dgraph.out_degree(1), 6)
    #     self.assertEqual(dgraph.out_degree(2), 6)
    #     self.assertEqual(dgraph.out_degree(3), 0)

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(0)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [11, 10, 9, 2, 1, 0])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(1)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [14, 13, 12, 5, 4, 3])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(2)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [17, 16, 15, 8, 7, 6])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(3)
    #     self.assertEqual(target_vertices.tolist(), [])
    #     self.assertEqual(timestamps.tolist(), [])
    #     self.assertEqual(edge_ids.tolist(), [])
    #     print("Test add edges multiple times passed. (replace policy)")

    # # def test_add_old_edges(self):
    # #     """
    # #     Test if raise an exception when adding edges with timestmaps that are 
    # #     smaller than the current timestamps.
    # #     """
    # #     dgraph = DynamicGraph()
    # #     source_vertices = np.array([0, 1, 2])
    # #     target_vertices = np.array([1, 2, 3])
    # #     timestamps = np.array([0, 1, 2])
    # #     dgraph.add_edges(source_vertices, target_vertices, timestamps, False)

    # #     source_vertices = np.array([0])
    # #     target_vertices = np.array([1])
    # #     timestamps = np.array([0])
    # #     with self.assertRaises(ValueError):
    # #         dgraph.add_edges(source_vertices, target_vertices, timestamps)

    # #     print("Test add old edges passed.")

    # def test_insertion_policy_replace(self):
    #     """
    #     Test if the "replace" insertion policy works.
    #     """
    #     dgraph = DynamicGraph(insertion_policy=InsertionPolicy.replace)
    #     source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    #     target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    #     timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    #     dgraph.add_edges(source_vertices, target_vertices, timestamps, False)

    #     source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    #     target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    #     timestamps = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5])
    #     dgraph.add_edges(source_vertices, target_vertices, timestamps, False)

    #     self.assertEqual(dgraph.num_edges(), 18)
    #     self.assertEqual(dgraph.num_vertices(), 4)
    #     self.assertEqual(dgraph.out_degree(0), 6)
    #     self.assertEqual(dgraph.out_degree(1), 6)
    #     self.assertEqual(dgraph.out_degree(2), 6)
    #     self.assertEqual(dgraph.out_degree(3), 0)

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(0)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [11, 10, 9, 2, 1, 0])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(1)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [14, 13, 12, 5, 4, 3])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(2)
    #     self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
    #     self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
    #     self.assertEqual(edge_ids.tolist(), [17, 16, 15, 8, 7, 6])

    #     target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(3)
    #     self.assertEqual(target_vertices.tolist(), [])
    #     self.assertEqual(timestamps.tolist(), [])
    #     self.assertEqual(edge_ids.tolist(), [])

    #     print("Test replace insertion policy passed.")

    # @parameterized.expand(itertools.product([32, 64], [2048, 2304], [InsertionPolicy.insert, InsertionPolicy.replace]))
    def test_swap(self, min_block_size=32, max_gpu_pool_size=2048, insertion_policy=InsertionPolicy.insert):
        """
        Test if the swap blocks to CPU works well.
        """
        dgraph = DynamicGraph(min_block_size=32, max_gpu_pool_size=2048, insertion_policy=InsertionPolicy.insert)
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices, timestamps, False)
        self.assertEqual(dgraph.num_edges(), 9)
        self.assertEqual(dgraph.num_vertices(), 4)
        self.assertEqual(dgraph.out_degree(0), 3)
        self.assertEqual(dgraph.out_degree(1), 3)
        self.assertEqual(dgraph.out_degree(2), 3)
        self.assertEqual(dgraph.out_degree(3), 0)

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(0)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [2, 1, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(1)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [5, 4, 3])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(2)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [8, 7, 6])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])
        self.assertEqual(edge_ids.tolist(), [])

        print("First Add pass")

        # edges with newer timestamps should be added
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 4])
        timestamps = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5, 6])
        dgraph.add_edges(source_vertices, target_vertices, timestamps, False)
        self.assertEqual(dgraph.num_edges(), 19)
        self.assertEqual(dgraph.num_vertices(), 4)
        self.assertEqual(dgraph.out_degree(0), 7)
        self.assertEqual(dgraph.out_degree(1), 6)
        self.assertEqual(dgraph.out_degree(2), 6)
        self.assertEqual(dgraph.out_degree(3), 0)

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(0)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [18, 11, 10, 9, 2, 1, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(1)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [14, 13, 12, 5, 4, 3])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(2)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [17, 16, 15, 8, 7, 6])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])
        self.assertEqual(edge_ids.tolist(), [])
        print("Test swap pass")

if __name__ == '__main__':
    unittest.main()
