import itertools
import unittest

import numpy as np
from parameterized import parameterized

from gnnflow import DynamicGraph

MB = 1 << 20
GB = 1 << 30

default_config = {
    "initial_pool_size": 1 * MB,
    "maximum_pool_size": 2 * MB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 64,
    "blocks_to_preallocate": 128,
    "insertion_policy": "insert",
}


class TestDynamicGraph(unittest.TestCase):

    @parameterized.expand(
        itertools.product(["cuda", "unified", "pinned", "shared"]))
    def test_add_edges_sorted_by_timestamp(self, mem_resource_type):
        """
        Test that adding edges sorted by timestamps works.
        """
        config = default_config.copy()
        config["mem_resource_type"] = mem_resource_type
        dgraph = DynamicGraph(**config)

        source_vertices = np.array(
            [0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array(
            [1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)
        self.assertEqual(dgraph.num_edges(), 9)
        self.assertEqual(dgraph.num_vertices(), 4)
        self.assertEqual(dgraph.out_degree(
            [0, 1, 2, 3]).tolist(), [3, 3, 3, 0])

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
        print("Test add edges sorted by timestamps passed. (mem_resource_type: {})".format(
            mem_resource_type))

    @parameterized.expand(
        itertools.product(["cuda", "unified", "pinned", "shared"]))
    def test_add_edges_sorted_by_timestamps_add_reverse(
            self, mem_resource_type):
        """
        Test that adding edges sorted by timestamps works.
        """
        config = default_config.copy()
        config["mem_resource_type"] = mem_resource_type
        dgraph = DynamicGraph(**config)
        source_vertices = np.array(
            [0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array(
            [1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=True)
        self.assertEqual(dgraph.num_edges(), 9)
        self.assertEqual(dgraph.num_vertices(), 4)
        self.assertEqual(dgraph.out_degree(
            [0, 1, 2, 3]).tolist(), [3, 6, 6, 3])

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
        print("Test add edges sorted by timestamps passed (add reverse) (mem_resource_type: {})".format(
            mem_resource_type))

    @parameterized.expand(
        itertools.product(["cuda", "unified", "pinned", "shared"]))
    def test_add_edges_unsorted(self, mem_resource_type):
        """
        Test that adding edges unsorted works.
        """
        config = default_config.copy()
        config["mem_resource_type"] = mem_resource_type
        dgraph = DynamicGraph(**config)
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([2, 1, 0, 2, 1, 0, 2, 1, 0])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)
        self.assertEqual(dgraph.num_edges(), 9)
        self.assertEqual(dgraph.num_vertices(), 4)
        self.assertEqual(dgraph.out_degree(
            [0, 1, 2, 3]).tolist(), [3, 3, 3, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            0)
        self.assertEqual(target_vertices.tolist(), [1, 2, 3])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [0, 1, 2])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            1)
        self.assertEqual(target_vertices.tolist(), [1, 2, 3])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [3, 4, 5])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            2)
        self.assertEqual(target_vertices.tolist(), [1, 2, 3])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [6, 7, 8])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])
        self.assertEqual(edge_ids.tolist(), [])
        print("Test add edges unsorted passed (mem_resource_type: {})".format(
            mem_resource_type))

    @parameterized.expand(
        itertools.product(["cuda", "unified", "pinned", "shared"]))
    def test_add_edges_multiple_times_insert(self, mem_resource_type):
        """
        Test that adding edges multiple times works.
        """
        config = default_config.copy()
        config["minimum_block_size"] = 4
        config["mem_resource_type"] = mem_resource_type
        dgraph = DynamicGraph(**config)
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)
        self.assertEqual(dgraph.num_edges(), 9)
        self.assertEqual(dgraph.num_vertices(), 4)
        self.assertEqual(dgraph.out_degree(
            [0, 1, 2, 3]).tolist(), [3, 3, 3, 0])

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

        # edges with newer timestamps should be added
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)
        self.assertEqual(dgraph.num_edges(), 18)
        self.assertEqual(dgraph.num_vertices(), 4)
        self.assertEqual(dgraph.out_degree(
            [0, 1, 2, 3]).tolist(), [6, 6, 6, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            0)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [11, 10, 9, 2, 1, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            1)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [14, 13, 12, 5, 4, 3])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            2)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [17, 16, 15, 8, 7, 6])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])
        self.assertEqual(edge_ids.tolist(), [])
        print("Test add edges multiple times passed. (insert policy) (mem_resource_type: {})".format(
            mem_resource_type))

    @parameterized.expand(
        itertools.product(["cuda", "unified", "pinned", "shared"]))
    def test_add_edges_multiple_times_replace(self, mem_resource_type):
        """
        Test that adding edges multiple times works.
        """
        config = default_config.copy()
        config["minimum_block_size"] = 4
        config["insertion_policy"] = "replace"
        config["mem_resource_type"] = mem_resource_type
        dgraph = DynamicGraph(**config)
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)
        self.assertEqual(dgraph.num_edges(), 9)
        self.assertEqual(dgraph.num_vertices(), 4)
        self.assertEqual(dgraph.out_degree(
            [0, 1, 2, 3]).tolist(), [3, 3, 3, 0])

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

        # edges with newer timestamps should be added
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)
        self.assertEqual(dgraph.num_edges(), 18)
        self.assertEqual(dgraph.num_vertices(), 4)
        self.assertEqual(dgraph.out_degree(
            [0, 1, 2, 3]).tolist(), [6, 6, 6, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            0)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [11, 10, 9, 2, 1, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            1)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [14, 13, 12, 5, 4, 3])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            2)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [17, 16, 15, 8, 7, 6])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])
        self.assertEqual(edge_ids.tolist(), [])
        print("Test add edges multiple times passed. (replace policy) (mem_resource_type: {})".format(
            mem_resource_type))

    @unittest.skip("Not implemented yet")
    def test_add_old_edges(self):
        """
        Test if raise an exception when adding edges with timestmaps that are
        smaller than the current timestamps.
        """
        config = default_config.copy()
        dgraph = DynamicGraph(**config)
        source_vertices = np.array([0, 1, 2])
        target_vertices = np.array([1, 2, 3])
        timestamps = np.array([0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)

        source_vertices = np.array([0])
        target_vertices = np.array([1])
        timestamps = np.array([0])
        with self.assertRaises(ValueError):
            dgraph.add_edges(source_vertices, target_vertices, timestamps)

        print("Test add old edges passed.")

    @parameterized.expand(
        itertools.product(["cuda", "unified", "pinned", "shared"]))
    def test_insertion_policy_replace(self, mem_resource_type):
        """
        Test if the "replace" insertion policy works.
        """
        config = default_config.copy()
        config["minimum_block_size"] = 4
        config["insertion_policy"] = "replace"
        config["mem_resource_type"] = mem_resource_type
        dgraph = DynamicGraph(**config)
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)

        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)

        self.assertEqual(dgraph.num_edges(), 18)
        self.assertEqual(dgraph.num_vertices(), 4)
        self.assertEqual(dgraph.out_degree(
            [0, 1, 2, 3]).tolist(), [6, 6, 6, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            0)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [11, 10, 9, 2, 1, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            1)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [14, 13, 12, 5, 4, 3])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            2)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [17, 16, 15, 8, 7, 6])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])
        self.assertEqual(edge_ids.tolist(), [])

        print("Test replace insertion policy passed. (mem_resource_type: {})".format(
            mem_resource_type))

    @parameterized.expand(
        itertools.product(["cuda", "unified", "pinned", "shared"]))
    def test_add_edges_with_eids(self, mem_resource_type):
        """
        Test if the "replace" insertion policy works.
        """
        config = default_config.copy()
        config["minimum_block_size"] = 4
        config["mem_resource_type"] = mem_resource_type
        dgraph = DynamicGraph(**config)
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        edge_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, edge_ids, add_reverse=False)

        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5])
        edge_ids = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, edge_ids, add_reverse=False)

        self.assertEqual(dgraph.num_edges(), 18)
        self.assertEqual(dgraph.num_vertices(), 4)
        self.assertEqual(dgraph.out_degree(
            [0, 1, 2, 3]).tolist(), [6, 6, 6, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            0)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [11, 10, 9, 2, 1, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            1)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [14, 13, 12, 5, 4, 3])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            2)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [17, 16, 15, 8, 7, 6])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])
        self.assertEqual(edge_ids.tolist(), [])

        print("Test add edges with eids passed. (mem_resource_type: {})".format(
            mem_resource_type))

    @parameterized.expand(
        itertools.product(["cuda", "unified", "pinned", "shared"]))
    def test_add_edges_with_noncontinous_eids(self, mem_resource_type):
        """
        Test if the "replace" insertion policy works.
        """
        config = default_config.copy()
        config["minimum_block_size"] = 4
        config["mem_resource_type"] = mem_resource_type
        dgraph = DynamicGraph(**config)
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        edge_ids = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, edge_ids, add_reverse=False)

        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5])
        edge_ids = np.array([17, 19, 21, 23, 25, 27, 29, 31, 33])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, edge_ids, add_reverse=False)

        self.assertEqual(dgraph.num_edges(), 18)
        self.assertEqual(dgraph.num_vertices(), 4)
        self.assertEqual(dgraph.out_degree(
            [0, 1, 2, 3]).tolist(), [6, 6, 6, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            0)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [21, 19, 17, 4, 2, 0])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            1)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [27, 25, 23, 10, 8, 6])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            2)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(edge_ids.tolist(), [33, 31, 29, 16, 14, 12])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])
        self.assertEqual(edge_ids.tolist(), [])

        print("Test add edges with non-continous eids passed. (mem_resource_type: {})".format(
            mem_resource_type))

    @parameterized.expand(
        itertools.product(["pinned"], [True, False]))
    def test_offload_old_blocks(self, mem_resource_type, to_file):
        """
        Test if the "offload_old_blocks(timestamp: float)" works
        """
        config = default_config.copy()
        config["minimum_block_size"] = 4
        config["mem_resource_type"] = mem_resource_type
        dgraph = DynamicGraph(**config)
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        edge_ids = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, edge_ids, add_reverse=False)

        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5])
        edge_ids = np.array([17, 19, 21, 23, 25, 27, 29, 31, 33])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, edge_ids, add_reverse=False)

        num_blocks = dgraph.offload_old_blocks(
            3.5, to_file)  # the first block is offloaded
        print("offloaded {} blocks".format(num_blocks))

        self.assertEqual(dgraph.num_edges(), 6)
        self.assertEqual(dgraph.num_vertices(), 4)

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            0)
        self.assertEqual(target_vertices.tolist(), [3, 2])
        self.assertEqual(timestamps.tolist(), [5, 4])
        self.assertEqual(edge_ids.tolist(), [21, 19])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            1)
        self.assertEqual(target_vertices.tolist(), [3, 2])
        self.assertEqual(timestamps.tolist(), [5, 4])
        self.assertEqual(edge_ids.tolist(), [27, 25])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            2)
        self.assertEqual(target_vertices.tolist(), [3, 2])
        self.assertEqual(timestamps.tolist(), [5, 4])
        self.assertEqual(edge_ids.tolist(), [33, 31])

        target_vertices, timestamps, edge_ids = dgraph.get_temporal_neighbors(
            3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])
        self.assertEqual(edge_ids.tolist(), [])

        print("Test offload old blocks passed. (mem_resource_type: {})".format(
            mem_resource_type))


if __name__ == '__main__':
    unittest.main()
