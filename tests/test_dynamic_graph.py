import unittest

import torch

from dgnn.dynamic_graph import DynamicGraph


class TestDynamicGraph(unittest.TestCase):
    def test_add_edges_for_one_vertex(self):
        """
        Test that adding edges for one vertex works.
        """
        dgraph = DynamicGraph(block_size=1)
        source_vertex = 0
        target_vertices = torch.tensor([1, 2, 3])
        timestamps = torch.tensor([0, 1, 2])
        dgraph.add_edges_for_one_vertex(
            source_vertex, target_vertices, timestamps)
        self.assertEqual(dgraph.num_edges, 3)
        self.assertEqual(dgraph.num_vertices, 4)
        self.assertEqual(dgraph.out_degree(source_vertex), 3)

        target_vertices, timestamps = dgraph.get_neighbors(source_vertex)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])
        print("Test add edges for one vertex passed.")

    def test_add_edges_for_one_vertex_with_duplicate_edges(self):
        """
        Test that adding edges with duplicate edges. 

        Note that duplicate edges are allowed.
        """
        dgraph = DynamicGraph(block_size=1)
        source_vertex = 0
        target_vertices = torch.tensor([1, 2, 3, 2])
        timestamps = torch.tensor([0, 1, 2, 3])
        dgraph.add_edges_for_one_vertex(
            source_vertex, target_vertices, timestamps)
        self.assertEqual(dgraph.num_edges, 4)
        self.assertEqual(dgraph.num_vertices, 4)
        self.assertEqual(dgraph.out_degree(source_vertex), 4)

        target_vertices, timestamps = dgraph.get_neighbors(source_vertex)
        self.assertEqual(target_vertices.tolist(), [2, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [3, 2, 1, 0])
        print("Test add edges for one vertex with duplicate edges passed.")

    def test_add_edges_sorted_by_timestamps(self):
        """
        Test that adding edges sorted by timestamps works.
        """
        dgraph = DynamicGraph(block_size=1)
        source_vertices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices, timestamps)
        self.assertEqual(dgraph.num_edges, 9)
        self.assertEqual(dgraph.num_vertices, 4)
        self.assertEqual(dgraph.out_degree(0), 3)
        self.assertEqual(dgraph.out_degree(1), 3)
        self.assertEqual(dgraph.out_degree(2), 3)
        self.assertEqual(dgraph.out_degree(3), 0)

        target_vertices, timestamps = dgraph.get_neighbors(0)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(1)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(2)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])
        print("Test add edges sorted by timestamps passed.")

    def test_add_edges_unsorted(self):
        """
        Test that adding edges unsorted works.
        """
        dgraph = DynamicGraph(block_size=1)
        source_vertices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = torch.tensor([2, 1, 0, 2, 1, 0, 2, 1, 0])
        dgraph.add_edges(source_vertices, target_vertices, timestamps)
        self.assertEqual(dgraph.num_edges, 9)
        self.assertEqual(dgraph.num_vertices, 4)
        self.assertEqual(dgraph.out_degree(0), 3)
        self.assertEqual(dgraph.out_degree(1), 3)
        self.assertEqual(dgraph.out_degree(2), 3)
        self.assertEqual(dgraph.out_degree(3), 0)

        target_vertices, timestamps = dgraph.get_neighbors(0)
        self.assertEqual(target_vertices.tolist(), [1, 2, 3])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(1)
        self.assertEqual(target_vertices.tolist(), [1, 2, 3])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(2)
        self.assertEqual(target_vertices.tolist(), [1, 2, 3])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])
        print("Test add edges unsorted passed.")

    def test_add_edges_multiple_times(self):
        """
        Test that adding edges multiple times works.
        """
        dgraph = DynamicGraph(block_size=1)
        source_vertices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices, timestamps)
        self.assertEqual(dgraph.num_edges, 9)
        self.assertEqual(dgraph.num_vertices, 4)
        self.assertEqual(dgraph.out_degree(0), 3)
        self.assertEqual(dgraph.out_degree(1), 3)
        self.assertEqual(dgraph.out_degree(2), 3)
        self.assertEqual(dgraph.out_degree(3), 0)

        target_vertices, timestamps = dgraph.get_neighbors(0)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(1)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(2)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1])
        self.assertEqual(timestamps.tolist(), [2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])

        # edges with newer timestamps should be added
        source_vertices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = torch.tensor([3, 4, 5, 3, 4, 5, 3, 4, 5])
        dgraph.add_edges(source_vertices, target_vertices, timestamps)
        self.assertEqual(dgraph.num_edges, 18)
        self.assertEqual(dgraph.num_vertices, 4)
        self.assertEqual(dgraph.out_degree(0), 6)
        self.assertEqual(dgraph.out_degree(1), 6)
        self.assertEqual(dgraph.out_degree(2), 6)
        self.assertEqual(dgraph.out_degree(3), 0)

        target_vertices, timestamps = dgraph.get_neighbors(0)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(1)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(2)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])
        print("Test add edges multiple times passed.")

    def test_add_old_edges(self):
        """
        Test if raise an exception when adding edges with timestmaps that are 
        smaller than the current timestamps.
        """
        dgraph = DynamicGraph(block_size=1)
        source_vertices = torch.tensor([0, 1, 2])
        target_vertices = torch.tensor([1, 2, 3])
        timestamps = torch.tensor([0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices, timestamps)

        source_vertices = torch.tensor([0])
        target_vertices = torch.tensor([1])
        timestamps = torch.tensor([0])
        with self.assertRaises(ValueError):
            dgraph.add_edges(source_vertices, target_vertices, timestamps)

        print("Test add old edges passed.")

    def test_new_insertion_policy(self):
        """
        Test if the "new" insertion policy works.
        """
        dgraph = DynamicGraph(block_size=1, insertion_policy="new")
        source_vertices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices, timestamps)

        source_vertices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = torch.tensor([3, 4, 5, 3, 4, 5, 3, 4, 5])
        dgraph.add_edges(source_vertices, target_vertices, timestamps)

        self.assertEqual(dgraph.num_edges, 18)
        self.assertEqual(dgraph.num_vertices, 4)
        self.assertEqual(dgraph.out_degree(0), 6)
        self.assertEqual(dgraph.out_degree(1), 6)
        self.assertEqual(dgraph.out_degree(2), 6)
        self.assertEqual(dgraph.out_degree(3), 0)

        target_vertices, timestamps = dgraph.get_neighbors(0)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(1)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(2)
        self.assertEqual(target_vertices.tolist(), [3, 2, 1, 3, 2, 1])
        self.assertEqual(timestamps.tolist(), [5, 4, 3, 2, 1, 0])

        target_vertices, timestamps = dgraph.get_neighbors(3)
        self.assertEqual(target_vertices.tolist(), [])
        self.assertEqual(timestamps.tolist(), [])

        print("Test new insertion policy passed.")

    def test_get_neighbors_before_timestamp(self):
        """
        Test if get_neighbors_before_timestamp works.
        """
        dgraph = DynamicGraph(block_size=1)
        source_vertices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices, timestamps)


        target_vertices, timestamps = dgraph.get_neighbors_before_timestamp(0, 1.5)
        self.assertEqual(target_vertices.tolist(), [2, 1])
        self.assertEqual(timestamps.tolist(), [1, 0])
        print("Test out edges before timestamp passed.")

