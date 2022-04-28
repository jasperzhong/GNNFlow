import unittest

import torch

from dgnn.dynamic_graph import DynamicGraph
from dgnn.temporal_sampler import TemporalSampler


class TestTemporalSampler(unittest.TestCase):
    def test_sampler_layer_for_single_vertex(self):
        dgraph = DynamicGraph(block_size=1)
        source_vertex = 0
        target_vertices = torch.tensor([1, 2, 3])
        timestamps = torch.tensor([0, 1, 2])
        dgraph.add_edges_for_one_vertex(
            source_vertex, target_vertices, timestamps)
        sampler = TemporalSampler(dgraph, fanouts=[2])
        temporal_graph_block = sampler.sample_layer(2, torch.tensor([source_vertex]),
                                                    torch.tensor([1.5]))
        self.assertEqual(temporal_graph_block.source_vertices.tolist(), [0, 0])
        self.assertEqual(temporal_graph_block.target_vertices.tolist(), [2, 1])
        self.assertEqual(temporal_graph_block.timestamps.tolist(), [1, 0])
        print("Test sampler_layer_for_single_vertex passed")

    def test_sampler_layer_for_multiple_vertices(self):
        dgraph = DynamicGraph(block_size=1)
        source_vertices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices, timestamps)
        sampler = TemporalSampler(dgraph, fanouts=[2])
        given_vertices = torch.tensor([0, 1, 2])
        temporal_graph_block = sampler.sample_layer(2, given_vertices,
                                                    torch.tensor([1.5, 1.5, 1.5]))

        self.assertEqual(temporal_graph_block.source_vertices.tolist(), [
                         0, 0, 1, 1, 2, 2])
        self.assertEqual(temporal_graph_block.target_vertices.tolist(), [
                         2, 1, 2, 1, 2, 1])
        self.assertEqual(temporal_graph_block.timestamps.tolist(), [
                         1, 0, 1, 0, 1, 0])
        print("Test sampler_layer_for_multiple_vertices passed")
