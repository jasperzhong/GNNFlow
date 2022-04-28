import unittest

import torch

from dgnn.dynamic_graph import DynamicGraph
from dgnn.temporal_sampler import TemporalSampler


class TestTemporalSampler(unittest.TestCase):
    def test_sample_layer(self):
        # build the dynamic graph
        dgraph = DynamicGraph(block_size=1)
        source_vertices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices, timestamps)

        # sample 1-hop neighbors
        sampler = TemporalSampler(dgraph)
        target_vertices = torch.tensor([0, 1, 2])
        blocks = sampler.sample_layer(2, target_vertices,
                                      torch.tensor([1.5, 1.5, 1.5]))
        block = blocks[0]
        self.assertEqual(block.srcdata['ID'].tolist(), [
                         0, 1, 2, 2, 1, 2, 1, 2, 1])
        self.assertEqual(block.srcdata['ts'].tolist(), [
                         1.5, 1.5, 1.5, 1, 0, 1, 0, 1, 0])
        self.assertEqual(block.edata['dt'].tolist(), [
                         0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
        self.assertEqual(block.edata['ID'].tolist(), [1, 0, 4, 3, 7, 6])

        print("Test sample_layer passed")

    def test_sample_multi_layers(self):
        # build the dynamic graph
        dgraph = DynamicGraph(block_size=1)
        source_vertices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices, timestamps)

        # sample 2-hop neighbors
        sampler = TemporalSampler(dgraph)
        target_vertices = torch.tensor([0, 1, 2])
        blocks = sampler.sample([2, 2], target_vertices,
                                torch.tensor([1.5, 1.5, 1.5]))

        block = blocks[1][0]
        self.assertEqual(block.srcdata['ID'].tolist(), [
                         0, 1, 2, 2, 1, 2, 1, 2, 1])
        self.assertEqual(block.srcdata['ts'].tolist(), [
                         1.5, 1.5, 1.5, 1, 0, 1, 0, 1, 0])
        self.assertEqual(block.edata['dt'].tolist(), [
                         0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
        self.assertEqual(block.edata['ID'].tolist(), [1, 0, 4, 3, 7, 6])

        block = blocks[0][0]
        self.assertEqual(block.srcdata['ID'].tolist(), [
            2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1])
        self.assertEqual(block.srcdata['ts'].tolist(), [
            1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
        self.assertEqual(block.edata['dt'].tolist(), [
            0, 1, 0, 0, 1, 0, 0, 1, 0])
        self.assertEqual(block.edata['ID'].tolist(), [
            7, 6, 3, 7, 6, 3, 7, 6, 3])

        print("Test sample_multi_layers passed")
