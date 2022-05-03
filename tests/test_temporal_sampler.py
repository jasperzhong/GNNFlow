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
        sampler = TemporalSampler(dgraph, [2])
        target_vertices = torch.tensor([0, 1, 2])
        blocks = sampler._sample_layer_from_root(2, target_vertices,
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

    def test_sampler_layer_with_duplicate_vertices(self):
        # build the dynamic graph
        dgraph = DynamicGraph(block_size=1)
        source_vertices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices, timestamps)

        # sample 1-hop neighbors
        sampler = TemporalSampler(dgraph, [2])
        target_vertices = torch.tensor([0, 1, 2, 0])
        blocks = sampler._sample_layer_from_root(2, target_vertices,
                                                 torch.tensor([1.5, 1.5, 1.5, 1.5]))
        block = blocks[0]
        self.assertEqual(block.srcdata['ID'].tolist(), [
                         0, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1])
        self.assertEqual(block.srcdata['ts'].tolist(), [
                         1.5, 1.5,1.5, 1.5, 1, 0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(block.edata['dt'].tolist(), [
                         0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
        self.assertEqual(block.edata['ID'].tolist(), [1, 0, 4, 3, 7, 6, 1, 0])

        print("Test sampler_layer_with_duplicate_vertices passed")

    def test_sample_multi_layers(self):
        # build the dynamic graph
        dgraph = DynamicGraph(block_size=1)
        source_vertices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices, timestamps)

        # sample 2-hop neighbors
        sampler = TemporalSampler(dgraph, [2, 2])
        target_vertices = torch.tensor([0, 1, 2])
        blocks = sampler.sample(target_vertices,
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

    def test_sample_multi_snapshots(self):
        # build the dynamic graph
        dgraph = DynamicGraph(block_size=1)
        source_vertices = torch.tensor(
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        target_vertices = torch.tensor(
            [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
        timestamps = torch.tensor(
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
        dgraph.add_edges(source_vertices, target_vertices, timestamps)

        # sample 1-hop neighbors with two snapshots
        sampler = TemporalSampler(dgraph, [2], num_snapshots=2,
                                  snapshot_time_window=1)
        target_vertices = torch.tensor([0, 1, 2])
        blocks = sampler.sample(target_vertices,
                                torch.tensor([5, 5, 5]))

        block = blocks[0][1] # timestamp range: [4, 5]
        self.assertEqual(block.srcdata['ID'].tolist(), [
                            0, 1, 2, 6, 5, 6, 5, 6, 5])
        self.assertEqual(block.srcdata['ts'].tolist(), [
                            5, 5, 5, 5, 4, 5, 4, 5, 4])
        self.assertEqual(block.edata['dt'].tolist(), [
                            0, 1, 0, 1, 0, 1])
        self.assertEqual(block.edata['ID'].tolist(), [
                            5, 4, 11, 10, 17, 16])

        block = blocks[0][0] # timestamp range: [3, 4]
        self.assertEqual(block.srcdata['ID'].tolist(), [
                            0, 1, 2, 5, 4, 5, 4, 5, 4])
        self.assertEqual(block.srcdata['ts'].tolist(), [
                            4, 4, 4, 4, 3, 4, 3, 4, 3])
        self.assertEqual(block.edata['dt'].tolist(), [
                            0, 1, 0, 1, 0, 1])
        self.assertEqual(block.edata['ID'].tolist(), [
                            4, 3, 10, 9, 16, 15])

        print("Test sample_multi_snapshots passed")


