import itertools
import unittest

import torch
from parameterized import parameterized

from dgnn.temporal_block import TemporalBlock


def is_sorted(tensor: torch.Tensor):
    return torch.all(torch.ge(tensor[1:], tensor[:-1]))


class TestTemporalBlock(unittest.TestCase):
    @parameterized.expand(itertools.product([1024, 2048, 4096], [torch.device("cuda:0"), torch.device("cpu")], [100, 200, 300]))
    def test_add_edges_from_cpu_tensor(self, capacity, device, num_edge_every_insertion, num_insertions=3):
        tblock = TemporalBlock(capacity, device)

        num_vertex = 10000
        edges = torch.randint(
            0, num_vertex, (num_insertions, num_edge_every_insertion))
        timestamps = torch.rand(num_insertions, num_edge_every_insertion)
        timestamps, _ = torch.sort(timestamps.reshape(-1))
        timestamps = timestamps.reshape(
            num_insertions, num_edge_every_insertion)

        for i in range(num_insertions):
            tblock.add_edges(edges[i], timestamps[i])

        self.assertTrue(is_sorted(tblock.timestamps),
                        "timestamps are not sorted")

        print("Add edges from cpu tensor test passed with capacity {}, "\
              "device {}, num_edge_every_insertion {}".format(
              capacity, device, num_edge_every_insertion))

    @parameterized.expand(itertools.product([1024, 2048, 4096], [torch.device("cuda:0"), torch.device("cpu")], [100, 200, 300]))
    def test_add_edges_from_gpu_tensor(self, capacity, device, num_edge_every_insertion, num_insertions=3):
        tblock = TemporalBlock(capacity, device)

        num_vertex = 10000
        edges = torch.randint(
            0, num_vertex, (num_insertions, num_edge_every_insertion))
        timestamps = torch.rand(num_insertions, num_edge_every_insertion)
        timestamps, _ = torch.sort(timestamps.reshape(-1))
        timestamps = timestamps.reshape(
            num_insertions, num_edge_every_insertion)

        for i in range(num_insertions):
            tblock.add_edges(edges[i].to(device), timestamps[i].to(device))

        self.assertTrue(is_sorted(tblock.timestamps),
                        "timestamps are not sorted")

        print("Add edges from gpu tensor test passed with capacity {}, "\
              "device {}, num_edge_every_insertion {}".format(
              capacity, device, num_edge_every_insertion))

    @parameterized.expand(itertools.product([torch.int32, torch.int64], [torch.int32, torch.float32, torch.float64]))
    def test_add_edges_in_different_dtypes(self, target_vertex_dtype, timestamp_dtype):
        tblock = TemporalBlock(2, torch.device("cuda:0"))
        tblock.add_edges(torch.tensor([0, 1], dtype=target_vertex_dtype), torch.tensor(
            [0, 1], dtype=timestamp_dtype))
        print("Add edges in different dtypes test passed")

    def test_add_edges_in_wrong_dtype(self):
        """
        The target_vertices should be of type int32 or int64.
        """
        tblock = TemporalBlock(2, torch.device("cuda:0"))
        self.assertRaises(ValueError, tblock.add_edges, torch.tensor(
            [0, 1], dtype=torch.float32), torch.tensor([0, 1], dtype=torch.float32))

        self.assertRaises(ValueError, tblock.add_edges, torch.tensor(
            [0, 1], dtype=torch.int16), torch.tensor([0, 1], dtype=torch.float32))
        print("Add edges in wrong dtype test passed")

    def test_out_of_capacity(self):
        """
        Test if the temporal block can raise an error when the capacity is exceeded.
        """
        tblock = TemporalBlock(2, torch.device("cuda:0"))
        tblock.add_edges(torch.tensor([0, 1]), torch.tensor([0, 1]))
        self.assertRaises(RuntimeError, tblock.add_edges,
                          torch.tensor([2]), torch.tensor([2]))
        print("Out of capacity test passed")
