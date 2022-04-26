import unittest
from parameterized import parameterized
import itertools

import torch

from dgnn.caching_allocator import CachingAllocator, align
from dgnn.temporal_block import capacity_to_bytes


class TestCachingAllocator(unittest.TestCase):
    def setUp(self):
        device = torch.device("cuda:0")
        gpu_mem_threshold_in_bytes = 50 * 1024 * 1024  # 50 MB
        self.block_size = 1024  # 1 KB
        self.alloc = CachingAllocator(
            device, gpu_mem_threshold_in_bytes, self.block_size)
        self.blocks = []

    def test_allocate_on_gpu(self):
        """
        Allocate on GPU when GPU memory is enough.
        """
        gpu_memory_usage_in_bytes = 0
        for _ in range(10):
            num_edges = torch.randint(1, 5000, (1,)).item()
            tblock = self.alloc.allocate_on_gpu(num_edges)
            self.blocks.append(tblock)
            requested_size_in_bytes = capacity_to_bytes(align(num_edges, self.block_size))
            gpu_memory_usage_in_bytes += requested_size_in_bytes

        self.assertEqual(gpu_memory_usage_in_bytes,
                         self.alloc.get_gpu_memory_usage_in_bytes())

        print("GPU memory usage: {}".format(gpu_memory_usage_in_bytes))

    def test_deallocate(self):
        """
        Deallocate when GPU memory is enough.
        """


    def test_swap_to_cpu(self):
        """
        Swap to CPU when GPU memory is too low.
        """
        pass




