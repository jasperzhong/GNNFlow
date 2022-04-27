import unittest
from parameterized import parameterized
import itertools

import torch

from dgnn.caching_allocator import CachingAllocator, align
from dgnn.temporal_block import capacity_to_bytes


class TestCachingAllocator(unittest.TestCase):
    def setUp(self, gpu_mem_threshold_in_bytes=1 * 1024 * 1024): # 1 MB
        device = torch.device("cuda:0")
        self.block_size =  1024  # 1 KB
        self.alloc = CachingAllocator(
            device, gpu_mem_threshold_in_bytes, self.block_size)
        self.blocks = []

    def test_allocate_on_gpu(self, gpu_mem_threshold_in_bytes=128 * 1024):
        """
        Allocate on GPU when GPU memory is enough.
        1. has free_gpu_blocks
        2. don't have free, swap first if exceed the threshold. test swap is ok
        3. 
        
        Test logic:
        1. free blocks on gpu and cpu is empty, just allocate new mem
        2. allocate too much and exceed the threshold, swap to cpu
        """
        self.setUp(gpu_mem_threshold_in_bytes)
        # allocate some blocks on gpu.
        gpu_memory_usage_in_bytes = 0
        for _ in range(10):
            num_edges = torch.randint(3000, 5000, (1,)).item()
            tblock = self.alloc.allocate_on_gpu(num_edges)
            self.blocks.append(tblock)
            requested_size_in_bytes = capacity_to_bytes(align(num_edges, self.block_size))
            gpu_memory_usage_in_bytes += requested_size_in_bytes
        
        # Test Deallocate
        blocks = self.alloc._used_gpu_blocks.keys()
        for block in list(blocks):
            self.alloc.deallocate(block)
        
        blocks = self.alloc._used_cpu_blocks.keys()
        for block in list(blocks):
            self.alloc.deallocate(block)
        
        self.assertEqual(len(self.alloc._used_gpu_blocks), 0)
        self.assertEqual(len(self.alloc._used_cpu_blocks), 0)
        
        # Test Swap to CPU
        sum_capacity = 0
        for blocks in self.alloc._free_gpu_blocks.values():
            for block in blocks:
                sum_capacity += block.capacity
        # swap all the blocks in free gpu
        self.alloc.swap_to_cpu(capacity_to_bytes(sum_capacity))
        
        num = 0
        for blocks in self.alloc._free_gpu_blocks.values():
            for block in blocks:
                num += 1
        
        self.assertEqual(num, 0)
        
        # Test allocate on gpu again
        # all free blocks are on cpu
        # it will use free cpu blocks.
        gpu_memory_usage_in_bytes = 0
        for _ in range(10):
            num_edges = torch.randint(3000, 5000, (1,)).item()
            tblock = self.alloc.allocate_on_gpu(num_edges)
            self.blocks.append(tblock)
            requested_size_in_bytes = capacity_to_bytes(align(num_edges, self.block_size))
            gpu_memory_usage_in_bytes += requested_size_in_bytes

        print("***allocate end****")
        print("GPU USED:{}".format(self.alloc._used_gpu_blocks))
        print("CPU USED:{}".format(self.alloc._used_cpu_blocks))
        print("GPU FREE:{}".format(self.alloc._free_gpu_blocks))
        print("CPU FREE:{}".format(self.alloc._free_cpu_blocks))
        




