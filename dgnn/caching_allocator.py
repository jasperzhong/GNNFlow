import math
from collections import defaultdict
from typing import Union

import torch

from .temporal_block import TemporalBlock, capacity_to_bytes


def align(size: int, block_size: int):
    """
    Align a size of the power of 2 of block_size
    """
    return (2 ** math.ceil(max(math.log2(size / block_size), 0))) * block_size


class CachingAllocator:
    """
    This class implements a caching allocator for a specified GPU device.

    The allocator allocates temporal blocks on the GPU. When the GPU memory
    usage reaches a threshold, the allocator swaps old and unused temporal
    blocks on the GPU to the CPU. The allocator keeps track of the temporal
    blocks that are currently in use. 

    Note that each GPU device has its own allocator.
    """

    def __init__(self, device: Union[torch.device, str],
                 gpu_mem_threshold_in_bytes: int, block_size: int):
        """
        Arguments:
            device: The GPU device.
            gpu_mem_threshold_in_bytes: The threshold of GPU memory usage.
            block_size: The size of temporal blocks.
        """
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError('device must be a GPU device')

        self._device = device
        self._gpu_mem_threshold_in_bytes = gpu_mem_threshold_in_bytes
        self._block_size = block_size
        self._free_gpu_blocks = defaultdict(list)
        self._used_gpu_blocks = defaultdict(int)
        self._free_cpu_blocks = defaultdict(list)
        self._used_cpu_blocks = defaultdict(int)
        self._gpu_mem_usage_in_bytes = 0

        # Sequence number is the logical time that is assigned to each 
        # temporal block. It increases by 1 every time a temporal block is 
        # allocated. The smaller the value, the older the block.
        self._sequence_number = 0

    def allocate_on_gpu(self, size: int) -> TemporalBlock:
        """
        Allocates a temporal block on the GPU. If failed to allocate a temporal
        block, raise an exception.

        Arguments:
            size: The size of the temporal block.
        """
        capacity = align(size, self._block_size)
        requested_size_in_bytes = capacity_to_bytes(capacity)
        block = None
        if len(self._free_gpu_blocks[capacity]) > 0:
            block = self._free_gpu_blocks[capacity].pop()

        if block is None and requested_size_in_bytes + self._gpu_mem_usage_in_bytes > self._gpu_mem_threshold_in_bytes:
            self.swap_to_cpu(requested_size_in_bytes + self._gpu_mem_usage_in_bytes 
                             - self._gpu_mem_threshold_in_bytes)

        if len(self._free_cpu_blocks[capacity]) > 0:
            block = self._free_cpu_blocks[capacity].pop()
            block.to(self._device)
            self._gpu_mem_usage_in_bytes += requested_size_in_bytes

        if block is None:
            # Allocate a new temporal block on the GPU.
            block = TemporalBlock(capacity, self._device)
            self._gpu_mem_usage_in_bytes += requested_size_in_bytes

        self._used_gpu_blocks[block] = self._sequence_number
        self._sequence_number += 1
        return block

    def deallocate(self, temporal_block: TemporalBlock) -> None:
        """
        Deallocates a temporal block. The deallocated temporal block is
        marked as free for later reuse.

        Arguments:
            temporal_block: The temporal block to be deallocated.
        """
        if temporal_block in self._used_gpu_blocks:
            self._used_gpu_blocks.pop(temporal_block)
            self._free_gpu_blocks[temporal_block.capacity].append(
                temporal_block)

        if temporal_block in self._used_cpu_blocks:
            self._used_cpu_blocks.pop(temporal_block)
            self._free_cpu_blocks[temporal_block.capacity].append(
                temporal_block)

    def reallocate_on_gpu(self, temporal_block: TemporalBlock, size: int) -> TemporalBlock:
        """
        Reallocates a temporal block on the GPU.

        Arguments:
            temporal_block: The temporal block to be reallocated.
            size: The size of the temporal block.
        """
        new_block = self.allocate_on_gpu(size)
        temporal_block.copy_to(new_block)
        self.deallocate(temporal_block)
        return new_block

    def swap_to_cpu(self, minimum_swap_size_in_bytes: int) -> None:
        """
        Swaps old and unused temporal blocks on the GPU to the CPU if the 
        minimum_swap_size_in_bytes is  reached. If failed to swap enough 
        temporal blocks,  raise an exception.
        """
        # TODO:may also need sort
        for capacity, blocks in self._free_gpu_blocks.items():
            while minimum_swap_size_in_bytes > 0 and len(blocks) > 0:
                block = blocks.pop()
                block.to('cpu')
                self._free_cpu_blocks[capacity].append(block)
                block_size_in_bytes = capacity_to_bytes(capacity)
                self._gpu_mem_usage_in_bytes -= block_size_in_bytes
                minimum_swap_size_in_bytes -= block_size_in_bytes

            if minimum_swap_size_in_bytes <= 0:
                break

        # free blocks are not enough to swap
        # start to swap old blocks in use
        if minimum_swap_size_in_bytes > 0:
            # sort the blocks in use by sequence number
            # the older the block, the smaller the sequence number
            used_gpu_blocks_sorted_by_sequence_number = {k: v for k, v in sorted(
                self._used_gpu_blocks.items(), key=lambda item: item[1])}

            for block, sequence_number in used_gpu_blocks_sorted_by_sequence_number.items():
                if minimum_swap_size_in_bytes > 0:
                    block.to('cpu')
                    self._used_cpu_blocks[block] = sequence_number
                    self._used_gpu_blocks.pop(block)
                    block_size_in_bytes = capacity_to_bytes(block.capacity)
                    self._gpu_mem_usage_in_bytes -= block_size_in_bytes
                    minimum_swap_size_in_bytes -= block_size_in_bytes

                if minimum_swap_size_in_bytes <= 0:
                    break

        if minimum_swap_size_in_bytes > 0:
            raise RuntimeError(
                'Failed to swap enough temporal blocks to the CPU')

    def get_gpu_memory_usage_in_bytes(self) -> int:
        """
        Returns the GPU memory usage in bytes.
        """
        return self._gpu_mem_usage_in_bytes

    @property
    def device(self) -> torch.device:
        return self._device
    
