from .temporal_block import TemporalBlock


class CachingAllocator:
    """
    This class implements a caching allocator.
    """

    def __init__(self, gpu_mem_threshold: int, block_size: int):
        pass

    def allocate_on_gpu(self, size: int) -> TemporalBlock:
        """
        Allocates a temporal block on the GPU.
        """
        raise NotImplementedError()

    def deallocate(self, temporal_block: TemporalBlock) -> None:
        """
        Deallocates a temporal block.
        """
        raise NotImplementedError()

    def reallocate_on_gpu(self, temporal_block: TemporalBlock, size: int) -> TemporalBlock:
        """
        Reallocates a temporal block on the GPU.
        """
        raise NotImplementedError()

    def swap_to_cpu(self, swap_minimum_size: int) -> None:
        """
        Swaps old and unused temporal blocks on the GPU to the CPU if the 
        swap_minimum_size is reached. If failed to swap enough temporal blocks,
        raise an exception.
        """
        raise NotImplementedError()
