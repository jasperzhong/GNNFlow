from typing import Union

import torch

from .caching_allocator import CachingAllocator
from .temporal_block import TemporalBlock


class DynamicGraph:
    """
    A dynamic graph is a graph that can be modified at runtime.

    The dynamic graph is implemented as block adjacency list. It has a vertex 
    table where each entry is a linked list of blocks. Each block contains 
    a list of edges. Each edge is a tuple of (target_vertex, timestamp).
    """

    def __init__(self, source_vertices: torch.Tensor, target_vertices: torch.Tensor,
                 timestamps: torch.Tensor, device: Union[torch.device, str],
                 block_size: int, gpu_mem_threshold_in_bytes: int,
                 insertion_policy: str = "reallocate"):
        """
        The graph is initially empty and can be optionaly initialized with 
        a list of edges. 

        Arguments:
            source_vertices: 1D tensor or None, the source vertices of the edges.
            target_vertices: 1D tensor or None, the target vertices of the edges.
            timestamps: 1D tensor or None, the timestamps of the edges. 
            device: the device to use.
            block_size: size of the blocks.
            gpu_mem_threshold_in_bytes: threshold for GPU memory.
            insertion_policy: the insertion policy to use ("reallocate" or "new"). 
                              Case insensitive.
        """
        # lazy initialization
        self._vertex_table = []
        self._num_vertex = 0
        self._allocator = CachingAllocator(
            device, block_size, gpu_mem_threshold_in_bytes)
        self._device = device
        insertion_policy = insertion_policy.lower()
        assert insertion_policy in ["reallocate", "new"], \
            "Invalid insertion policy: {}".format(insertion_policy)
        self._insertion_policy = insertion_policy

        # initialize the graph
        if source_vertices is not None and target_vertices is not None and timestamps is not None:
            self.add_edges(source_vertices, target_vertices, timestamps)

    def add_edges_for_one_vertex(self, source_vertex: int,
                                 target_vertices: torch.Tensor,
                                 timestamps: torch.Tensor):
        """
        Add edges for a specified vertex. Assume that the vertex has already 
        been added and target_vertices have been sorted in ascending order
        of timestamps.

        Arguments:
            source_vertex: the vertex to add edges for.
            target_vertices: the target vertices of the edges.
            timestamps: the timestamps of the edges.
        """
        assert target_vertices.shape[0] == timestamps.shape[0], "Number of edges must match"
        assert len(target_vertices.shape) == 1 and len(
            timestamps.shape) == 1, "Target vertices and timestamps must be 1D"

        if source_vertex < 0 or source_vertex >= self._num_vertex:
            raise ValueError(
                "source_vertex must be between 0 and num_vertex - 1")

        incoming_size = target_vertices.size(0)
        if self._vertex_table[source_vertex] is None:
            # lazy initialization
            block = self._allocator.allocate_on_gpu(incoming_size)
            self._vertex_table[source_vertex] = block

        curr_block = self._vertex_table[source_vertex]
        if curr_block.size + incoming_size > curr_block.capacity:
            # if current block cannot hold the incoming edges, we need to
            # create a new block or reallocate the current block based on
            # the insertion policy.
            if self._insertion_policy == "reallocate":
                # reallocate
                block = self._allocator.reallocate_on_gpu(
                    curr_block, incoming_size)
                self._vertex_table[source_vertex] = block
            elif self._insertion_policy == "new":
                # create a new block
                block = self._allocator.allocate_on_gpu(incoming_size)
                block.next_block = curr_block
                self._vertex_table[source_vertex] = block
                curr_block = block

        # add edges to the current block
        curr_block.add_edges(target_vertices, timestamps)

    def add_edges(self, source_vertices: torch.Tensor, target_vertices: torch.Tensor,
                  timestamps: torch.Tensor):
        """
        Add edges to the graph. 

        Arguments:
            source_vertices: 1D tensor, the source vertices of the edges.
            target_vertices: 1D tensor, the target vertices of the edges.
            timestamps: 1D tensor, the timestamps of the edges.

        The input tensors can be on CPU or GPU.

        Note that we do not assume that the incoming edges are sorted by
        timestamps. The function will sort the incoming edges by timestamps.
        """
        assert source_vertices.shape[0] == target_vertices.shape[0] == \
            timestamps.shape[0], "Number of edges must match"
        assert len(source_vertices.shape) == 1 and len(
            target_vertices.shape) == 1 and len(timestamps.shape) == 1, \
            "Source vertices, target vertices and timestamps must be 1D"

        # group by source vertex
        sorted_idx = torch.argsort(source_vertices)
        unique, counts = torch.unique(
            source_vertices, sorted=True, return_counts=True)
        split_idx = torch.split(sorted_idx, tuple(counts.tolist()))
        for i, indices in enumerate(split_idx):
            source_vertex = unique[i]
            target_vertices_i = target_vertices[indices]
            timestamps_i = timestamps[indices]
            # sorted target_vertices_i by timestamps_i
            sorted_idx = torch.argsort(timestamps_i)
            target_vertices_i = target_vertices_i[sorted_idx]
            timestamps_i = timestamps_i[sorted_idx]

            if source_vertex >= self._num_vertex:
                # lazy initialization
                self.add_vertices(source_vertex)

            self.add_edges_for_one_vertex(source_vertex, target_vertices_i,
                                          timestamps_i)

    def add_vertices(self, max_vertex: int):
        """
        Add vertices to the graph.
        """
        assert max_vertex >= self._num_vertex, "max_vertex must be greater than or equal to num_vertex"

        diff = max_vertex - self._num_vertex
        self._vertex_table.extend([None for _ in range(diff)])
        self._num_vertex = max_vertex
