from typing import Optional, Union, Tuple, List

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

    def __init__(self, source_vertices: Optional[torch.Tensor] = None,
                 target_vertices: Optional[torch.Tensor] = None,
                 timestamps: Optional[torch.Tensor] = None,
                 device: Union[torch.device, str] = "cuda",
                 gpu_mem_threshold_in_bytes: int = 50 * 1024 * 1024,
                 block_size: int = 1024,
                 insertion_policy: str = "reallocate"):
        """
        The graph is initially empty and can be optionaly initialized with 
        a list of edges. 

        Arguments:
            source_vertices: optional, 1D tensor, the source vertices of the edges.
            target_vertices: optional, 1D tensor, the target vertices of the edges.
            timestamps: optional, 1D tensor, the timestamps of the edges. 
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
            device, gpu_mem_threshold_in_bytes, block_size)
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
        Add edges for a specified vertex. Assume that target_vertices have been 
        sorted in ascending order of timestamps and that the timestamps are
        newer than the existing edges.

        Arguments:
            source_vertex: the vertex to add edges for.
            target_vertices: the target vertices of the edges.
            timestamps: the timestamps of the edges.

        Note that duplicate edges are allowed.
        """
        assert target_vertices.shape[0] == timestamps.shape[0], "Number of edges must match"
        assert len(target_vertices.shape) == 1 and len(
            timestamps.shape) == 1, "Target vertices and timestamps must be 1D"

        if source_vertex < 0:
            raise ValueError("source_vertex must be non-negative")

        if source_vertex >= self._num_vertex:
            # lazy initialization
            self.add_vertices(source_vertex)

        incoming_size = target_vertices.size(0)
        if self._vertex_table[source_vertex] is None:
            # lazy initialization
            block = self._allocator.allocate_on_gpu(incoming_size)
            self._vertex_table[source_vertex] = block

        curr_block = self._vertex_table[source_vertex]
        if curr_block.size + incoming_size > curr_block.capacity:
            # if current block cannot hold the incoming edges, we need to
            # reallocate the current block or create a new block based on
            # the insertion policy.
            if self._insertion_policy == "reallocate":
                # reallocate
                block = self._allocator.reallocate_on_gpu(
                    curr_block, curr_block.size + incoming_size)
                self._vertex_table[source_vertex] = block
                print("block capacity: {}".format(block.capacity))
            elif self._insertion_policy == "new":
                # create a new block
                block = self._allocator.allocate_on_gpu(incoming_size)
                block.next_block = curr_block
                self._vertex_table[source_vertex] = block
            else:
                raise ValueError("Invalid insertion policy: {}".format(
                    self._insertion_policy))

            curr_block = block

        # add edges to the current block
        max_vertex = int(target_vertices.max().item())
        if max_vertex >= self._num_vertex:
            # lazy initialization
            self.add_vertices(max_vertex)

        # check timestamps are newer than the existing edges
        if curr_block.size > 0:
            if timestamps[0] <= curr_block.timestamps[curr_block.size - 1]:
                raise ValueError(
                    "Timestamps must be newer than the existing edges")

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

            self.add_edges_for_one_vertex(source_vertex, target_vertices_i,
                                          timestamps_i)

    def add_vertices(self, max_vertex: int):
        """
        Add vertices to the graph.

        Arguments:
            max_vertex: the maximum vertex id to add.
        """
        assert max_vertex >= self._num_vertex, "max_vertex must be greater than or equal to num_vertex"

        diff = max_vertex - self._num_vertex + 1
        self._vertex_table.extend([None for _ in range(diff)])
        self._num_vertex = max_vertex + 1

    def num_vertices(self):
        """
        Return the number of vertices in the graph.
        """
        return self._num_vertex

    def num_edges(self):
        """
        Return the number of edges in the graph.
        """
        num_edges = 0
        for i in range(self._num_vertex):
            curr_block = self._vertex_table[i]
            while curr_block is not None:
                num_edges += curr_block.size
                curr_block = curr_block.next_block

        return num_edges

    def out_degree(self, vertex: int):
        """
        Return the out degree of the specified vertex.
        """
        assert vertex >= 0 and vertex < self._num_vertex, "vertex must be in range"

        out_degree = 0
        curr_block = self._vertex_table[vertex]
        while curr_block is not None:
            out_degree += curr_block.size
            curr_block = curr_block.next_block

        return out_degree

    def out_edges(self, vertex: int) -> Tuple[List, List]:
        """
        Return the out edges of the specified vertex.
        """
        assert vertex >= 0 and vertex < self._num_vertex, "vertex must be in range"

        target_vertices = []
        timestamps = []
        curr_block = self._vertex_table[vertex]
        while curr_block is not None:
            if curr_block.size > 0:
                target_vertices.extend(curr_block.target_vertices.tolist())
                timestamps.extend(curr_block.timestamps.tolist())

            curr_block = curr_block.next_block

        return target_vertices, timestamps
