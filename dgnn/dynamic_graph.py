import torch

from .caching_allocator import CachingAllocator
from .temporal_block import TemporalBlock


class DynamicGraph:
    """
    A dynamic graph is a graph that can be modified at runtime.
    """

    def __init__(self, num_vertex: int, block_size: int, gpu_mem_threshold: int):
        """
        Arguments:
            num_vertex: number of vertices in the graph.
            block_size: size of the blocks.
            gpu_mem_threshold: threshold for GPU memory to store the graph.
        """
        pass

    def add_edges_for_one_vertex(self, source_vertex: int,
                                 target_vertices: torch.Tensor,
                                 timestamps: torch.Tensor):
        """
        Add edges for one vertex.

        Arguments:
            source_vertex: the vertex to add edges for.
            target_vertices: the target vertices of the edges.
            timestamps: the timestamps of the edges.
        """
        raise NotImplementedError()

    def add_edges_for_all_vertices(self, source_vertices: torch.Tensor, target_vertices: torch.Tensor, timestamps: torch.Tensor):
        """
        Add edges for all vertices.

        Arguments:
            source_vertices: the source vertices of the edges.
            target_vertices: the target vertices of the edges.
            timestamps: the timestamps of the edges.
        """
        raise NotImplementedError()
