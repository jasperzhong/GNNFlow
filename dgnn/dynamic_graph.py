from typing import Optional, Tuple

import numpy as np

from libdgnn import MemoryResourceType, InsertionPolicy, _DynamicGraph


class DynamicGraph:
    """
    A dynamic graph is a graph that can be updated at runtime.

    The dynamic graph is implemented as block adjacency list. It has a vertex
    table where each entry is a linked list of blocks. Each block contains
    a list of edges. Each edge is a tuple of (target_vertex, timestamp).
    """

    def __init__(
            self, initial_pool_size: int,
            maximum_pool_size: int,
            mem_resource_type: str,
            initial_pool_size_for_metadata: int,
            maximum_pool_size_for_metadata: int,
            minimum_block_size: int,
            insertion_policy: str,
            source_vertices: Optional[np.ndarray] = None,
            target_vertices: Optional[np.ndarray] = None,
            timestamps: Optional[np.ndarray] = None,
            add_reverse: bool = False,
    ):
        """
        The graph is initially empty and can be optionaly initialized with
        a list of edges.

        Args:
            initial_pool_size: optional, int, the initial pool size of the graph.
            maximum_pool_size: optional, int, the maximum pool size of the graph.
            mem_resource_type: optional, str, the memory resource type. 
                valid options: ("cuda", "unified", or "pinned") (case insensitive).
            initial_pool_size_for_metadata: optional, int, the initial pool size of the metadata.
            maximum_pool_size_for_metadata: optional, int, the maximum pool size of the metadata.
            minimum_block_size: optional, int, the minimum block size of the graph.
            insertion_policy: the insertion policy to use 
                valid options: ("insert" or "replace") (case insensitive).
            source_vertices: optional, 1D tensor, the source vertices of the edges.
            target_vertices: optional, 1D tensor, the target vertices of the edges.
            timestamps: optional, 1D tensor, the timestamps of the edges.
            add_reverse: optional, bool, whether to add reverse edges.
        """
        mem_resource_type = mem_resource_type.lower()
        if mem_resource_type == "cuda":
            mem_resource_type = MemoryResourceType.CUDA
        elif mem_resource_type == "unified":
            mem_resource_type = MemoryResourceType.UNIFIED
        elif mem_resource_type == "pinned":
            mem_resource_type = MemoryResourceType.PINNED
        else:
            raise ValueError("Invalid memory resource type: {}".format(
                mem_resource_type))

        insertion_policy = insertion_policy.lower()
        if insertion_policy == "insert":
            insertion_policy = InsertionPolicy.INSERT
        elif insertion_policy == "replace":
            insertion_policy = InsertionPolicy.REPLACE
        else:
            raise ValueError("Invalid insertion policy: {}".format(
                insertion_policy))

        self._dgraph = _DynamicGraph(
            initial_pool_size, maximum_pool_size, mem_resource_type,
            initial_pool_size_for_metadata, maximum_pool_size_for_metadata,
            minimum_block_size, insertion_policy)

        # initialize the graph with edges
        if source_vertices is not None and target_vertices is not None \
                and timestamps is not None:
            self.add_edges(source_vertices, target_vertices,
                           timestamps, add_reverse)

    def add_edges(
            self, source_vertices: np.ndarray, target_vertices: np.ndarray,
            timestamps: np.ndarray, add_reverse: bool = False):
        """
        Add edges to the graph. Note that we do not assume that the incoming
        edges are sorted by timestamps. The function will sort the incoming
        edges by timestamps.

        Args:
            source_vertices: 1D tensor, the source vertices of the edges.
            target_vertices: 1D tensor, the target vertices of the edges.
            timestamps: 1D tensor, the timestamps of the edges.
            add_reverse: optional, bool, whether to add reverse edges.

        Raises:
            ValueError: if the timestamps are older than the existing edges in
                        the graph.
        """
        assert source_vertices.shape[0] == target_vertices.shape[0] == \
            timestamps.shape[0], "Number of edges must match"
        assert len(source_vertices.shape) == 1 and len(
            target_vertices.shape) == 1 and len(timestamps.shape) == 1, \
            "Source vertices, target vertices and timestamps must be 1D"

        self._dgraph.add_edges(
            source_vertices, target_vertices, timestamps, add_reverse)

    def num_vertices(self) -> int:
        return self._dgraph.num_vertices()

    def num_edges(self) -> int:
        return self._dgraph.num_edges()

    def out_degree(self, vertex: int) -> int:
        return self._dgraph.out_degree(vertex)

    def get_temporal_neighbors(self, vertex: int) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the neighbors of the specified vertex. The neighbors are sorted
        by timestamps in decending order.

        Note that this function is inefficient and should be used sparingly.

        Args:
            vertex: the vertex to get neighbors for.

        Returns: A tuple of (target_vertices, timestamps, edge_ids)
        """
        return self._dgraph.get_temporal_neighbors(vertex)
