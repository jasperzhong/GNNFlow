from typing import Optional, Tuple

import numpy as np

from libdgnn import InsertionPolicy, _DynamicGraph

import time


class DynamicGraph:
    """
    A dynamic graph is a graph that can be modified at runtime.

    The dynamic graph is implemented as block adjacency list. It has a vertex
    table where each entry is a linked list of blocks. Each block contains
    a list of edges. Each edge is a tuple of (target_vertex, timestamp).
    The dynamicg graph is located on the GPU and would offload old edges 
    to the CPU when the GPU consumption exceeds the threshold.
    """

    def __init__(
            self, source_vertices: Optional[np.ndarray] = None,
            target_vertices: Optional[np.ndarray] = None,
            timestamps: Optional[np.ndarray] = None,
            add_reverse: bool = False,
            max_gpu_pool_size: int = 1 << 30, min_block_size: int = 64,
            insertion_policy: str = "insert"):
        """
        The graph is initially empty and can be optionaly initialized with
        a list of edges.

        Args:
            source_vertices: optional, 1D tensor, the source vertices of the edges.
            target_vertices: optional, 1D tensor, the target vertices of the edges.
            timestamps: optional, 1D tensor, the timestamps of the edges.
            add_reverse: optional, bool, whether to add reverse edges.
            max_gpu_pool_size: threshold for GPU memory in bytes (default: 1GB).
            min_block_size: size of the blocks.
            insertion_policy: the insertion policy to use ("insert" or "replace")
                              (case insensitive).
        """
        insertion_policy = insertion_policy.lower()
        if insertion_policy not in ['insert', 'replace']:
            raise ValueError(
                'Insertion policy must be either insert or replace')

        insertion_policy = InsertionPolicy.INSERT if insertion_policy == 'insert' \
            else InsertionPolicy.REPLACE

        start = time.time()
        self._dgraph = _DynamicGraph(
            max_gpu_pool_size, min_block_size, insertion_policy)
        end = time.time()
        print("init graph time: {}".format(end - start))

        start = time.time()
        # initialize the graph with edges
        if source_vertices is not None and target_vertices is not None and timestamps is not None:
            self.add_edges(source_vertices, target_vertices,
                           timestamps, add_reverse)
        end = time.time()
        print("add time: {}".format(end - start))

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
