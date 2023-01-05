from typing import Optional, Tuple

import numpy as np

from libgnnflow import InsertionPolicy, MemoryResourceType, _DynamicGraph


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
            minimum_block_size: int,
            blocks_to_preallocate: int,
            insertion_policy: str,
            source_vertices: Optional[np.ndarray] = None,
            target_vertices: Optional[np.ndarray] = None,
            timestamps: Optional[np.ndarray] = None,
            eids: Optional[np.ndarray] = None,
            add_reverse: bool = False,
            device: int = 0,
            adaptive_block_size: bool = True):
        """
        The graph is initially empty and can be optionaly initialized with
        a list of edges.

        Args:
            initial_pool_size: optional, int, the initial pool size of the graph.
            maximum_pool_size: optional, int, the maximum pool size of the graph.
            mem_resource_type: optional, str, the memory resource type.
                valid options: ("cuda", "unified", "pinned", or "shared") (case insensitive).
            minimum_block_size: optional, int, the minimum block size of the graph.
            blocks_to_preallocate: optional, int, the number of blocks to preallocate.
            insertion_policy: the insertion policy to use
                valid options: ("insert" or "replace") (case insensitive).
            source_vertices: optional, 1D tensor, the source vertices of the edges.
            target_vertices: optional, 1D tensor, the target vertices of the edges.
            timestamps: optional, 1D tensor, the timestamps of the edges.
            eids: optional, 1D tensor, the edge ids of the edges.
            add_reverse: optional, bool, whether to add reverse edges.
            device: optional, int, the device to use.
            adaptive_block_size: optional, bool, whether to use adaptive block size.
        """
        mem_resource_type = mem_resource_type.lower()
        if mem_resource_type == "cuda":
            mem_resource_type = MemoryResourceType.CUDA
        elif mem_resource_type == "unified":
            mem_resource_type = MemoryResourceType.UNIFIED
        elif mem_resource_type == "pinned":
            mem_resource_type = MemoryResourceType.PINNED
        elif mem_resource_type == "shared":
            mem_resource_type = MemoryResourceType.SHARED
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
            minimum_block_size, blocks_to_preallocate, insertion_policy,
            device, adaptive_block_size)

        # initialize the graph with edges
        if source_vertices is not None and target_vertices is not None \
                and timestamps is not None:
            self.add_edges(source_vertices, target_vertices,
                           timestamps, eids, add_reverse)

    def add_edges(
            self, source_vertices: np.ndarray, target_vertices: np.ndarray,
            timestamps: np.ndarray, eids: Optional[np.ndarray] = None, add_reverse: bool = False):
        """
        Add edges to the graph. Note that we do not assume that the incoming
        edges are sorted by timestamps. The function will sort the incoming
        edges by timestamps.

        Args:
            source_vertices: 1D tensor, the source vertices of the edges.
            target_vertices: 1D tensor, the target vertices of the edges.
            timestamps: 1D tensor, the timestamps of the edges.
            eids: 1D tensor, the edge ids of the edges.
            add_reverse: optional, bool, whether to add reverse edges.

        Raises:
            ValueError: if the timestamps are older than the existing edges in
                        the graph.
        """
        assert len(source_vertices.shape) == 1 and len(
            target_vertices.shape) == 1 and len(timestamps.shape) == 1, "Edges must be 1D tensors"

        assert source_vertices.shape[0] == target_vertices.shape[0] == \
            timestamps.shape[0], "The number of source vertices, target vertices, timestamps, " \
            "and edge ids must be the same."

        if eids is None:
            num_edges = self.num_edges()
            eids = np.arange(num_edges, num_edges + len(source_vertices))

        if add_reverse:
            source_vertices_ext = np.concatenate(
                [source_vertices, target_vertices])
            target_vertices_ext = np.concatenate(
                [target_vertices, source_vertices])
            source_vertices = source_vertices_ext
            target_vertices = target_vertices_ext
            timestamps = np.concatenate([timestamps, timestamps])
            eids = np.concatenate([eids, eids])

        self._dgraph.add_edges(
            source_vertices, target_vertices, timestamps, eids)

    def offload_old_blocks(self, timestamp: float, to_file: bool = False):
        """
        Offload the old blocks from the graph.

        Args:
            timestamp: the current timestamp.
            to_file: whether to offload the blocks to file.

        Return:
            the number of blocks offloaded.
        """
        return self._dgraph.offload_old_blocks(timestamp, to_file)

    def num_vertices(self) -> int:
        return self._dgraph.num_vertices()

    def num_source_vertices(self) -> int:
        return self._dgraph.num_source_vertices()

    def max_vertex_id(self) -> int:
        return self._dgraph.max_vertex_id()

    def num_edges(self) -> int:
        return self._dgraph.num_edges()

    def out_degree(self, vertexs: np.ndarray) -> np.ndarray:
        return self._dgraph.out_degree(vertexs)

    def nodes(self) -> np.ndarray:
        """
        Return the nodes of the graph.
        """
        return self._dgraph.nodes()

    def src_nodes(self) -> np.ndarray:
        """
        Return the source nodes of the graph.
        """
        return self._dgraph.src_nodes()

    def edges(self) -> np.ndarray:
        """
        Return the edges of the graph.
        """
        return self._dgraph.edges()

    def get_temporal_neighbors(self, vertex: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the neighbors of the specified vertex. The neighbors are sorted
        by timestamps in decending order.

        Note that this function is inefficient and should be used sparingly.

        Args:
            vertex: the vertex to get neighbors for.

        Returns: A tuple of (target_vertices, timestamps, edge_ids)
        """
        return self._dgraph.get_temporal_neighbors(vertex)

    def avg_linked_list_length(self) -> float:
        """
        Return the average linked list length.
        """
        return self._dgraph.avg_linked_list_length()

    def get_graph_memory_usage(self) -> int:
        """
        Return the graph memory usage of the graph in bytes.
        """
        return self._dgraph.get_graph_memory_usage()

    def get_metadata_memory_usage(self) -> int:
        """
        Return the metadata memory usage of the graph in bytes.
        """
        return self._dgraph.get_metadata_memory_usage()
