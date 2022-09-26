import numpy as np

from dgnn import DynamicGraph


class DistributedDynamicGraph:
    """
    Distributed dynamic graph.
    """

    def __init__(self, dgraph: DynamicGraph):
        """
        Initialize the distributed dynamic graph.

        Args:
            dgraph (DynamicGraph): The dynamic graph instance.
        """
        self._dgraph = dgraph
        self._partition_table = None
        self._num_vertices = 0
        self._num_edges = 0

    def num_vertices(self) -> int:
        """
        Get the number of vertices in the dynamic graph.
        Returns:
            int: The number of vertices.
        """
        return self._num_vertices

    def num_edges(self) -> int:
        """
        Get the number of edges in the dynamic graph.
        Returns:
            int: The number of edges.
        """
        return self._num_edges

    def set_num_vertices(self, num_vertices: int):
        """
        Set the number of vertices in the dynamic graph.
        Args:
            num_vertices (int): The number of vertices.
        """
        self._num_vertices = num_vertices

    def set_num_edges(self, num_edges: int):
        """
        Set the number of edges in the dynamic graph.
        Args:
            num_edges (int): The number of edges.
        """
        self._num_edges = num_edges

    def add_edges(self, source_vertices: np.ndarray, target_vertices: np.ndarray,
                  timestamps: np.ndarray, eids: np.ndarray):
        return self._dgraph.add_edges(source_vertices, target_vertices, timestamps, eids)

    def out_degree(self, vertex: int):
        return self._dgraph.out_degree(vertex)
