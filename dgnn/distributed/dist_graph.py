import numpy as np
import torch

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
        self._num_partitions = None
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

    def max_vertex_id(self) -> int:
        return self._dgraph.max_vertex_id()

    def set_partition_table(self, partition_table: torch.Tensor):
        """
        Set the partition table.

        Args:
            partition_table (torch.Tensor): The partition table.
        """
        self._partition_table = partition_table

    def get_partition_table(self) -> torch.Tensor:
        """
        Get the partition table.

        Returns:
            torch.Tensor: The partition table.
        """
        if self._partition_table is None:
            raise RuntimeError('Partition table is not set.')
        return self._partition_table

    def set_num_partitions(self, num_partitions: int):
        """
        Set the number of partitions.

        Args:
            num_partitions (int): The number of partitions.
        """
        self._num_partitions = num_partitions

    def num_partitions(self) -> int:
        """
        Get the number of partitions.

        Returns:
            int: The number of partitions.
        """
        if self._num_partitions is None:
            raise RuntimeError('Number of partitions is not set.')
        return self._num_partitions

    def add_edges(self, source_vertices: np.ndarray, target_vertices: np.ndarray,
                  timestamps: np.ndarray, eids: np.ndarray):
        return self._dgraph.add_edges(source_vertices, target_vertices, timestamps, eids)

    def out_degree(self, vertices: np.ndarray):
        return self._dgraph.out_degree(vertices)
