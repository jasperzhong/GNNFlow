import logging
import threading
import time
from queue import Queue

import numpy as np
import torch

from gnnflow import DynamicGraph
from gnnflow.distributed.utils import HandleManager


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
        self._max_vertex_id = 0

        self._handle_manager = HandleManager()
        self._handles = set()
        self._add_edges_thread = threading.Thread(target=self._add_edges_loop)
        self._add_edges_queue = Queue()
        self._add_edges_thread.start()

    def shutdown(self):
        logging.info('DistributedDynamicGraph shutdown')
        self._add_edges_queue.put((None, None, None, None, None))
        self._add_edges_thread.join()

    def _add_edges_loop(self):
        while True:
            while not self._add_edges_queue.empty():
                source_vertices, target_vertices, timestamps, eids, handle = self._add_edges_queue.get()
                if handle is None:
                    return

                self._dgraph.add_edges(
                    source_vertices, target_vertices, timestamps, eids)

                self._handle_manager.mark_done(handle)

            time.sleep(0.001)

    def enqueue_add_edges_task(self, source_vertices: np.ndarray, target_vertices: np.ndarray,
                               timestamps: np.ndarray, eids: np.ndarray):
        handle = self._handle_manager.allocate_handle()
        self._add_edges_queue.put(
            (source_vertices, target_vertices, timestamps, eids, handle))
        self._handles.add(handle)

    def poll(self, handle):
        return self._handle_manager.poll(handle)

    def wait_for_all_updates_to_finish(self):
        for handle in self._handles.copy():
            if self.poll(handle):
                self._handles.remove(handle)

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

    def num_source_vertices(self) -> int:
        return self._dgraph.num_source_vertices()

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
        return self._max_vertex_id

    def set_max_vertex_id(self, max_vertex_id: int):
        self._max_vertex_id = max_vertex_id

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
