from typing import Optional, Tuple, Union

import torch

from .caching_allocator import CachingAllocator


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

        Args:
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
        self._num_vertices = 0
        self._num_edges = 0
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

    def add_edges(self, source_vertices: torch.Tensor, target_vertices: torch.Tensor,
                  timestamps: torch.Tensor):
        """
        Add edges to the graph. The input tensors can be on CPU or GPU.

        Note that we do not assume that the incoming edges are sorted by
        timestamps. The function will sort the incoming edges by timestamps.

        Args:
            source_vertices: 1D tensor, the source vertices of the edges.
            target_vertices: 1D tensor, the target vertices of the edges.
            timestamps: 1D tensor, the timestamps of the edges.

        Raises:
            ValueError: if the timestamps are not in ascending order.
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

            self._add_edges_for_one_vertex(source_vertex, target_vertices_i,
                                           timestamps_i)

    def add_vertices(self, max_vertex: int):
        """
        Add vertices to the graph.

        Args:
            max_vertex: the maximum vertex id to add.
        """
        assert max_vertex >= self._num_vertices, "max_vertex must be greater " \
            "than or equal to num_vertex"

        diff = max_vertex - self._num_vertices + 1
        self._vertex_table.extend([None for _ in range(diff)])
        self._num_vertices = max_vertex + 1

    def _add_edges_for_one_vertex(self, source_vertex: int,
                                  target_vertices: torch.Tensor,
                                  timestamps: torch.Tensor):
        """
        Add edges for a specified vertex. Assume that target_vertices have been
        sorted in ascending order of timestamps and that the timestamps are
        newer than the existing edges.

        Args:
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

        max_vertex = int(target_vertices.max().item())
        max_vertex = max(max_vertex, source_vertex)
        if max_vertex >= self._num_vertices:
            # lazy initialization
            self.add_vertices(max_vertex)

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
            elif self._insertion_policy == "new":
                # create a new block
                block = self._allocator.allocate_on_gpu(incoming_size)
                block.next_block = curr_block
                self._vertex_table[source_vertex] = block
            else:
                raise ValueError("Invalid insertion policy: {}".format(
                    self._insertion_policy))

            curr_block = block

        # check timestamps are newer than the existing edges
        if not curr_block.empty():
            if timestamps[0] <= curr_block.end_timestamp():
                raise ValueError(
                    "Timestamps must be newer than the existing edges")

        # add the edges to the current block
        edges_ids = torch.arange(
            self._num_edges, self._num_edges + incoming_size)
        curr_block.add_edges(target_vertices, timestamps, edges_ids)
        self._num_edges += incoming_size

    @property
    def num_vertices(self):
        return self._num_vertices

    @property
    def num_edges(self):
        return self._num_edges

    def out_degree(self, vertex: int) -> int:
        """
        Return the out degree of the specified vertex.
        """
        assert vertex >= 0 and vertex < self._num_vertices, "vertex must be in range"

        out_degree = 0
        curr_block = self._vertex_table[vertex]
        while curr_block is not None:
            out_degree += curr_block.size
            curr_block = curr_block.next_block

        return out_degree

    def get_temporal_neighbors(self, vertex: int, start_timestamp: float = float("-inf"),
                               end_timestamp: float = float("inf")) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the neighbors of the specified vertex (vertex ids, timestamps, edge ids)
        in the specified time range [start_timestamp, end_timestamp].The neighbors 
        are sorted by timestamps in decending order.

        Args:
            vertex: the vertex to get neighbors for.
            start_timestamp: the start timestamp. Default to float("-inf").
            end_timestamp: the end timestamp. Default to float("inf").

        Returns: A tuple of (target_vertices, timestamps, edge_ids)
        """
        assert vertex >= 0 and vertex < self._num_vertices, "vertex must be in range"
        assert start_timestamp <= end_timestamp, "start_timestamp must be less" \
            " than or equal to end_timestamp"

        if start_timestamp == float("-inf") and end_timestamp == float("inf"):
            # no need to filter
            return self._get_neighbors(vertex)
        elif start_timestamp == float("-inf") and end_timestamp < float("inf"):
            # filter by end_timestamp
            return self._get_neighbors_before_timestamp(vertex, end_timestamp)
        elif start_timestamp > float("-inf") and end_timestamp == float("inf"):
            # filter by start_timestamp
            return self._get_neighbors_after_timestamp(vertex, start_timestamp)
        else:
            # filter by start_timestamp and end_timestamp
            return self._get_neighbors_between_timestamps(vertex, start_timestamp, end_timestamp)

    def _get_neighbors(self, vertex: int) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        target_vertices = torch.LongTensor()
        timestamps = torch.FloatTensor()
        edge_ids = torch.LongTensor()
        curr_block = self._vertex_table[vertex]
        while curr_block is not None:
            if not curr_block.empty():
                target_vertices = torch.cat(
                    (target_vertices, curr_block.target_vertices.flip(dims=[0]).cpu()), dim=0)
                timestamps = torch.cat(
                    (timestamps, curr_block.timestamps.flip(dims=[0]).cpu()), dim=0)
                edge_ids = torch.cat(
                    (edge_ids, curr_block.edge_ids.flip(dims=[0]).cpu()), dim=0)

            curr_block = curr_block.next_block

        return target_vertices, timestamps, edge_ids

    def _get_neighbors_before_timestamp(self, vertex: int, timestamp: float) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        target_vertices = torch.LongTensor()
        timestamps = torch.FloatTensor()
        edge_ids = torch.LongTensor()
        curr_block = self._vertex_table[vertex]
        while curr_block is not None:
            if not curr_block.empty():
                if timestamp < curr_block.start_timestamp():
                    # this block does not contain any edges before the timestamp
                    pass
                elif timestamp > curr_block.end_timestamp():
                    # this block contains all edges before the timestamp
                    target_vertices = torch.cat(
                        (target_vertices, curr_block.target_vertices.flip(dims=[0]).cpu()), dim=0)
                    timestamps = torch.cat(
                        (timestamps, curr_block.timestamps.flip(dims=[0]).cpu()), dim=0)
                    edge_ids = torch.cat(
                        (edge_ids, curr_block.edge_ids.flip(dims=[0]).cpu()), dim=0)
                else:
                    # find the first edge before the timestamp
                    idx = torch.searchsorted(curr_block.timestamps, timestamp,
                                             right=True)
                    if idx > 0:
                        target_vertices = torch.cat(
                            (target_vertices, curr_block.target_vertices[:idx].flip(dims=[0]).cpu()), dim=0)
                        timestamps = torch.cat(
                            (timestamps, curr_block.timestamps[:idx].flip(dims=[0]).cpu()), dim=0)
                        edge_ids = torch.cat(
                            (edge_ids, curr_block.edge_ids[:idx].flip(dims=[0]).cpu()), dim=0)

                curr_block = curr_block.next_block

        return target_vertices, timestamps, edge_ids

    def _get_neighbors_after_timestamp(self, vertex: int, timestamp: float) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        target_vertices = torch.LongTensor()
        timestamps = torch.FloatTensor()
        edge_ids = torch.LongTensor()
        curr_block = self._vertex_table[vertex]
        while curr_block is not None:
            if not curr_block.empty():
                if timestamp > curr_block.end_timestamp():
                    # this block does not contain any edges after the timestamp
                    # no need to search in the next block
                    break
                elif timestamp < curr_block.start_timestamp():
                    # this block contains all edges after the timestamp
                    target_vertices = torch.cat(
                        (target_vertices, curr_block.target_vertices.flip(dims=[0]).cpu()), dim=0)
                    timestamps = torch.cat(
                        (timestamps, curr_block.timestamps.flip(dims=[0]).cpu()), dim=0)
                    edge_ids = torch.cat(
                        (edge_ids, curr_block.edge_ids.flip(dims=[0]).cpu()), dim=0)
                else:
                    # find the first edge after the timestamp
                    idx = torch.searchsorted(curr_block.timestamps, timestamp,
                                             side='left')
                    if idx < curr_block.size:
                        target_vertices = torch.cat(
                            (target_vertices, curr_block.target_vertices[idx:].flip(dims=[0]).cpu()), dim=0)
                        timestamps = torch.cat(
                            (timestamps, curr_block.timestamps[idx:].flip(dims=[0]).cpu()), dim=0)
                        edge_ids = torch.cat(
                            (edge_ids, curr_block.edge_ids[idx:].flip(dims=[0]).cpu()), dim=0)

                curr_block = curr_block.next_block

        return target_vertices, timestamps, edge_ids

    def _get_neighbors_between_timestamps(self, vertex: int, start_timestamp: float,
                                          end_timestamp: float) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        target_vertices = torch.LongTensor()
        timestamps = torch.FloatTensor()
        edge_ids = torch.LongTensor()
        curr_block = self._vertex_table[vertex]
        while curr_block is not None:
            if not curr_block.empty():
                if end_timestamp < curr_block.start_timestamp():
                    # search in the next block
                    curr_block = curr_block.next_block
                    continue

                if start_timestamp > curr_block.end_timestamp():
                    # no need to search in the next block
                    break

                # search in the current block
                if start_timestamp >= curr_block.start_timestamp() and \
                        end_timestamp <= curr_block.end_timestamp():
                    # all edges are in the current block
                    start_idx = torch.searchsorted(curr_block.timestamps, start_timestamp,
                                                   side='left')
                    end_idx = torch.searchsorted(curr_block.timestamps, end_timestamp,
                                                 side='right')
                    target_vertices = torch.cat(
                        (target_vertices, curr_block.target_vertices[start_idx:end_idx].flip(dims=[0]).cpu()), dim=0)
                    timestamps = torch.cat(
                        (timestamps, curr_block.timestamps[start_idx:end_idx].flip(dims=[0]).cpu()), dim=0)
                    edge_ids = torch.cat(
                        (edge_ids, curr_block.edge_ids[start_idx:end_idx].flip(dims=[0]).cpu()), dim=0)

                    break
                elif start_timestamp < curr_block.start_timestamp() and \
                        end_timestamp <= curr_block.end_timestamp():
                    # only the edges before end_timestamp are in the current block
                    idx = torch.searchsorted(curr_block.timestamps, end_timestamp,
                                             side='right')
                    target_vertices = torch.cat(
                        (target_vertices, curr_block.target_vertices[:idx].flip(dims=[0]).cpu()), dim=0)
                    timestamps = torch.cat(
                        (timestamps, curr_block.timestamps[:idx].flip(dims=[0]).cpu()), dim=0)
                    edge_ids = torch.cat(
                        (edge_ids, curr_block.edge_ids[:idx].flip(dims=[0]).cpu()), dim=0)

                    curr_block = curr_block.next_block
                    continue
                elif start_timestamp >= curr_block.start_timestamp() and \
                        end_timestamp > curr_block.end_timestamp():
                    # only the edges after start_timestamp are in the current block
                    idx = torch.searchsorted(curr_block.timestamps, start_timestamp,
                                             side='left')
                    target_vertices = torch.cat(
                        (target_vertices, curr_block.target_vertices[idx:].flip(dims=[0]).cpu()), dim=0)
                    timestamps = torch.cat(
                        (timestamps, curr_block.timestamps[idx:].flip(dims=[0]).cpu()), dim=0)
                    edge_ids = torch.cat(
                        (edge_ids, curr_block.edge_ids[idx:].flip(dims=[0]).cpu()), dim=0)

                    break
                else:
                    # the whole block is in the range
                    target_vertices = torch.cat(
                        (target_vertices, curr_block.target_vertices.flip(dims=[0]).cpu()), dim=0)
                    timestamps = torch.cat(
                        (timestamps, curr_block.timestamps.flip(dims=[0]).cpu()), dim=0)
                    edge_ids = torch.cat(
                        (edge_ids, curr_block.edge_ids.flip(dims=[0]).cpu()), dim=0)

                    curr_block = curr_block.next_block
                    continue

        return target_vertices, timestamps, edge_ids
