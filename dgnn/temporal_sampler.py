from dataclasses import dataclass
from typing import List

import torch

from .dynamic_graph import DynamicGraph


@dataclass
class TemporalGraphBlock:
    source_vertices: torch.LongTensor = torch.LongTensor(0)
    target_vertices: torch.LongTensor = torch.LongTensor(0)
    timestamps: torch.FloatTensor = torch.FloatTensor(0)


class TemporalSampler:

    def __init__(self, graph: DynamicGraph, fanouts: List[int],
                 strategy: str = 'recent', num_snapshots: int = 1,
                 snapshot_time_window: float = 0):
        """
        Initialize the sampler.

        Arguments:
            graph: the dynamic graph
            fanouts: fanouts of each layer
            strategy: sampling strategy, 'recent' or 'uniform'
            num_snapshots: number of snapshots to sample
            snapshot_time_window: time window every snapshot covers
        """
        self._graph = graph
        self._fanouts = fanouts
        if strategy not in ['recent', 'uniform']:
            raise ValueError(
                'Sampling strategy must be either recent or uniform')
        self._strategy = strategy
        self._num_snapshots = num_snapshots
        self._snapshot_time_window = snapshot_time_window

    def sample(self, vertices: torch.Tensor, timestamps: torch.Tensor):
        """
        Sample k-hop neighbors of given vertices.

        Arguments:
            vertices: given vertices
            timestamps: timestamps of given vertices

        Returns: message flow graphs
        """
        raise NotImplementedError

    def sample_layer(self, fanout: int, vertices: torch.Tensor, timestamps: torch.Tensor) -> TemporalGraphBlock:
        """
        Sample 1-hop neighbors of given vertices. Arguments: fanout: fanout of the layer
            vertices: given vertices
            timestamps: timestamps of given vertices

        Returns: sampled neighbors (source_vertices, target_vertices, timestamps).
        Tensors are on the CPU.
        """
        assert vertices.shape[0] == timestamps.shape[0], "Number of edges must match"
        assert len(vertices.shape) == 1 and len(
            timestamps.shape) == 1, "Given vertices and timestamps must be 1D"

        temporal_block = TemporalGraphBlock()
        # TODO: parallelize this
        for vertex, timestamp in zip(vertices, timestamps):
            vertex, timestamp = int(vertex), float(timestamp)
            if vertex < 0 or vertex >= self._graph.num_vertices():
                raise ValueError("Vertex must be in [0, {})".format(
                    self._graph.num_vertices))

            target_vertices_i, timestamps_i = self._graph.get_neighbors_before_timestamp(
                vertex, timestamp)
            if len(target_vertices_i) == 0:
                continue
            if self._strategy == 'recent':
                target_vertices_i = target_vertices_i[:fanout]
                timestamps_i = timestamps_i[:fanout]
            elif self._strategy == 'uniform':
                indices = torch.randint(0, len(target_vertices_i), (fanout,))
                target_vertices_i = target_vertices_i[indices]
                timestamps_i = timestamps_i[indices]
            else:
                raise ValueError(
                    'Sampling strategy must be either recent or uniform')

            num_edges_sampled = len(target_vertices_i)
            repeated_source_vetices = torch.full((num_edges_sampled,), vertex)
            temporal_block.source_vertices = torch.cat(
                (temporal_block.source_vertices, repeated_source_vetices), dim=0)
            temporal_block.target_vertices = torch.cat(
                (temporal_block.target_vertices, target_vertices_i), dim=0)
            temporal_block.timestamps = torch.cat(
                (temporal_block.timestamps, timestamps_i), dim=0)

        return temporal_block
