import warnings
from typing import List, Optional

import dgl
import torch
import numpy as np
from dgl.heterograph import DGLBlock

from .dynamic_graph import DynamicGraph


class TemporalSampler:

    def __init__(self, graph: DynamicGraph, fanouts:  List[int], strategy: str = 'recent',
                 num_snapshots: int = 1, snapshot_time_window: float = 0,
                 num_workers: int = 1):
        """
        Initialize the sampler.

        Args:
            graph: the dynamic graph
            fanouts: fanouts of each layer
            strategy: sampling strategy, 'recent' or 'uniform'
            num_snapshots: number of snapshots to sample
            snapshot_time_window: time window every snapshot cover. It only makes
                sense when num_snapshots > 1.
            num_workers: number of workers to use for parallel sampling
        """
        self._graph = graph
        assert all([fanout > 0 for fanout in fanouts]), \
            "Fanouts must be positive"
        self._fanouts = fanouts

        if strategy not in ['recent', 'uniform']:
            raise ValueError(
                'Sampling strategy must be either recent or uniform')
        self._strategy = strategy

        if num_snapshots <= 0:
            raise ValueError('Number of snapshots must be positive')

        if num_snapshots == 1 and snapshot_time_window != 0:
            warnings.warn(
                'Snapshot time window must be 0 when num_snapshots = 1. Ignore'
                'the snapshot time window.')

        self._num_snapshots = num_snapshots
        self._snapshot_time_window = snapshot_time_window
        self._num_workers = num_workers

    def sample(self, target_vertices: np.array, timestamps: np.array) \
            -> List[List[DGLBlock]]:
        """
        Sample k-hop neighbors of given vertices.

        Args:
            target_vertices: root vertices to sample. CPU tensor. 
            timestamps: timestamps of target vertices in the graph. CPU tensor.

        Returns: 
            list of message flow graphs (# of graphs = # of snapshots) for
            each layer.
        """
        # np to cpu tensor
        target_vertices = torch.tensor(target_vertices)
        timestamps = torch.tensor(timestamps)

        assert target_vertices.shape[0] == timestamps.shape[0], "Number of edges must match"
        assert len(target_vertices.shape) == 1 and len(
            timestamps.shape) == 1, "Target vertices and timestamps must be 1D tensors"

        blocks = []
        for layer, fanout in enumerate(self._fanouts):
            if layer == 0:
                blocks_i = self._sample_layer_from_root(
                    fanout, target_vertices, timestamps)
            else:
                blocks_i = self._sample_layer_from_previous_layer(
                    fanout, blocks[-1])

            blocks.append(blocks_i)

        blocks.reverse()
        return blocks

    def _sample_layer_from_root(self, fanout: int, target_vertices: torch.Tensor,
                                timestamps: torch.Tensor) -> List[DGLBlock]:

        end_timestamps = timestamps.clone()

        blocks = [None for _ in range(self._num_snapshots)]
        for snapshot in reversed(range(self._num_snapshots)):
            # from the last snapshot, we sample the vertices with the largest
            # timestamps
            rows = torch.LongTensor()
            source_vertices = torch.LongTensor()
            source_timestamps = torch.FloatTensor()
            delta_timestamps = torch.FloatTensor()
            edge_ids = torch.LongTensor()

            # update the timestamps
            offset = self._snapshot_time_window * \
                (self._num_snapshots - snapshot - 1)
            end_timestamps -= offset

            for i in range(len(target_vertices)):
                vertex = int(target_vertices[i])
                end_timestamp = float(end_timestamps[i])
                start_timestamp = end_timestamp - self._snapshot_time_window \
                    if self._snapshot_time_window != 0 else float("-inf")

                result = self._sample_layer_helper(fanout, i, vertex,
                                                   start_timestamp, end_timestamp)
                if result:
                    rows = torch.cat(
                        [rows, result[0]])
                    source_vertices = torch.cat(
                        [source_vertices, result[1]])
                    source_timestamps = torch.cat(
                        [source_timestamps, result[2]])
                    edge_ids = torch.cat([edge_ids, result[3]])
                    delta_timestamp = torch.full_like(
                        result[2], timestamps[i].item()) - result[2]
                    delta_timestamps = torch.cat(
                        [delta_timestamps, delta_timestamp])

            all_vertices = torch.cat((target_vertices, source_vertices), dim=0)
            all_timestamps = torch.cat(
                (timestamps, source_timestamps), dim=0)
            cols = torch.arange(len(target_vertices), len(all_vertices))
            block = dgl.create_block((cols, rows),
                                     num_src_nodes=len(all_vertices),
                                     num_dst_nodes=len(target_vertices))
            block.srcdata['ID'] = all_vertices
            block.srcdata['ts'] = all_timestamps
            block.edata['dt'] = delta_timestamps
            block.edata['ID'] = edge_ids
            blocks[snapshot] = block

        return blocks

    def _sample_layer_from_previous_layer(self, fanout: int, prev_blocks: List[DGLBlock]) \
            -> List[DGLBlock]:

        assert len(
            prev_blocks) == self._num_snapshots, "Number of snapshots must match"

        blocks = [None for _ in range(self._num_snapshots)]
        for snapshot in reversed(range(self._num_snapshots)):
            rows = torch.LongTensor()
            source_vertices = torch.LongTensor()
            source_timestamps = torch.FloatTensor()
            delta_timestamps = torch.FloatTensor()
            edge_ids = torch.LongTensor()

            target_vertices = prev_blocks[snapshot].srcdata['ID']
            timestamps = prev_blocks[snapshot].srcdata['ts']

            end_timestamps = timestamps.clone()

            # update the timestamps
            offset = self._snapshot_time_window * \
                (self._num_snapshots - snapshot - 1)
            end_timestamps -= offset

            for i in range(len(target_vertices)):
                vertex = int(target_vertices[i])
                end_timestamp = float(end_timestamps[i])
                start_timestamp = end_timestamp - self._snapshot_time_window \
                    if self._snapshot_time_window != 0 else float("-inf")
                result = self._sample_layer_helper(fanout, i, vertex,
                                                   start_timestamp, end_timestamp)
                if result:
                    rows = torch.cat(
                        [rows, result[0]])
                    source_vertices = torch.cat(
                        [source_vertices, result[1]])
                    source_timestamps = torch.cat(
                        [source_timestamps, result[2]])
                    edge_ids = torch.cat([edge_ids, result[3]])
                    delta_timestamp = torch.full_like(
                        result[2], timestamps[i].item()) - result[2]
                    delta_timestamps = torch.cat(
                        [delta_timestamps, delta_timestamp])

            all_vertices = torch.cat((target_vertices, source_vertices), dim=0)
            all_timestamps = torch.cat(
                (timestamps, source_timestamps), dim=0)
            cols = torch.arange(len(target_vertices), len(all_vertices))
            block = dgl.create_block((cols, rows),
                                     num_src_nodes=len(all_vertices),
                                     num_dst_nodes=len(target_vertices))
            block.srcdata['ID'] = all_vertices
            block.srcdata['ts'] = all_timestamps
            block.edata['dt'] = delta_timestamps
            block.edata['ID'] = edge_ids
            blocks[snapshot] = block

        return blocks

    def _sample_layer_helper(self, fanout: int, vertex_index: int, vertex: int,
                             start_timestamp: float, end_timestamp: float) \
            -> Optional[List[torch.Tensor]]:

        source_vertices, timestamps, edge_ids = self._graph.get_temporal_neighbors(
            vertex, start_timestamp, end_timestamp)

        if len(source_vertices) == 0:
            return None

        if self._strategy == 'recent' or len(source_vertices) < fanout:
            source_vertices = source_vertices[: fanout]
            timestamps = timestamps[: fanout]
            edge_ids = edge_ids[: fanout]
        elif self._strategy == 'uniform':
            indices = torch.randint(0, len(source_vertices), (fanout,))
            source_vertices = source_vertices[indices]
            timestamps = timestamps[indices]
            edge_ids = edge_ids[indices]

        rows = torch.full((len(source_vertices), ),
                          vertex_index, dtype=torch.long)

        return [rows, source_vertices, timestamps, edge_ids]
