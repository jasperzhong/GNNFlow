import concurrent.futures
import threading
import warnings
from typing import List, Optional

import dgl
import torch
from dgl.heterograph import DGLBlock

from .dynamic_graph import DynamicGraph


class TemporalSampler:

    def __init__(self, graph: DynamicGraph, fanouts:  List[int], strategy: str = 'recent',
                 num_snapshots: int = 1, snapshot_time_window: float = 0,
                 num_workers: int = 1):
        """
        Initialize the sampler.

        Arguments:
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

    def sample(self, target_vertices: torch.Tensor, timestamps: torch.Tensor) \
            -> List[List[DGLBlock]]:
        """
        Sample k-hop neighbors of given vertices.

        Arguments:
            target_vertices: root vertices to sample
            timestamps: timestamps of target vertices

        Returns: list of message flow graphs (# of graphs = # of snapshots) for
            each layer.
        """
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

        blocks = [None for _ in range(self._num_snapshots)]
        for i, snapshot in enumerate(reversed(range(self._num_snapshots))):
            # from the last snapshot, we sample the vertices with the largest
            # timestamps
            repeated_target_vertices = torch.LongTensor()
            source_vertices = torch.LongTensor()
            source_timestamps = torch.FloatTensor()
            delta_timestamps = torch.FloatTensor()
            edge_ids = torch.LongTensor()

            with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                futures = []
                for i in range(len(target_vertices)):
                    vertex = int(target_vertices[i])
                    end_timestamp = float(
                        timestamps[i]) - self._snapshot_time_window * i
                    start_timestmap = end_timestamp - self._snapshot_time_window \
                        if self._snapshot_time_window != 0 else float("-inf")
                    future = executor.submit(
                        self._sample_layer_helper, fanout, vertex,
                        start_timestmap, end_timestamp)
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        repeated_target_vertices = torch.cat(
                            [repeated_target_vertices, result[0]])
                        source_vertices = torch.cat(
                            [source_vertices, result[1]])
                        source_timestamps = torch.cat(
                            [source_timestamps, result[2]])
                        delta_timestamps = torch.cat(
                            [delta_timestamps, result[3]])
                        edge_ids = torch.cat([edge_ids, result[4]])

            all_vertices = torch.cat((target_vertices, source_vertices), dim=0)
            all_timestamps = torch.cat((timestamps, source_timestamps), dim=0)
            block = dgl.create_block((source_vertices, repeated_target_vertices),
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
        for i, snapshot in enumerate(reversed(range(self._num_snapshots))):
            repeated_target_vertices = torch.LongTensor()
            source_vertices = torch.LongTensor()
            source_timestamps = torch.FloatTensor()
            delta_timestamps = torch.FloatTensor()
            edge_ids = torch.LongTensor()

            with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                futures = []
                start_index = prev_blocks[snapshot].num_dst_nodes()
                target_vertices = prev_blocks[snapshot].srcdata['ID'][start_index:]
                timestamps = prev_blocks[snapshot].srcdata['ts'][start_index:]

                for i in range(len(target_vertices)):
                    vertex = int(target_vertices[i])
                    end_timestamp = float(
                        timestamps[i]) - self._snapshot_time_window * i
                    start_timestmap = end_timestamp - self._snapshot_time_window \
                        if self._snapshot_time_window != 0 else float("-inf")
                    future = executor.submit(
                        self._sample_layer_helper, fanout, vertex,
                        start_timestmap, end_timestamp)
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        repeated_target_vertices = torch.cat(
                            [repeated_target_vertices, result[0]])
                        source_vertices = torch.cat(
                            [source_vertices, result[1]])
                        source_timestamps = torch.cat(
                            [source_timestamps, result[2]])
                        delta_timestamps = torch.cat(
                            [delta_timestamps, result[3]])
                        edge_ids = torch.cat([edge_ids, result[4]])

            all_vertices = torch.cat((target_vertices, source_vertices), dim=0)
            all_timestamps = torch.cat((timestamps, source_timestamps), dim=0)
            block = dgl.create_block((source_vertices, repeated_target_vertices),
                                     num_src_nodes=len(all_vertices),
                                     num_dst_nodes=len(target_vertices))
            block.srcdata['ID'] = all_vertices
            block.srcdata['ts'] = all_timestamps
            block.edata['dt'] = delta_timestamps
            block.edata['ID'] = edge_ids
            blocks[snapshot] = block

        return blocks

    def _sample_layer_helper(self, fanout: int, vertex: int, start_timestamp: float,
                             end_timestamp: float) -> Optional[List[torch.Tensor]]:

        source_vertices, timestamps, edge_ids = self._graph.get_temporal_neighbors(
            vertex, start_timestamp, end_timestamp)

        if len(source_vertices) == 0:
            return None

        if self._strategy == 'recent':
            source_vertices = source_vertices[: fanout]
            timestamps = timestamps[: fanout]
            edge_ids = edge_ids[: fanout]
        elif self._strategy == 'uniform':
            indices = torch.randint(0, len(source_vertices), (fanout,))
            source_vertices = source_vertices[indices]
            timestamps = timestamps[indices]
            edge_ids = edge_ids[indices]

        repeated_target_vertices = torch.full_like(source_vertices, vertex)

        delta_timestamps = torch.full_like(
            timestamps, end_timestamp) - timestamps

        return [repeated_target_vertices, source_vertices, timestamps,
                delta_timestamps, edge_ids]
