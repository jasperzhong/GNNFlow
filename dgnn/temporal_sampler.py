from typing import List

import torch
import dgl
from dgl.heterograph import DGLBlock

from .dynamic_graph import DynamicGraph


class TemporalSampler:

    def __init__(self, graph: DynamicGraph, strategy: str = 'recent',
                 num_snapshots: int = 1, snapshot_time_window: float = 0):
        """
        Initialize the sampler.

        Arguments:
            graph: the dynamic graph
            strategy: sampling strategy, 'recent' or 'uniform'
            num_snapshots: number of snapshots to sample
            snapshot_time_window: time window every snapshot covers
        """
        self._graph = graph
        if strategy not in ['recent', 'uniform']:
            raise ValueError(
                'Sampling strategy must be either recent or uniform')
        self._strategy = strategy
        self._num_snapshots = num_snapshots
        self._snapshot_time_window = snapshot_time_window

    def sample(self, fanouts: List[int], target_vertices: torch.Tensor,
               timestamps: torch.Tensor) -> List[List[DGLBlock]]:
        """
        Sample k-hop neighbors of given vertices.

        Arguments:
            fanouts: fanouts of each layer
            target_vertices: root vertices to sample
            timestamps: timestamps of target vertices

        Returns: list of message flow graphs (# of graphs = # of snapshots) for 
            each layer.
        """
        assert all([fanout > 0 for fanout in fanouts]), \
            "Fanouts must be positive"
        assert target_vertices.shape[0] == timestamps.shape[0], "Number of edges must match"
        assert len(target_vertices.shape) == 1 and len(
            timestamps.shape) == 1, "Target vertices and timestamps must be 1D tensors"

        blocks = []
        # TODO: support multiple snapshots
        for layer in range(len(fanouts)):
            blocks_i = self.sample_layer(
                fanouts[layer], target_vertices, timestamps)
            blocks.append(blocks_i)

            target_vertices = blocks_i[0].srcdata["ID"][blocks_i[0].num_dst_nodes():]
            timestamps = blocks_i[0].srcdata["ts"][blocks_i[0].num_dst_nodes():]

        blocks.reverse()
        return blocks

    def sample_layer(self, fanout: int, target_vertices: torch.Tensor,
                     timestamps: torch.Tensor) -> List[DGLBlock]:
        """
        Sample 1-hop neighbors of target vertices. 

        Arguments: 
            fanout: fanout of the layer
            target_vertices: root vertices to sample
            timestamps: timestamps of target vertices

        Returns: message flow graphs (# of graphs = # of snapshots)
        """
        assert target_vertices.shape[0] == timestamps.shape[0], "Number of edges must match"
        assert len(target_vertices.shape) == 1 and len(
            timestamps.shape) == 1, "Given vertices and timestamps must be 1D"

        blocks = []
        # TODO: support multiple snapshots
        for snapshot in range(self._num_snapshots):
            repeated_target_vertices = torch.LongTensor()
            source_vertices = torch.LongTensor()
            source_timestamps = torch.FloatTensor()
            delta_timestamps = torch.FloatTensor()
            edge_ids = torch.LongTensor()

            # TODO: parallelize this
            for vertex, timestamp in zip(target_vertices, timestamps):
                vertex, timestamp = int(vertex), float(timestamp)
                if vertex < 0 or vertex >= self._graph.num_vertices:
                    raise ValueError("Vertex must be in [0, {})".format(
                        self._graph.num_vertices))

                # TODO: remove memcpy's synchronization
                source_vertices_i, timestamps_i, edge_ids_i = self._graph.get_neighbors_before_timestamp(
                    vertex, timestamp)

                if len(source_vertices_i) == 0:
                    continue

                if self._strategy == 'recent':
                    source_vertices_i = source_vertices_i[:fanout]
                    timestamps_i = timestamps_i[:fanout]
                    edge_ids_i = edge_ids_i[:fanout]
                elif self._strategy == 'uniform':
                    indices = torch.randint(
                        0, len(source_vertices_i), (fanout,))
                    source_vertices_i = source_vertices_i[indices]
                    timestamps_i = timestamps_i[indices]
                    edge_ids_i = edge_ids_i[indices]
                else:
                    raise ValueError(
                        'Sampling strategy must be either recent or uniform')

                delta_timestamps_i = torch.FloatTensor(
                    [timestamp - t for t in timestamps_i])

                repeated_target_vertices = torch.cat((repeated_target_vertices, torch.full(
                    (len(source_vertices_i),), vertex, dtype=torch.long)))

                source_vertices = torch.cat(
                    (source_vertices, source_vertices_i), dim=0)
                source_timestamps = torch.cat(
                    (source_timestamps, timestamps_i), dim=0)
                delta_timestamps = torch.cat(
                    (delta_timestamps, delta_timestamps_i), dim=0)
                edge_ids = torch.cat((edge_ids, edge_ids_i), dim=0)

            all_vertices = torch.cat((target_vertices, source_vertices), dim=0)
            all_timestamps = torch.cat((timestamps, source_timestamps), dim=0)
            block = dgl.create_block((source_vertices, repeated_target_vertices),
                                     num_src_nodes=len(all_vertices),
                                     num_dst_nodes=len(target_vertices))
            block.srcdata['ID'] = all_vertices
            block.srcdata['ts'] = all_timestamps
            block.edata['dt'] = delta_timestamps
            block.edata['ID'] = edge_ids
            blocks.append(block)

        return blocks
