from typing import List

import numpy as np
from dgl.heterograph import DGLBlock

from libdgnn import DynamicGraph, SamplingPolicy, _TemporalSampler


class TemporalSampler:
    def __init__(self, graph: DynamicGraph, fanouts: List[int], strategy: SamplingPolicy,
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
        self._sampler = _TemporalSampler(graph, fanouts, strategy, num_snapshots,
                                         snapshot_time_window, num_workers)

    def sample(self, target_vertices: np.array, timestamps: np.array, prop_time: bool = False, reverse: bool = False) \
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
        # TODO: convert to dgl block
        pass

     def _sample_layer_from_root(self, fanout: int, target_vertices: torch.Tensor,
                                timestamps: torch.Tensor, prop_time: bool = False, reverse: bool = False) -> List[DGLBlock]:
         pass
