from typing import List

import torch
import numpy as np
import dgl
from dgl.heterograph import DGLBlock

from libdgnn import (DynamicGraph, SamplingPolicy, SamplingResult,
        _TemporalSampler)


class TemporalSampler:
    def __init__(self, graph: DynamicGraph, fanouts: List[int], strategy: str = "recent",
            num_snapshots: int = 1, snapshot_time_window: float = 0.0,
            prop_time: bool = False, reverse: bool = False, seed: int = 1234):
        """
        Initialize the sampler.
        Args:
            graph: the dynamic graph
            fanouts: fanouts of each layer
            strategy: sampling strategy, 'recent' or 'uniform'
            num_snapshots: number of snapshots to sample
            snapshot_time_window: time window every snapshot cover. It only makes
                sense when num_snapshots > 1.
        """
        if strategy not in ["recent", "uniform"]:
            raise ValueError("strategy must be 'recent' or 'uniform'")

        if strategy == "recent":
            strategy = SamplingPolicy.RECENT
        else:
            strategy = SamplingPolicy.UNIFORM

        self._sampler = _TemporalSampler(graph, fanouts, strategy, num_snapshots,
                snapshot_time_window, prop_time, seed)
        self._num_snapshots = num_snapshots
        self._reverse = reverse

    def sample(self, target_vertices: np.ndarray, timestamps: np.ndarray) -> List[List[DGLBlock]]:
        """
        Sample k-hop neighbors of given vertices.
        Args:
            target_vertices: root vertices to sample. CPU tensor.
            timestamps: timestamps of target vertices in the graph. CPU tensor.
        Returns:
            list of message flow graphs (# of graphs = # of snapshots) for
            each layer.
        """
        sampling_results = self._sampler.sample(
                target_vertices, timestamps)
        return self._to_dgl_block(sampling_results)

    def _to_dgl_block(self, sampling_results: SamplingResult) -> List[List[DGLBlock]]:
        mfgs = list()
        for sampling_results_layer in sampling_results:
            for r in sampling_results_layer:
                if not self._reverse:
                    b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.num_src_nodes(), num_dst_nodes=r.num_dst_nodes())
                    b.srcdata['ID'] = torch.from_numpy(r.all_nodes())
                    b.edata['dt'] = torch.from_numpy(r.delta_timestamps())
                    b.srcdata['ts'] = torch.from_numpy(r.all_timestamps())
                else:
                    b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.num_dst_nodes(), num_dst_nodes=r.num_src_nodes())
                    b.dstdata['ID'] = torch.from_numpy(r.all_nodes())
                    b.edata['dt'] = torch.from_numpy(r.delta_timestamps())
                    b.dstdata['ts'] = torch.from_numpy(r.all_timestamps())
                b.edata['ID'] = torch.from_numpy(r.eids())
                mfgs.append(b)
        mfgs = list(map(list, zip(*[iter(mfgs)] * self._num_snapshots)))
        mfgs.reverse()
        return mfgs
