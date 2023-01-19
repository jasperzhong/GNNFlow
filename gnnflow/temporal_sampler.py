from typing import List, Union

import dgl
import numpy as np
import torch
from dgl.heterograph import DGLBlock

from libgnnflow import SamplingPolicy, SamplingResult, _TemporalSampler

from .dynamic_graph import DynamicGraph


class TemporalSampler:
    """
    TemporalSampler samples k-hop multi-snapshots neighbors of given vertices.
    """

    def __init__(
            self, graph: DynamicGraph, fanouts: List[int],
            sample_strategy: str = "recent", num_snapshots: int = 1,
            snapshot_time_window: float = 0.0, prop_time: bool = False,
            seed: int = 1234, *args, **kwargs):
        """
        Initialize the sampler.

        Args:
            graph: the dynamic graph.
            fanouts: fanouts of each layer.
            samplle_strategy: sampling strategy, 'recent' or 'uniform' (case insensitive).
            num_snapshots: number of snapshots to sample.
            snapshot_time_window: time window every snapshot cover. It only makes
                                  sense when num_snapshots > 1.
            prop_time: whether to propagate timestamps to neighbors.
            seed: random seed.
        """
        sample_strategy = sample_strategy.lower()
        if sample_strategy not in ["recent", "uniform"]:
            raise ValueError("strategy must be 'recent' or 'uniform'")

        if sample_strategy == "recent":
            sample_strategy = SamplingPolicy.RECENT
        else:
            sample_strategy = SamplingPolicy.UNIFORM

        print("TemporalSampler: sample_strategy={}, num_snapshots={}, snapshot_time_window={}, prop_time={}".format(
            sample_strategy, num_snapshots, snapshot_time_window, prop_time))

        self._sampler = _TemporalSampler(
            graph._dgraph, fanouts, sample_strategy, num_snapshots,
            snapshot_time_window, prop_time, seed)
        self._num_layers = len(fanouts)
        self._num_snapshots = num_snapshots

        if 'is_static' in kwargs and kwargs['is_static'] == True:
            self._is_static = True
        else:
            self._is_static = False

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
        if self._is_static:
            sampling_results = self._sampler.sample(
                target_vertices,
                np.full(target_vertices.shape,
                        np.finfo(np.float32).max))
        else:
            sampling_results = self._sampler.sample(
                target_vertices, timestamps)
        return self._to_dgl_block(sampling_results)

    def sample_layer(self, target_vertices:  np.ndarray, timestamps: np.ndarray,
                     layer: int, snapshot: int, to_dgl_block: bool = True) \
            -> Union[DGLBlock, SamplingResult]:
        """
        Sample neighbors of given vertices in a specific layer and snapshot.

        Args:
            target_vertices: root vertices to sample. CPU tensor.
            timestamps: timestamps of target vertices in the graph. CPU tensor.
            layer: layer to sample.
            snapshot: snapshot to sample.

        Returns:
            either a DGLBlock or a SamplingResult.
        """
        sampling_result = self._sampler.sample_layer(
            target_vertices, timestamps, layer, snapshot)
        if to_dgl_block:
            return self._to_dgl_block_layer_snapshot(sampling_result)
        return sampling_result

    def _to_dgl_block(self, sampling_results: SamplingResult) -> List[List[DGLBlock]]:
        mfgs = list()
        for sampling_results_layer in sampling_results:
            for r in sampling_results_layer:
                b = dgl.create_block(
                    (r.col(),
                     r.row()),
                    num_src_nodes=r.num_src_nodes(),
                    num_dst_nodes=r.num_dst_nodes())
                b.srcdata['ID'] = torch.from_numpy(r.all_nodes())
                b.edata['dt'] = torch.from_numpy(r.delta_timestamps())
                b.srcdata['ts'] = torch.from_numpy(r.all_timestamps())
                b.edata['ID'] = torch.from_numpy(r.eids())
                mfgs.append(b)
        mfgs = list(map(list, zip(*[iter(mfgs)] * self._num_snapshots)))
        mfgs.reverse()
        return mfgs

    def _to_dgl_block_layer_snapshot(self, sampling_result: SamplingResult) -> DGLBlock:
        mfg = dgl.create_block(
            (sampling_result.col(),
             sampling_result.row()),
            num_src_nodes=sampling_result.num_src_nodes(),
            num_dst_nodes=sampling_result.num_dst_nodes())
        mfg.srcdata['ID'] = torch.from_numpy(sampling_result.all_nodes())
        mfg.edata['dt'] = torch.from_numpy(sampling_result.delta_timestamps())
        mfg.srcdata['ts'] = torch.from_numpy(sampling_result.all_timestamps())
        mfg.edata['ID'] = torch.from_numpy(sampling_result.eids())
        return mfg
