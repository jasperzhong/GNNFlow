from typing import List

import dgl
import numpy as np
import torch
from dgl.heterograph import DGLBlock

from dgnn import TemporalSampler
from dgnn.distributed.dist_graph import DistributedDynamicGraph
from libdgnn import SamplingResult


class DistributedTemporalSampler:

    def __init__(self, sampler: TemporalSampler, dgraph: DistributedDynamicGraph):
        """
        Initialize the distributed temporal sampler.

        Args:
            sampler (TemporalSampler): The temporal sampler.
            dgraph (DistributedDynamicGraph): The distributed dynamic graph.
        """
        self._sampler = sampler
        self._dgraph = dgraph

        self._num_layers = self._sampler._num_layers
        self._num_snapshots = self._sampler._num_snapshots
        self._partition_table = self._dgraph.get_partition_table()

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
        mfgs = []
        for layer in range(self._num_layers):
            mfgs.append([])
            if layer == 0:
                for snapshot in range(self._num_snapshots):
                    mfgs[layer].append(self.sample_layer_global(
                        target_vertices, timestamps, layer, snapshot))
            else:
                for snapshot in range(self._num_snapshots):
                    prev_mfg = mfgs[layer - 1][snapshot]
                    all_vertices = prev_mfg.srcdata['ID'].numpy()
                    all_timestamps = prev_mfg.srcdata['ts'].numpy()
                    mfgs[layer].append(self.sample_layer_global(
                        all_vertices, all_timestamps, layer, snapshot))

        mfgs.reverse()
        return mfgs

    def sample_layer_global(self, target_vertices: np.ndarray, timestamps: np.ndarray,
                            layer: int, snapshot: int) -> DGLBlock:
        """
        Sample neighbors of given vertices in a specific layer and snapshot.

        Args:
            target_vertices: root vertices to sample. CPU tensor.
            timestamps: timestamps of target vertices in the graph. CPU tensor.
            layer: layer to sample.
            snapshot: snapshot to sample.

        Returns:
            message flow graph for the specific layer and snapshot.
        """
        # dispatch target vertices and timestamps to different partitions
        pass

    def sample_layer_local(self, target_vertices:  np.ndarray, timestamps: np.ndarray,
                           layer: int, snapshot: int) -> SamplingResult:
        """
        Sample neighbors of given vertices in a specific layer and snapshot.

        Args:
            target_vertices: root vertices to sample. CPU tensor.
            timestamps: timestamps of target vertices in the graph. CPU tensor.
            layer: layer to sample.
            snapshot: snapshot to sample.

        Returns:
            sampling result.
        """
        return self._sampler.sample_layer(target_vertices, timestamps, layer, snapshot, False)
