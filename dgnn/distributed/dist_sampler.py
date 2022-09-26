import os
from typing import List

import dgl
import numpy as np
import torch
import torch.distributed.rpc as rpc
from dgl.heterograph import DGLBlock

import dgnn.distributed.graph_services as graph_services
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

        self._local_rank = int(os.environ['LOCAL_RANK'])
        self._num_workers_per_machine = torch.cuda.device_count()
        self._num_layers = self._sampler._num_layers
        self._num_snapshots = self._sampler._num_snapshots
        self._partition_table = self._dgraph.get_partition_table()
        self._num_partitions = self._dgraph.num_partitions()

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
        partition_table = self._partition_table
        partition_ids = partition_table[target_vertices]

        futures = []
        for partition_id in range(self._num_partitions):
            partition_mask = partition_ids == partition_id
            if partition_mask.sum() == 0:
                continue
            partition_vertices = target_vertices[partition_mask]
            partition_timestamps = timestamps[partition_mask]

            worker_rank = partition_id * self._num_workers_per_machine + self._local_rank

            futures.append(rpc.rpc_async(
                'worker{}'.format(worker_rank),
                graph_services.sampler_layer_local,
                args=(partition_vertices, partition_timestamps, layer, snapshot)))

        # collect sampling results
        sampling_results = []
        for future in futures:
            sampling_results.append(future.wait())

        # merge sampling results
        return self._merge_sampling_results(sampling_results)

    def _merge_sampling_results(self, sampling_results: List[SamplingResult]) -> DGLBlock:
        """
        Merge sampling results from different partitions.

        Args:
            sampling_results: sampling results from different partitions.

        Returns:
            merged sampling result.
        """
        assert len(sampling_results) > 0

        col = np.array([], dtype=np.int64)
        row = np.array([], dtype=np.int64)
        num_src_nodes = 0
        num_dst_nodes = 0
        for sampling_result in sampling_results:
            num_dst_nodes += sampling_result.num_dst_nodes
            num_src_nodes += sampling_result.num_src_nodes

        src_nodes = np.array([], dtype=np.int64)
        dst_nodes = np.array([], dtype=np.int64)
        src_timestamps = np.array([], dtype=np.float32)
        dst_timestamps = np.array([], dtype=np.float32)
        delta_timestamps = np.array([], dtype=np.float32)
        eids = np.array([], dtype=np.int64)

        col_offset = num_dst_nodes
        row_offset = 0
        for sampling_result in sampling_results:
            src_nodes = np.concatenate((src_nodes, sampling_result.src_nodes))
            dst_nodes = np.concatenate((dst_nodes, sampling_result.dst_nodes))
            src_timestamps = np.concatenate(
                (src_timestamps, sampling_result.src_timestamps))
            dst_timestamps = np.concatenate(
                (dst_timestamps, sampling_result.dst_timestamps))
            delta_timestamps = np.concatenate(
                (delta_timestamps, sampling_result.delta_timestamps))
            eids = np.concatenate((eids, sampling_result.eids))

            sampling_result.col += col_offset
            sampling_result.row += row_offset
            col = np.concatenate((col, sampling_result.col))
            row = np.concatenate((row, sampling_result.row))

            col_offset += sampling_result.num_src_nodes
            row_offset += sampling_result.num_dst_nodes

        mfg = dgl.create_block((col, row), num_src_nodes=num_src_nodes,
                               num_dst_nodes=num_dst_nodes)

        all_nodes = np.concatenate([dst_nodes, src_nodes])
        all_timestamps = np.concatenate([dst_timestamps, src_timestamps])
        mfg.srcdata['ID'] = torch.from_numpy(all_nodes)
        mfg.srcdata['ts'] = torch.from_numpy(all_timestamps)
        mfg.edata['dt'] = torch.from_numpy(delta_timestamps)
        mfg.edata['ID'] = torch.from_numpy(eids)
        return mfg

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
