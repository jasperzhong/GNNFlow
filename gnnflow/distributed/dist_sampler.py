import logging
import threading
import time
from queue import Queue
from typing import List

import dgl
import numpy as np
import torch
import torch.distributed
import torch.distributed.rpc as rpc
from dgl.heterograph import DGLBlock

import gnnflow.distributed.graph_services as graph_services
from gnnflow import TemporalSampler
from gnnflow.distributed.common import SamplingResultTorch
from gnnflow.distributed.utils import HandleManager
from gnnflow.distributed.dist_graph import DistributedDynamicGraph
from gnnflow.utils import local_rank, local_world_size
from libgnnflow import SamplingResult


class DistributedTemporalSampler:
    """
    Distributed Temporal Sampler API
    """

    def __init__(self, sampler: TemporalSampler, dgraph: DistributedDynamicGraph):
        """
        Initialize the distributed temporal sampler.

        Args:
            sampler (TemporalSampler): The temporal sampler.
            dgraph (DistributedDynamicGraph): The distributed dynamic graph.
        """
        self._sampler = sampler
        self._dgraph = dgraph

        self._rank = torch.distributed.get_rank()
        self._local_rank = local_rank()
        self._local_world_size = local_world_size()
        self._num_layers = self._sampler._num_layers
        self._num_snapshots = self._sampler._num_snapshots
        self._partition_table = self._dgraph.get_partition_table()
        self._num_partitions = self._dgraph.num_partitions()

        self._handle_manager = HandleManager()
        self._sampling_thread = threading.Thread(target=self._sampling_loop)
        self._sampling_task_queue = Queue()
        self._sampling_thread.start()

        # profiling
        self._sampling_time = torch.zeros(self._num_partitions)

    def _sampling_loop(self):
        while True:
            while not self._sampling_task_queue.empty():
                target_vertices, timestamps, layer, snapshot, result, \
                    handle = self._sampling_task_queue.get()

                ret = self.sample_layer_local(
                    target_vertices, timestamps, layer, snapshot)

                self._transform_output(ret, result)

                self._handle_manager.mark_done(handle)
            time.sleep(0.001)

    def _transform_output(self, input: SamplingResult, output: SamplingResultTorch):
        output.row = torch.from_numpy(input.row())
        output.col = torch.from_numpy(input.col())
        output.num_src_nodes = input.num_src_nodes()
        output.num_dst_nodes = input.num_dst_nodes()
        output.all_nodes = torch.from_numpy(input.all_nodes())
        output.all_timestamps = torch.from_numpy(input.all_timestamps())
        output.delta_timestamps = torch.from_numpy(input.delta_timestamps())
        output.eids = torch.from_numpy(input.eids())

    def enqueue_sampling_task(self, target_vertices: np.ndarray, timestamps: np.ndarray,
                              layer: int, snapshot: int, result:  SamplingResultTorch):
        handle = self._handle_manager.allocate_handle()
        self._sampling_task_queue.put(
            (target_vertices, timestamps, layer, snapshot, result, handle))
        return handle

    def poll(self, handle: int):
        return self._handle_manager.poll(handle)

    def get_sampling_time(self):
        """
        Get the sampling time of all partitions.

        Returns:
            sampling time of all partitions. shape: [num_partition, num_partition]
        """
        sampling_time = self._sampling_time.clone()
        # all gather
        all_sampling_time = [torch.zeros_like(
            sampling_time) for _ in range(self._num_partitions * self._local_world_size)]
        torch.distributed.all_gather(all_sampling_time, sampling_time)

        # merge by local machine
        all_sampling_time = torch.stack(all_sampling_time)
        all_sampling_time = all_sampling_time.reshape(
            self._num_partitions, self._local_world_size, self._num_partitions)
        all_sampling_time = all_sampling_time.sum(dim=1)

        return all_sampling_time

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
        masks = []
        for partition_id in range(self._num_partitions):
            partition_mask = partition_ids == partition_id
            if partition_mask.sum() == 0:
                continue
            partition_vertices = torch.from_numpy(
                target_vertices[partition_mask]).contiguous()
            partition_timestamps = torch.from_numpy(
                timestamps[partition_mask]).contiguous()

            # static scheduling
            worker_rank = partition_id * self._local_world_size + self._local_rank
            if worker_rank == self._rank:
                logging.debug(
                    "worker %d call local sample_layer_local", self._rank)
                start = time.time()
                futures.append(graph_services.sample_layer_local(partition_vertices, partition_timestamps,
                                                                 layer, snapshot))
                self._sampling_time[partition_id] += time.time() - start
            else:
                logging.debug(
                    "worker %d call remote sample_layer_local on worker %d", self._rank, worker_rank)
                futures.append(rpc.rpc_async(
                    'worker{}'.format(worker_rank),
                    graph_services.sample_layer_local,
                    args=(partition_vertices, partition_timestamps, layer, snapshot)))
            masks.append(partition_mask)

        # collect sampling results
        sampling_results = []
        for i, future in enumerate(futures):
            if isinstance(future, SamplingResultTorch):
                sampling_results.append(future)
            else:
                start = time.time()
                sampling_results.append(future.wait())
                self._sampling_time[i] += time.time() - start

        # deal with non-partitioned nodes
        non_partition_mask = partition_ids == -1
        if non_partition_mask.sum() > 0:
            masks.append(non_partition_mask)
            result = SamplingResultTorch()
            result.row = torch.tensor([])
            result.num_dst_nodes = int(non_partition_mask.sum())
            result.num_src_nodes = result.num_dst_nodes
            result.all_nodes = torch.from_numpy(
                target_vertices[non_partition_mask]).contiguous()
            result.all_timestamps = torch.from_numpy(
                timestamps[non_partition_mask]).contiguous()
            result.delta_timestamps = torch.tensor([])
            result.eids = torch.tensor([])
            sampling_results.append(result)

        # merge sampling results
        mfg = self._merge_sampling_results(sampling_results, masks)
        assert mfg.num_dst_nodes() == len(
            target_vertices), 'Layer {}\tError: Number of destination nodes does not match'.format(layer)
        return mfg

    def _merge_sampling_results(self, sampling_results: List[SamplingResultTorch], masks: List[torch.Tensor]) -> DGLBlock:
        """
        Merge sampling results from different partitions.

        Args:
            sampling_results: sampling results from different partitions.
            masks: masks for each partition.

        Returns:
            merged sampling result.
        """
        assert len(sampling_results) > 0

        all_num_nodes = 0
        all_num_dst_nodes = 0
        all_num_src_nodes = 0
        for sampling_result in sampling_results:
            all_num_dst_nodes += sampling_result.num_dst_nodes
            all_num_nodes += sampling_result.num_src_nodes

        all_num_src_nodes = all_num_nodes - all_num_dst_nodes

        all_col = np.arange(start=all_num_dst_nodes,
                            stop=all_num_nodes, dtype=np.int64)
        all_row = np.zeros(all_num_src_nodes, dtype=np.int64)
        all_src_nodes = np.zeros(all_num_src_nodes, dtype=np.int64)
        all_dst_nodes = np.zeros(all_num_dst_nodes, dtype=np.int64)
        all_src_timestamps = np.zeros(all_num_src_nodes, dtype=np.float32)
        all_dst_timestamps = np.zeros(all_num_dst_nodes, dtype=np.float32)
        all_delta_timestamps = np.zeros(all_num_src_nodes, dtype=np.float32)
        all_eids = np.zeros(all_num_src_nodes, dtype=np.int64)

        offset = 0
        # use mask to restore dst node order
        for i, sampling_result in enumerate(sampling_results):
            num_dst_nodes = sampling_result.num_dst_nodes
            num_edges = sampling_result.num_src_nodes - num_dst_nodes
            dst_nodes = sampling_result.all_nodes[:num_dst_nodes]
            dst_timestamps = sampling_result.all_timestamps[:num_dst_nodes]
            src_nodes = sampling_result.all_nodes[num_dst_nodes:]
            src_timestamps = sampling_result.all_timestamps[num_dst_nodes:]
            delta_timestamps = sampling_result.delta_timestamps
            eids = sampling_result.eids

            mask = masks[i]
            dst_idx = mask.nonzero().squeeze(dim=1).numpy()
            all_row[offset:offset + num_edges] = dst_idx[sampling_result.row]
            all_dst_nodes[dst_idx] = dst_nodes
            all_dst_timestamps[dst_idx] = dst_timestamps

            all_src_nodes[offset:offset + num_edges] = src_nodes
            all_src_timestamps[offset:offset + num_edges] = src_timestamps
            all_delta_timestamps[offset:offset + num_edges] = delta_timestamps
            all_eids[offset:offset + num_edges] = eids

            offset += num_edges

        logging.debug('num_src_nodes: {}'.format(all_num_nodes))
        logging.debug('num_dst_nodes: {}'.format(all_num_dst_nodes))

        mfg = dgl.create_block((all_col, all_row), num_src_nodes=all_num_nodes,
                               num_dst_nodes=all_num_dst_nodes)

        all_nodes = np.concatenate([all_dst_nodes, all_src_nodes])
        all_timestamps = np.concatenate(
            [all_dst_timestamps, all_src_timestamps])
        mfg.srcdata['ID'] = torch.from_numpy(all_nodes)
        mfg.srcdata['ts'] = torch.from_numpy(all_timestamps)
        mfg.edata['dt'] = torch.from_numpy(all_delta_timestamps)
        mfg.edata['ID'] = torch.from_numpy(all_eids)
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
        logging.debug("Rank %d: sampling layer %d, snapshot %d, %d target vertices",
                      self._rank, layer, snapshot, len(target_vertices))
        self._dgraph.wait_for_all_updates_to_finish()
        ret = self._sampler.sample_layer(
            target_vertices, timestamps, layer, snapshot, False)
        return ret
