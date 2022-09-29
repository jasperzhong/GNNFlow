import logging
import time
from typing import List, Tuple

import torch
import torch.distributed

from dgnn import DynamicGraph, TemporalSampler
from dgnn.distributed.common import SamplingResultType
from dgnn.distributed.dist_graph import DistributedDynamicGraph
from dgnn.distributed.dist_sampler import DistributedTemporalSampler
from dgnn.distributed.kvstore import KVStoreServer
from dgnn.distributed.utils import HandleManager

global handle_manager
handle_manager = HandleManager()

global DGRAPH
global DSAMPLER
global KVSTORE_SERVER


def get_dgraph() -> DistributedDynamicGraph:
    """
    Get the dynamic graph instance.

    Returns:
        DistributedDynamicGraph: The dynamic graph instance.
    """
    global DGRAPH
    if DGRAPH is None:
        raise RuntimeError("The dynamic graph has not been initialized.")
    return DGRAPH


def set_dgraph(dgraph: DynamicGraph):
    """
    Set the dynamic graph instance.

    Args:
        dgraph(DynamicGraph): The local partition of dynamic graph.
    """
    global DGRAPH
    DGRAPH = DistributedDynamicGraph(dgraph)


def get_dsampler() -> DistributedTemporalSampler:
    """
    Get the distributed temporal sampler.

    Returns:
        DistributedTemporalSampler: The distributed temporal sampler.
    """
    global DSAMPLER
    if DSAMPLER is None:
        raise RuntimeError(
            "The distributed temporal sampler has not been initialized.")
    return DSAMPLER


def set_dsampler(sampler: TemporalSampler):
    """
    Set the distributed temporal sampler.

    Args:
        dsampler (DistributedTemporalSampler): The distributed temporal sampler.
    """
    global DSAMPLER
    dgraph = get_dgraph()
    DSAMPLER = DistributedTemporalSampler(sampler, dgraph)


def get_kvstore_server() -> KVStoreServer:
    """
    Get the kvstore server

    Returns:
        KVStoreServer: The kvstore server.
    """
    global KVSTORE_SERVER
    if KVSTORE_SERVER is None:
        raise RuntimeError("The kvstore client has not been initialized.")
    return KVSTORE_SERVER


def set_kvstore_server(kvstore_server: KVStoreServer):
    """
    Set the kvstore client.

    Args:
        kvstore_server (KVStoreServer): The kvstore server.
    """
    global KVSTORE_SERVER
    KVSTORE_SERVER = kvstore_server


def add_edges(source_vertices: torch.Tensor, target_vertices: torch.Tensor,
              timestamps: torch.Tensor, eids: torch.Tensor):
    """
    Add edges to the dynamic graph.

    Args:
        source_vertices (torch.Tensor): The source vertices of the edges.
        target_vertices (torch.Tensor): The target vertices of the edges.
        timestamps (torch.Tensor): The timestamps of the edges.
        eids (torch.Tensor): The edge IDs of the edges.
    """
    dgraph = get_dgraph()
    logging.debug("Rank %d: Adding %d edges to the dynamic graph.",
                  torch.distributed.get_rank(), source_vertices.size(0))
    dgraph.add_edges(source_vertices.numpy(),
                     target_vertices.numpy(), timestamps.numpy(), eids.numpy())


def set_graph_metadata(num_vertices: int, num_edges: int, num_partitions: int):
    """
    Set the graph metadata.

    Args:
        num_vertices (int): The number of vertices.
        num_edges (int): The number of edges.
    """
    dgraph = get_dgraph()
    dgraph.set_num_vertices(num_vertices)
    dgraph.set_num_edges(num_edges)
    dgraph.set_num_partitions(num_partitions)


def set_partition_table(partition_table: torch.Tensor):
    """
    Set the partition table.

    Args:
        partition_table (torch.Tensor): The partition table.
    """
    dgraph = get_dgraph()
    dgraph.set_partition_table(partition_table)


def get_partition_table() -> torch.Tensor:
    """
    Get the partition table.

    Returns:
        torch.Tensor: The partition table.
    """
    dgraph = get_dgraph()
    return dgraph.get_partition_table()


def num_vertices() -> int:
    """
    Get the number of vertices in the dynamic graph.

    Returns:
        int: The number of vertices.
    """
    dgraph = get_dgraph()
    return dgraph.num_vertices()


def num_edges() -> int:
    """
    Get the number of edges in the dynamic graph.

    Returns:
        int: The number of edges.
    """
    dgraph = get_dgraph()
    return dgraph.num_edges()


def out_degree(vertex: int) -> int:
    """
    Get the out-degree of a vertex.

    Args:
        vertex (int): The vertex.

    Returns:
        int: The out-degree of the vertex.
    """
    # TODO: rpc call
    pass


def get_temporal_neighbors(vertex: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the temporal neighbors of a vertex at a given timestamp.

    Args:
        vertex (int): The vertex.

    Returns:
        torch.Tensor: The temporal neighbors of the vertex.
    """
    # TODO: rpc call
    pass


def sample_layer_local(target_vertices: torch.Tensor, timestamps: torch.Tensor, layer: int, snapshot: int):
    """
    Sample neighbors of given vertices in a specific layer and snapshot locally.

    Args:
        target_vertices (torch.Tensor): The target vertices.
        timestamps (torch.Tensor): The timestamps.
        layer (int): The layer.
        snapshot (int): The snapshot.

    Returns:
        torch.Tensor: The temporal neighbors of the vertex.
    """
    def callback(handle: int):
        global handle_manager
        handle_manager.mark_done(handle)

    dsampler = get_dsampler()
    ret = SamplingResultType()
    handle = handle_manager.allocate_handle()
    dsampler.enqueue_sampling_task(
        target_vertices.numpy(), timestamps.numpy(), layer, snapshot, ret, callback, handle)

    # Wait for the sampling task to finish.
    while not handle_manager.poll(handle):
        time.sleep(0.01)

    return ret


def push_tensors(keys: torch.Tensor, tensors: List[torch.Tensor]):
    """
    Push tensors to the remote workers for KVStore servers.

    Args:
        keys (torch.Tensor): The key of the tensors.
        tensors (List[torch.Tensor]): The tensors.
    """
    kvstore_server = get_kvstore_server()
    kvstore_server.push(keys, tensors)


def pull_tensors(keys: torch.Tensor) -> List[torch.Tensor]:
    """
    Pull tensors from the remote workers for KVStore servers.

    Args:
        keys (torch.Tensor): The key of the tensors.

    Returns:
        List[torch.Tensor]: The tensors.
    """
    kvstore_server = get_kvstore_server()
    return kvstore_server.pull(keys)
