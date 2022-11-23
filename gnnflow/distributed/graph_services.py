import logging
import time
from typing import Tuple

import torch
import torch.distributed

from gnnflow import DynamicGraph, TemporalSampler
from gnnflow.distributed.common import SamplingResultTorch
from gnnflow.distributed.dist_graph import DistributedDynamicGraph
from gnnflow.distributed.dist_sampler import DistributedTemporalSampler
from gnnflow.distributed.kvstore import KVStoreServer

global DGRAPH
global DSAMPLER
global KVSTORE_SERVER
global DIM_NODE
global DIM_EDGE


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


def set_dsampler(sampler: TemporalSampler, dynamic_scheduling: bool = False):
    """
    Set the distributed temporal sampler.

    Args:
        dsampler (DistributedTemporalSampler): The distributed temporal sampler.
    """
    global DSAMPLER
    dgraph = get_dgraph()
    DSAMPLER = DistributedTemporalSampler(sampler, dgraph, dynamic_scheduling)


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

    dgraph.enqueue_add_edges_task(source_vertices.numpy(),
                                  target_vertices.numpy(), timestamps.numpy(), eids.numpy())
    # NB: no need to wait for the task to finish


def set_graph_metadata(num_vertices: int, num_edges: int, max_vertex_id: int, num_partitions: int):
    """
    Set the graph metadata.

    Args:
        num_vertices (int): The number of vertices.
        num_edges (int): The number of edges.
        max_vertex_id (int): The maximum vertex ID.
        num_partitions (int): The number of partitions.
    """
    dgraph = get_dgraph()
    dgraph.set_num_vertices(num_vertices)
    dgraph.set_num_edges(num_edges)
    dgraph.set_max_vertex_id(max_vertex_id)
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


def sample_layer_local(target_vertices: torch.Tensor, timestamps: torch.Tensor,
                       layer: int, snapshot: int) -> SamplingResultTorch:
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
    logging.debug("Rank %d: receiving sample_layer_local request. #target_vertices: %d",
                  torch.distributed.get_rank(), target_vertices.size(0))

    dsampler = get_dsampler()
    ret = SamplingResultTorch()
    handle = dsampler.enqueue_sampling_task(
        target_vertices.numpy(), timestamps.numpy(), layer, snapshot, ret)

    # Wait for the sampling task to finish.
    while not dsampler.poll(handle):
        time.sleep(0.001)

    logging.debug("Rank %d: Sampling task %d finished. num sampled vertices: %d",
                  torch.distributed.get_rank(), handle, ret.num_src_nodes)
    return ret


def sample_layer_local_proxy(target_vertices: torch.Tensor, timestamps: torch.Tensor,
                             layer: int, snapshot: int):
    """
    Dispatch the sample_layer_local request to the correct rank.

    Args:
        target_vertices (torch.Tensor): The target vertices.
        timestamps (torch.Tensor): The timestamps.
        layer (int): The layer.
        snapshot (int): The snapshot.

    Returns:
        RemoteReference of SamplingResultTorch: The remote reference of the sampling result.
    """
    dsampler = get_dsampler()
    rref = dsampler.dispatch_sampling_task(
        target_vertices.numpy(), timestamps.numpy(), layer, snapshot)
    return rref


def push_tensors(keys: torch.Tensor, tensors: torch.Tensor, mode: str):
    """
    Push tensors to the remote workers for KVStore servers.

    Args:
        keys (torch.Tensor): The key of the tensors.
        tensors (torch.Tensor): The tensors.
    """
    kvstore_server = get_kvstore_server()
    kvstore_server.push(keys, tensors, mode)


def pull_tensors(keys: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Pull tensors from the remote workers for KVStore servers.

    Args:
        keys (torch.Tensor): The key of the tensors.
        mode (str): The mode of the pull operation.

    Returns:
        torch.Tensor: The pulled tensors.
    """
    kvstore_server = get_kvstore_server()
    return kvstore_server.pull(keys, mode)


def init_cache(capacity: int) -> Tuple[torch.Tensor, torch.Tensor]:
    kvstore_server = get_kvstore_server()
    keys = torch.tensor(list(kvstore_server._edge_feat_map.keys()))
    cache_edge_id = keys[:capacity]
    feats = kvstore_server.pull(cache_edge_id, mode='edge')
    return cache_edge_id, feats


def reset_memory():
    """
    Reset all the values in the memory & mailbox.
    """
    kvstore_server = get_kvstore_server()
    kvstore_server.reset_memory()


def set_dim_node(dim_node: int):
    """
    Set the dim node.

    Args:
        dim_node (int): The dimension of node features.
    """
    global DIM_NODE
    DIM_NODE = dim_node


def get_dim_node() -> int:
    """
    Get the dim_node.

    Returns:
        int: The dimension of node features.
    """
    global DIM_NODE
    if DIM_NODE is None:
        raise RuntimeError(
            "The dim_node has not been initialized.")
    return DIM_NODE


def set_dim_edge(dim_edge: int):
    """
    Set the dim edge.

    Args:
        dim_node (int): The dimension of node features.
    """
    global DIM_EDGE
    DIM_EDGE = dim_edge


def get_dim_edge() -> int:
    """
    Get the dim_edge.

    Returns:
        int: The dimension of edge features.
    """
    global DIM_EDGE
    if DIM_EDGE is None:
        raise RuntimeError(
            "The dim_edge has not been initialized.")
    return DIM_EDGE


def set_dim_node_edge(dim_node: int, dim_edge: int):
    """
    Set the dim node/edge.

    Args:
        dim_node (int): The dimension of node/edge features.
    """
    global DIM_EDGE
    global DIM_NODE
    DIM_EDGE = dim_edge
    DIM_NODE = dim_node


def get_dim_node_edge() -> Tuple[int, int]:
    """
    Get the dim_edge.

    Returns:
        int: The dimension of edge features.
    """
    global DIM_EDGE
    if DIM_EDGE is None:
        raise RuntimeError(
            "The dim_edge has not been initialized.")

    global DIM_NODE
    if DIM_NODE is None:
        raise RuntimeError(
            "The dim_node has not been initialized.")

    return DIM_NODE, DIM_EDGE
