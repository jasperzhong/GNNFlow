import logging
from typing import Tuple

import torch
import torch.distributed

from dgnn import DynamicGraph, TemporalSampler
from dgnn.distributed.dist_graph import DistributedDynamicGraph
from dgnn.distributed.dist_sampler import DistributedTemporalSampler

global DGRAPH
global DSAMPLER


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


def sampler_layer_local(target_vertices: torch.Tensor, timestamps: torch.Tensor, layer: int, snapshot: int):
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
    dsampler = get_dsampler()
    return dsampler.sample_layer_local(target_vertices, timestamps, layer, snapshot)
