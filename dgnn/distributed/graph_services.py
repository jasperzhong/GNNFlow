import logging
from typing import Tuple

import torch
import torch.distributed

from dgnn import DynamicGraph
from dgnn.distributed.dist_graph import DistributedDynamicGraph

global DGRAPH


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


def set_graph_metadata(num_vertices: int, num_edges: int):
    """
    Set the graph metadata.

    Args:
        num_vertices (int): The number of vertices.
        num_edges (int): The number of edges.
    """
    dgraph = get_dgraph()
    dgraph.set_num_vertices(num_vertices)
    dgraph.set_num_edges(num_edges)


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
