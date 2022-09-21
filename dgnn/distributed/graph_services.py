import logging
from typing import Tuple

import numpy as np
import torch
import torch.distributed

from dgnn import DynamicGraph

global DGRAPH


def get_dgraph() -> DynamicGraph:
    """
    Get the dynamic graph instance.

    Returns:
        DynamicGraph: The dynamic graph instance.
    """
    global DGRAPH
    if DGRAPH is None:
        raise RuntimeError("The dynamic graph has not been initialized.")
    return DGRAPH


def set_dgraph(dgraph: DynamicGraph):
    """
    Set the dynamic graph instance.

    Args:
        dgraph (DynamicGraph): The dynamic graph instance.
    """
    global DGRAPH
    DGRAPH = dgraph


def add_edges(source_vertices: np.ndarray, target_vertices: np.ndarray,
              timestamps: np.ndarray, eids: np.ndarray):
    """
    Add edges to the dynamic graph.

    Args:
        source_vertices (np.ndarray): The source vertices of the edges.
        target_vertices (np.ndarray): The target vertices of the edges.
        timestamps (np.ndarray): The timestamps of the edges.
        eids (np.ndarray): The edge IDs of the edges.
    """
    dgraph = get_dgraph()
    logging.debug("Rank %d: Adding %d edges.",
                  torch.distributed.get_rank(), len(source_vertices))
    dgraph.add_edges(source_vertices, target_vertices, timestamps, eids)


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
    dgraph = get_dgraph()
    return dgraph.out_degree(vertex)


def get_temporal_neighbors(vertex: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the temporal neighbors of a vertex at a given timestamp.

    Args:
        vertex (int): The vertex.

    Returns:
        np.ndarray: The temporal neighbors of the vertex.
    """
    dgraph = get_dgraph()
    return dgraph.get_temporal_neighbors(vertex)
