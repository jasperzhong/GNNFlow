from typing import List, Tuple

import numpy as np


class Partitioner:
    """
    Partition the dataset into multiple partitions.
    """

    def __init__(self, num_partition: int):
        self.num_partition = num_partition

    def partition(self, src_nodes: np.ndarray, dst_nodes: np.ndarray,
                  timestamps: np.ndarray, eids: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Partition the dataset into multiple partitions.

        Args:
            src_nodes (np.ndarray): The source nodes of the edges.
            dst_nodes (np.ndarray): The destination nodes of the edges.
            timestamps (np.ndarray): The timestamps of the edges.
            eids (np.ndarray): The edge IDs of the edges.

        Returns:
            A list of partitions.
        """
        raise NotImplementedError


class RoundRobinPartitioner(Partitioner):
    """
    Round-robin partitioning.
    """

    def __init__(self, num_partition: int):
        super().__init__(num_partition)
        self.partition_id = 0

    def partition(self, src_nodes: np.ndarray, dst_nodes: np.ndarray,
                  timestamps: np.ndarray, eids: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Partition the dataset into multiple partitions.

        Args:
            src_nodes (np.ndarray): The source nodes of the edges.
            dst_nodes (np.ndarray): The destination nodes of the edges.
            timestamps (np.ndarray): The timestamps of the edges.
            eids (np.ndarray): The edge IDs of the edges.

        Returns:
            A list of partitions.
        """
        partitions = [[] for _ in range(self.num_partition)]
        for i in range(len(src_nodes)):
            partitions[self.partition_id].append(
                (src_nodes[i], dst_nodes[i], timestamps[i], eids[i]))
            self.partition_id = (self.partition_id + 1) % self.num_partition

        for i in range(self.num_partition):
            partitions[i] = list(zip(*partitions[i]))
            partitions[i] = (np.array(partitions[i][0]), np.array(partitions[i][1]),
                             np.array(partitions[i][2]), np.array(partitions[i][3]))

        return partitions


class SpatialTemporalPartitioner(Partitioner):
    """
    Spatial-temporal partitioning.
    """

    def __init__(self, num_partition: int):
        super().__init__(num_partition)

    def partition(self, src_nodes: np.ndarray, dst_nodes: np.ndarray,
                  timestamps: np.ndarray, eids: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Partition the dataset into multiple partitions.

        Args:
            src_nodes (np.ndarray): The source nodes of the edges.
            dst_nodes (np.ndarray): The destination nodes of the edges.
            timestamps (np.ndarray): The timestamps of the edges.

        Returns:
            A list of partitions.
        """
        raise NotImplementedError


def get_partitioner(partition_strategy: str, num_partition: int) -> Partitioner:
    """
    Get the partitioner.

    Args:
        partition_strategy (str): The partitioning strategy.
        num_partition (int): The number of partitions to split the dataset into.

    Returns:
        Partitioner: The partitioner.
    """
    if partition_strategy == "roundrobin":
        return RoundRobinPartitioner(num_partition)
    elif partition_strategy == "spatialtemporal":
        return SpatialTemporalPartitioner(num_partition)
    else:
        raise ValueError("Invalid partition strategy: %s" % partition_strategy)
