from typing import List, NamedTuple

import numpy as np
import torch


class Partition(NamedTuple):
    """
    Partition of the graph.
    """

    src_nodes: torch.Tensor
    dst_nodes: torch.Tensor
    timestamps: torch.Tensor
    eids: torch.Tensor


class Partitioner:
    """
    Partition the dataset into multiple partitions.

    NB: we partition the graph by the vertices, not the edges. Edges are 
    partitioned by their source vertices.
    """
    UNASSIGNED = -1

    def __init__(self, num_partitions: int, assign_with_dst_node: bool = False, enable_neighbour_memory: bool = False):
        """
        Initialize the partitioner.

        Args:
            num_partitions (int): The number of partitions.
            assign_with_dst_node (bool): Whether to assign the edges to the partition of the
                asigned destination node. Default: False.
        """
        self._num_partitions = num_partitions
        self._assign_with_dst_node = assign_with_dst_node

        self._max_node = 0
        # NID -> partition ID, maximum 128 partitions
        self._partition_table = torch.empty(self._max_node, dtype=torch.int8)
        self._partition_table[:] = self.UNASSIGNED

        # key: NID -> value: List[num_partitions]
        self._enable_neighbor_memory = False
        self._neighbor_memory = {}
        # ideal partition capacity
        self._partition_capacity = 0
        # edges partitioned
        self._edges_partitioned = 0

    def get_num_partitions(self) -> int:
        """
        Get the number of partitions.

        Returns:
            int: The number of partitions.
        """
        return self._num_partitions

    def set_edges_partitioned(self, val:int):
        self._edges_partitioned = val
        return

    def get_edges_partitioned(self) -> int:
        return self._edges_partitioned

    def partition(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                  timestamps: torch.Tensor, eids: torch.Tensor) -> List[Partition]:
        """
        Partition the dataset into multiple partitions.

        Args:
            src_nodes (torch.Tensor): The source nodes of the edges.
            dst_nodes (torch.Tensor): The destination nodes of the edges.
            timestamps (torch.Tensor): The timestamps of the edges.
            eids (torch.Tensor): The edge IDs of the edges.

        Returns:
            A list of partitions.
        """
        # resize the partition table if necessary
        max_node = int(torch.max(torch.max(src_nodes), torch.max(dst_nodes)))
        if max_node >= self._max_node:
            self._partition_table.resize_(max_node + 1)
            self._partition_table[self._max_node:] = self.UNASSIGNED
            self._max_node = max_node + 1

        # update edges partitioned
        self._edges_partitioned = self._edges_partitioned + len(src_nodes)

        # TODO: 1.1 is a heuristic setting
        upsilon = 1.1
        self._partition_capacity = (max_node * upsilon) / self._num_partitions
        print('partition capacity C is :{} \n'.format(self._partition_capacity))

        # dispatch edges to already assigned source nodes
        partitions = []
        for i in range(self._num_partitions):
            mask = self._partition_table[src_nodes] == i
            # enable memory
            if self._enable_neighbor_memory:
                for src_id, dst_id in zip(src_nodes[mask], dst_nodes[mask]):
                    if dst_id not in self._neighbor_memory.keys():
                        self._neighbor_memory[dst_id] = [set() for i in range(self._num_partitions)]
                    else:
                        self._neighbor_memory[dst_id][i].add(src_id)
            partitions.append(Partition(
                src_nodes[mask], dst_nodes[mask], timestamps[mask], eids[mask]))

        # partition the edges for the unseen source nodes
        unassigned_mask = self._partition_table[src_nodes] == self.UNASSIGNED

        if self._assign_with_dst_node:
            # assign the edges to the partition of the assined destination node
            for i in range(self._num_partitions):
                mask = self._partition_table[dst_nodes[unassigned_mask]] == i
                partitions[i].src_nodes = torch.cat(
                    [partitions[i].src_nodes, src_nodes[unassigned_mask][mask]])
                partitions[i].dst_nodes = torch.cat(
                    [partitions[i].dst_nodes, dst_nodes[unassigned_mask][mask]])
                partitions[i].timestamps = torch.cat(
                    [partitions[i].timestamps, timestamps[unassigned_mask][mask]])
                partitions[i].eids = torch.cat(
                    [partitions[i].eids, eids[unassigned_mask][mask]])

                # update unassigned mask
                unassigned_mask = unassigned_mask & ~mask

        partition_table_for_unseen_nodes = self._do_partition_for_unseen_nodes(
            src_nodes[unassigned_mask], dst_nodes[unassigned_mask],
            timestamps[unassigned_mask], eids[unassigned_mask])

        assert partition_table_for_unseen_nodes.shape[0] == unassigned_mask.sum(
        )

        # merge the partitions
        for i in range(self._num_partitions):
            mask = partition_table_for_unseen_nodes == i

            # update the partition table
            self._partition_table[src_nodes[unassigned_mask][mask]] = i

            # no need to sort edges here
            partitions[i] = Partition(
                torch.cat([partitions[i].src_nodes,
                          src_nodes[unassigned_mask][mask]]),
                torch.cat([partitions[i].dst_nodes,
                          dst_nodes[unassigned_mask][mask]]),
                torch.cat([partitions[i].timestamps,
                          timestamps[unassigned_mask][mask]]),
                torch.cat([partitions[i].eids, eids[unassigned_mask][mask]]))

        return partitions

    def get_partition_table(self) -> torch.Tensor:
        """
        Get the partition table.

        Returns:
            torch.Tensor: The partition table.
        """
        return self._partition_table

    def _do_partition_for_unseen_nodes(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                                       timestamps: torch.Tensor, eids: torch.Tensor) -> torch.Tensor:
        """
        Partition the edges for the unseen source nodes. 

        Args:
            src_nodes (torch.Tensor): The source nodes of the edges.
            dst_nodes (torch.Tensor): The destination nodes of the edges.
            timestamps (torch.Tensor): The timestamps of the edges.
            eids (torch.Tensor): The edge IDs of the edges.
        Returns:
            partition table (torch.Tensor): The partition table for the unseen source nodes.
        """
        raise NotImplementedError


class HashPartitioner(Partitioner):
    """
    Hash-based partitioner.

    It assigns the source vertex to a partition by the hash value of the vertex ID.
    """

    def _do_partition_for_unseen_nodes(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                                       timestamps: torch.Tensor, eids: torch.Tensor) -> torch.Tensor:
        partition_table = src_nodes.clone().detach()
        partition_table.apply_(lambda x: hash(str(x)) % self._num_partitions)
        return partition_table.to(torch.int8)


class RoundRobinPartitioner(Partitioner):
    """
    Round-robin partitioning.

    It assigns the source vertex to a partition by the round-robin algorithm.
    """

    def _do_partition_for_unseen_nodes(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                                       timestamps: torch.Tensor, eids: torch.Tensor) -> torch.Tensor:
        return (torch.arange(0, len(src_nodes)) % self._num_partitions).to(torch.int8)


class LeastLoadedPartitioner(Partitioner):
    """
    Least-loaded edges partitioner.

    It assigns the source vertex to a partition by the least-loaded algorithm.
    Different least-loaded algorithms differ in how to compute the load of a partition.
    """

    def __init__(self, num_partitions: int, assign_with_dst_node: bool = False):
        super().__init__(num_partitions, assign_with_dst_node)
        self._metrics = torch.zeros(num_partitions, dtype=torch.float32)

    def update_metrics_for_one_edge(self, partition_id: int, src_node: int,
                                    dst_node: int, timestamp: float, eid: int):
        """
        Update the metrics of the partition for one edge.

        Args:
            partition_id (int): The current partition of the edge.
            src_node (int): The source node of the edge.
            dst_node (int): The destination node of the edge.
            timestamp (float): The timestamp of the edge.
            eid (int): The edge ID of the edge.
        """
        raise NotImplementedError

    def _do_partition_for_unseen_nodes(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                                       timestamps: torch.Tensor, eids: torch.Tensor) -> torch.Tensor:
        partition_table = torch.zeros(len(src_nodes), dtype=torch.int8)
        for i in range(len(src_nodes)):
            partition_id = int(torch.argmin(self._metrics).item())
            partition_table[int(src_nodes[i])] = partition_id
            self.update_metrics_for_one_edge(partition_id,
                                             int(src_nodes[i]),
                                             int(dst_nodes[i]),
                                             float(timestamps[i]), int(eids[i]))
        return partition_table


class LeastLoadedPartitionerByEdgeCount(LeastLoadedPartitioner):
    """
    Least-loaded edges partitioner by edge count.

    It assigns the source vertex to a partition with the least number of edges.
    """

    def update_metrics(self, partitions: List[Partition]):
        for i in range(self._num_partitions):
            self._metrics[i] += len(partitions[i].src_nodes)

    def update_metrics_for_one_edge(self, partition_id: int, src_node: int,
                                    dst_node: int, timestamp: float, eid: int) -> float:
        self._metrics[partition_id] += 1


class LeastLoadedPartitionerByTimestampSum(LeastLoadedPartitioner):
    """
    Least-loaded edges partitioner by timestamp sum.

    It assigns the source vertex to a partition with the least sum of timestamps.
    """

    def update_metrics_for_one_edge(self, partition_id: int, src_node: int,
                                    dst_node: int, timestamp: float, eid: int) -> float:
        self._metrics[partition_id] += timestamp


class LeastLoadedPartitionerByTimestampAvg(LeastLoadedPartitioner):
    """
    Least-loaded edges partitioner by timestamp average.

    It assigns the source vertex to a partition with the least average of timestamps.

    average = (average * count + a1 + a2 + ... + ak) / (count + k)
                                <=>
    average += (a1 + a2 + ... + ak - average * k) / (count + k)
    """

    def __init__(self, num_partitions: int, assign_with_dst_node: bool = False):
        super().__init__(num_partitions, assign_with_dst_node)
        self._num_edges = torch.zeros(num_partitions, dtype=torch.int64)

    def update_metrics_for_one_edge(self, partition_id: int, src_node: int,
                                    dst_node: int, timestamp: float, eid: int) -> float:
        self._num_edges[partition_id] += 1
        self._metrics[partition_id] += (
            timestamp - self._metrics[partition_id]) / self._num_edges[partition_id]


# SOTA Partitioner
class LDGPartitioner(Partitioner):
    """
    Linear Deterministic Greedy (LDG) Partiton Algorithm

    """

    def __init__(self, num_partitions: int, assign_with_dst_node: bool = False):
        super().__init__(num_partitions, assign_with_dst_node, True)

    def LDG(self, vid: int):
        partition_score = []

        # hyper parameter
        alpha = (self._num_partitions ** 0.5) * self._edges_partitioned / (self._max_node ** 1.5)
        gamma = 1.5

        for i in range(self._num_partitions):
            partition_size = self._partition_table.tolist().count(i)

            if partition_size >= self._partition_capacity:
                partition_score.append(-2147483647)
                continue

            neighbour_in_partition_size = 0
            if vid in self._neighbor_memory.keys():
                neighbour_in_partition_size = len(self._neighbor_memory[vid][i])

            partition_score.append(neighbour_in_partition_size - alpha * gamma * (partition_size ** (gamma - 1)))

        if 1000 < vid < 2000:
            print(partition_score)

        return np.argmax(partition_score)

    def _do_partition_for_unseen_nodes(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                                       timestamps: torch.Tensor, eids: torch.Tensor) -> torch.Tensor:
        partition_table = torch.zeros(len(src_nodes), dtype=torch.int8)
        for i in range(len(src_nodes)):
            pid = self.LDG(int(src_nodes[i]))
            partition_table[int(src_nodes[i])] = pid

            # update partition_table simultaneously
            self._partition_table[int(src_nodes[i])] = pid

            # update memory table
            if int(dst_nodes[i]) in self._neighbor_memory.keys():
                self._neighbor_memory[int(dst_nodes[i])][pid].add(int(src_nodes[i]))
            else:
                self._neighbor_memory[int(dst_nodes[i])] = [set() for i in range(self._num_partitions)]
                self._neighbor_memory[int(dst_nodes[i])][pid].add(int(src_nodes[i]))

        return partition_table


def get_partitioner(partition_strategy: str, num_partitions: int, assign_with_dst_node: bool = False):
    """
    Get the partitioner.

    Args:
        partition_strategy (str): The partitioning strategy.
        num_partitions (int): The number of partitions to split the dataset into.
        assign_with_dst_node (bool): Whether to assign the edges to the partition of the destination node.

    Returns:
        Partitioner: The partitioner.
    """
    if partition_strategy == "hash":
        return HashPartitioner(num_partitions, assign_with_dst_node)
    elif partition_strategy == "roundrobin":
        return RoundRobinPartitioner(
            num_partitions, assign_with_dst_node)
    elif partition_strategy == "edgecount":
        return LeastLoadedPartitionerByEdgeCount(
            num_partitions, assign_with_dst_node)
    elif partition_strategy == "timestampsum":
        return LeastLoadedPartitionerByTimestampSum(
            num_partitions, assign_with_dst_node)
    elif partition_strategy == "timestampavg":
        return LeastLoadedPartitionerByTimestampAvg(
            num_partitions, assign_with_dst_node)
    elif partition_strategy == "ldg":
        return LDGPartitioner(num_partitions, assign_with_dst_node)
    else:
        raise ValueError("Invalid partition strategy: %s" % partition_strategy)
