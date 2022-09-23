from typing import List, NamedTuple

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

    def __init__(self, num_partitions: int, assign_with_dst_node: bool = False):
        """
        Initialize the partitioner.

        Args:
            num_partitions (int): The number of partitions.
            assign_with_dst_node (bool): Whether to assign the edges to the partition of the
                asigned destination node. Default: False.
        """
        self._num_partitions = num_partitions
        self._assign_with_dst_node = assign_with_dst_node

        self._num_nodes = 0
        # NID -> partition ID, maximum 128 partitions
        self._partition_table = torch.empty(self._num_nodes, dtype=torch.int8)
        self._partition_table[:] = self.UNASSIGNED

    def get_num_partitions(self) -> int:
        """
        Get the number of partitions.

        Returns:
            int: The number of partitions.
        """
        return self._num_partitions

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
        if max_node >= self._num_nodes:
            self._partition_table.resize_(max_node + 1)
            self._partition_table[self._num_nodes:] = self.UNASSIGNED
            self._num_nodes = max_node + 1

        # dispatch edges to already assigned source nodes
        partitions = []
        for i in range(self._num_partitions):
            mask = self._partition_table[src_nodes] == i
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

    def partition(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                  timestamps: torch.Tensor, eids: torch.Tensor) -> List[Partition]:
        partitions = super().partition(src_nodes, dst_nodes, timestamps, eids)
        self.update_metrics(partitions)
        return partitions

    def update_metrics(self, partitions: List[Partition]):
        """
        Update the metrics of the partitions.

        Args:
            partitions (List[Partition]): The partitions.
        """
        raise NotImplementedError

    def update_metrics_for_one_edge(self, current_partition: int, src_node: int,
                                    dst_node: int, timestamp: float, eid: int):
        """
        Update the metrics of the partition for one edge.

        Args:
            current_partition (int): The current partition of the edge.
            src_node (int): The source node of the edge.
            dst_node (int): The destination node of the edge.
            timestamp (float): The timestamp of the edge.
            eid (int): The edge ID of the edge.
        """
        raise NotImplementedError

    def _do_partition_for_unseen_nodes(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                                       timestamps: torch.Tensor, eids: torch.Tensor) -> torch.Tensor:
        # NB: we update the number of edges for each partition later in the partition() function.
        # Therefore, we make a copy of the number of edges here.
        metrics = self._metrics.clone().detach()

        partition_table = torch.zeros(len(src_nodes), dtype=torch.int8)
        for i in range(len(src_nodes)):
            partition_table[i] = torch.argmin(metrics)
            self.update_metrics_for_one_edge(
                i, int(src_nodes[i]), int(dst_nodes[i]), float(timestamps[i]), int(eids[i]))
        return partition_table


class LeastLoadedPartitionerByEdgeCount(LeastLoadedPartitioner):
    """
    Least-loaded edges partitioner by edge count.

    It assigns the source vertex to a partition with the least number of edges.
    """

    def update_metrics(self, partitions: List[Partition]):
        for i in range(self._num_partitions):
            self._metrics[i] += len(partitions[i].src_nodes)

    def update_metrics_for_one_edge(self, current_partition: int, src_node: int,
                                    dst_node: int, timestamp: float, eid: int) -> float:
        self._metrics[current_partition] += 1


class LeastLoadedPartitionerByTimestampSum(LeastLoadedPartitioner):
    """
    Least-loaded edges partitioner by timestamp sum.

    It assigns the source vertex to a partition with the least sum of timestamps.
    """

    def update_metrics(self, partitions: List[Partition]):
        for i in range(self._num_partitions):
            self._metrics[i] += partitions[i].timestamps.sum().item()

    def update_metrics_for_one_edge(self, current_partition: int, src_node: int,
                                    dst_node: int, timestamp: float, eid: int) -> float:
        self._metrics[current_partition] += timestamp


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

    def update_metrics(self, partitions: List[Partition]):
        for i in range(self._num_partitions):
            k = len(partitions[i].src_nodes)
            self._num_edges[i] += k
            self._metrics[i] += (partitions[i].timestamps.sum().item() -
                                 self._metrics[i] * k) / self._num_edges[i]

    def update_metrics_for_one_edge(self, current_partition: int, src_node: int,
                                    dst_node: int, timestamp: float, eid: int) -> float:
        self._num_edges[current_partition] += 1
        self._metrics[current_partition] += (
            timestamp - self._metrics[current_partition]) / self._num_edges[current_partition]


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
    else:
        raise ValueError("Invalid partition strategy: %s" % partition_strategy)
