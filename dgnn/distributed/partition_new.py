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


class NewPartitioner:
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

        self._max_node = 0
        # NID -> partition ID, maximum 128 partitions
        self._partition_table = torch.empty(self._max_node, dtype=torch.int8)
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
        if max_node > self._max_node:
            self._partition_table.resize_(max_node + 1)
            self._partition_table[self._max_node+1:] = self.UNASSIGNED
            self._max_node = max_node

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
        # group by src_nodes
        sorted_idx = torch.argsort(src_nodes)
        unique_src_nodes, inverse_idx, counts = torch.unique(
            src_nodes[sorted_idx], sorted=False, return_inverse=True, return_counts=True)
        split_idx = torch.split(sorted_idx, counts.tolist())
        dst_nodes_list = [dst_nodes[c] for c in split_idx]
        timestamps_list = [timestamps[c] for c in split_idx]
        eids_list = [eids[c] for c in split_idx]

        # partition for each src_node
        partition_table = self._do_partition_for_unseen_nodes_impl(
            unique_src_nodes, dst_nodes_list, timestamps_list, eids_list)

        # restore partition table to the original src_nodes's size
        partition_table = partition_table[inverse_idx]
        partition_table = partition_table.gather(0, sorted_idx.argsort(0))

        partition_table2 = self._do_partition_for_unseen_nodes_impl(
            src_nodes, dst_nodes_list, timestamps_list, eids_list)

        assert torch.all(partition_table == partition_table2), "partition_table: {}, partition_table2: {}".format(
            partition_table, partition_table2)
        return partition_table

    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class NewHashPartitioner(NewPartitioner):
    """
    Hash-based partitioner.

    It assigns the source vertex to a partition by the hash value of the vertex ID.
    """

    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        partition_table = unique_src_nodes.clone().detach()
        partition_table.apply_(lambda x: hash(str(int(x))) % self._num_partitions)
        return partition_table.to(torch.int8)


class NewRoundRobinPartitioner(NewPartitioner):
    """
    Round-robin partitioning.

    It assigns the source vertex to a partition by the round-robin algorithm.
    """

    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.arange(unique_src_nodes.shape[0]) % self._num_partitions


class NewLeastLoadedPartitioner(NewPartitioner):
    """
    Least-loaded edges partitioner.

    It assigns the source vertex to a partition by the least-loaded algorithm.
    Different least-loaded algorithms differ in how to compute the load of a partition.
    """

    def __init__(self, num_partitions: int, assign_with_dst_node: bool = False):
        super().__init__(num_partitions, assign_with_dst_node)
        self._metrics = torch.zeros(num_partitions, dtype=torch.float32)

    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        partition_table = torch.zeros(len(unique_src_nodes), dtype=torch.int8)
        for i in range(len(unique_src_nodes)):
            partition_table[i] = torch.argmin(self._metrics)
            self._metrics[partition_table[i]] += self._compute_metric(
                int(unique_src_nodes[i]), dst_nodes_list[i], timestamps_list[i], eids_list[i])
        return partition_table

    def _compute_metric(self, src_node: int, dst_nodes: torch.Tensor,
                        timestamps: torch.Tensor, eids: torch.Tensor) -> float:
        raise NotImplementedError


class NewLeastLoadedPartitionerByEdgeCount(NewLeastLoadedPartitioner):
    """
    Least-loaded edges partitioner by edge count.

    It assigns the source vertex to a partition with the least number of edges.
    """

    def _compute_metric(self, src_node: int, dst_nodes: torch.Tensor,
                        timestamps: torch.Tensor, eids: torch.Tensor) -> float:
        return len(dst_nodes)


class NewLeastLoadedPartitionerByTimestampSum(NewLeastLoadedPartitioner):
    """
    Least-loaded edges partitioner by timestamp sum.

    It assigns the source vertex to a partition with the least sum of timestamps.
    """

    def _compute_metric(self, src_node: int, dst_nodes: torch.Tensor,
                        timestamps: torch.Tensor, eids: torch.Tensor) -> float:
        return timestamps.sum().item()


class NewLeastLoadedPartitionerByTimestampAvg(NewLeastLoadedPartitioner):
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

    def _compute_metric(self, src_node: int, dst_nodes: torch.Tensor,
                        timestamps: torch.Tensor, eids: torch.Tensor) -> float:
        count = len(dst_nodes)
        average = self._metrics[src_node]
        self._metrics[src_node] += (timestamps.sum().item() -
                                    average * count) / (self._num_edges[src_node] + count)
        self._num_edges[src_node] += count
        return count


def get_new_partitioner(partition_strategy: str, num_partitions: int, assign_with_dst_node: bool = False):
    """
    Get the partitioner.

    Args:
        partition_strategy (str): The partitioning strategy.
        num_partitions (int): The number of partitions to split the dataset into.
        assign_with_dst_node (bool): Whether to assign the edges to the partition of the destination node.

    Returns:
        Partitioner: The partitioner.
    """
    # TODO(tianzuo): add a test for existing partitioners in tests/
    if partition_strategy == "hash":
        return NewHashPartitioner(num_partitions, assign_with_dst_node)
    elif partition_strategy == "roundrobin":
        return NewRoundRobinPartitioner(
            num_partitions, assign_with_dst_node)
    elif partition_strategy == "edgecount":
        return NewLeastLoadedPartitionerByEdgeCount(
            num_partitions, assign_with_dst_node)
    elif partition_strategy == "timestampsum":
        return NewLeastLoadedPartitionerByTimestampSum(
            num_partitions, assign_with_dst_node)
    elif partition_strategy == "timestampavg":
        return NewLeastLoadedPartitionerByTimestampAvg(
            num_partitions, assign_with_dst_node)
    # TODO(tianzuo): SOTA partitioners.
    else:
        raise ValueError("Invalid partition strategy: %s" % partition_strategy)