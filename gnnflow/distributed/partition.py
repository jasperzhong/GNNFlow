from typing import List, NamedTuple

import numpy as np
import torch
import math


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

        self._max_node = 0
        # NID -> partition ID, maximum 128 partitions
        self._partition_table = torch.empty(self._max_node, dtype=torch.int8)

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
            if self._max_node == 0:
                self._partition_table[:] = self.UNASSIGNED
            else:
                self._partition_table[self._max_node + 1:] = self.UNASSIGNED
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
                # partitions[i].src_nodes = torch.cat(
                #     [partitions[i].src_nodes, src_nodes[unassigned_mask][mask]])
                # partitions[i].dst_nodes = torch.cat(
                #     [partitions[i].dst_nodes, dst_nodes[unassigned_mask][mask]])
                # partitions[i].timestamps = torch.cat(
                #     [partitions[i].timestamps, timestamps[unassigned_mask][mask]])
                # partitions[i].eids = torch.cat(
                #     [partitions[i].eids, eids[unassigned_mask][mask]])

                partitions[i] = partitions[i]._replace(
                    src_nodes=torch.cat([partitions[i].src_nodes, src_nodes[unassigned_mask][mask]]),
                    dst_nodes=torch.cat([partitions[i].dst_nodes, dst_nodes[unassigned_mask][mask]]),
                    timestamps=torch.cat([partitions[i].timestamps, timestamps[unassigned_mask][mask]]),
                    eids=torch.cat([partitions[i].eids, eids[unassigned_mask][mask]])
                )

                # assign to src node partition
                self._partition_table[src_nodes[unassigned_mask][mask]] = i
                # mask in global edge set
                mask_global = self._partition_table[dst_nodes] == i

                # update unassigned mask
                unassigned_mask = unassigned_mask & ~mask_global

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
        split_idx = torch.split(sorted_idx, tuple(counts.tolist()))
        dst_nodes_list = [dst_nodes[idx] for idx in split_idx]
        timestamps_list = [timestamps[idx] for idx in split_idx]
        eids_list = [eids[idx] for idx in split_idx]

        # partition for each src_node
        partition_table = self._do_partition_for_unseen_nodes_impl(
            unique_src_nodes, dst_nodes_list, timestamps_list, eids_list)

        # restore partition table to the original src_nodes's size
        partition_table = partition_table[inverse_idx]
        partition_table = partition_table.gather(0, sorted_idx.argsort(0))

        return partition_table

    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class HashPartitioner(Partitioner):
    """
    Hash-based partitioner.

    It assigns the source vertex to a partition by the hash value of the vertex ID.
    """

    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        partition_table = unique_src_nodes.clone().detach()
        partition_table.apply_(lambda x: hash(str(x)) % self._num_partitions)
        return partition_table.to(torch.int8)


class OnePartitioner(Partitioner):
    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        partition_table = torch.zeros(len(unique_src_nodes))
        return partition_table.to(torch.int8)


class RoundRobinPartitioner(Partitioner):
    """
    Round-robin partitioning.

    It assigns the source vertex to a partition by the round-robin algorithm.
    """

    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.arange(unique_src_nodes.shape[0]) % self._num_partitions


class LeastLoadedPartitioner(Partitioner):
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


class LeastLoadedPartitionerByEdgeCount(LeastLoadedPartitioner):
    """
    Least-loaded edges partitioner by edge count.

    It assigns the source vertex to a partition with the least number of edges.
    """

    def _compute_metric(self, src_node: int, dst_nodes: torch.Tensor,
                        timestamps: torch.Tensor, eids: torch.Tensor) -> float:
        return len(dst_nodes)


class LeastLoadedPartitionerByTimestampSum(LeastLoadedPartitioner):
    """
    Least-loaded edges partitioner by timestamp sum.

    It assigns the source vertex to a partition with the least sum of timestamps.
    """

    def _compute_metric(self, src_node: int, dst_nodes: torch.Tensor,
                        timestamps: torch.Tensor, eids: torch.Tensor) -> float:
        return timestamps.sum().item()


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

    def _compute_metric(self, src_node: int, dst_nodes: torch.Tensor,
                        timestamps: torch.Tensor, eids: torch.Tensor) -> float:
        count = len(dst_nodes)
        average = self._metrics[src_node]
        self._metrics[src_node] += (timestamps.sum().item() -
                                    average * count) / (self._num_edges[src_node] + count)
        self._num_edges[src_node] += count
        return count


# SOTA Partitoner
class LDGPartitioner(Partitioner):
    """
    Linear Deterministic Greedy (LDG) Partiton Algorithm
    paper: http://www.vldb.org/pvldb/vol11/p1590-abbas.pdf
    """

    def __init__(self, num_partitions: int, assign_with_dst_node: bool = False):
        super().__init__(num_partitions, assign_with_dst_node)

        # key: NID -> value: List[num_partitions]
        self._neighbor_memory = {}
        # ideal partition capacity
        self._partition_capacity = 0
        # edges partitioned
        self._edges_partitioned = 0

    # LDG Partition
    def partition(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                  timestamps: torch.Tensor, eids: torch.Tensor) -> List[Partition]:
        # resize the partition table if necessary
        max_node = int(torch.max(torch.max(src_nodes), torch.max(dst_nodes)))
        if max_node > self._max_node:
            self._partition_table.resize_(max_node + 1)
            if self._max_node == 0:
                self._partition_table[:] = self.UNASSIGNED
            else:
                self._partition_table[self._max_node + 1:] = self.UNASSIGNED
            self._max_node = max_node

        # update edges partitioned
        self._edges_partitioned = self._edges_partitioned + len(src_nodes)
        # update the capacity
        upsilon = 1.1
        self._partition_capacity = (max_node * upsilon) / self._num_partitions

        # dispatch edges to already assigned source nodes
        partitions = []
        for i in range(self._num_partitions):
            mask = self._partition_table[src_nodes] == i

            # enable memory
            for src_id, dst_id in zip(src_nodes[mask], dst_nodes[mask]):
                if int(dst_id) in self._neighbor_memory.keys():
                    self._neighbor_memory[int(dst_id)][i].add(int(src_id))
                else:
                    self._neighbor_memory[int(dst_id)] = [set() for i in range(self._num_partitions)]
                    self._neighbor_memory[int(dst_id)][i].add(int(src_id))

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

            partition_score.append(neighbour_in_partition_size - 0.7 * alpha * gamma * (partition_size ** (gamma - 1)))

        partition_score = np.array(partition_score)

        return np.random.choice(np.where(partition_score == partition_score.max())[0])
        # return np.argmax(partition_score)

    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        partition_table = torch.zeros(len(unique_src_nodes), dtype=torch.int8)
        for i in range(len(unique_src_nodes)):
            pid = self.LDG(int(unique_src_nodes[i]))
            partition_table[i] = pid
            self._partition_table[int(unique_src_nodes[i])] = pid

            for dst_nid in dst_nodes_list[i]:
                # update memory table
                if int(dst_nid) in self._neighbor_memory.keys():
                    self._neighbor_memory[int(dst_nid)][pid].add(int(unique_src_nodes[i]))
                else:
                    self._neighbor_memory[int(dst_nid)] = [set() for i in range(self._num_partitions)]
                    self._neighbor_memory[int(dst_nid)][pid].add(int(unique_src_nodes[i]))

        return partition_table


# Own designed partitioner
class IncrPartitioner(Partitioner):
    """
    Partitioner With Incremental Time-Evolving Graph
    """

    def __init__(self, num_partitions: int, assign_with_dst_node: bool = False):
        super().__init__(num_partitions, assign_with_dst_node)

        # key: NID -> value: List[num_partitions]
        self._neighbor_memory = {}
        # ideal partition capacity
        self._partition_capacity = 0
        # edges partitioned
        self._edges_partitioned = 0

        # pid -> [num_edges, time_sum]
        self._partition_time_map = {}

        # init
        for i in range(num_partitions):
            self._partition_time_map[i] = [0, 0.0]

    # partition
    def partition(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                  timestamps: torch.Tensor, eids: torch.Tensor) -> List[Partition]:
        # resize the partition table if necessary
        max_node = int(torch.max(torch.max(src_nodes), torch.max(dst_nodes)))
        if max_node > self._max_node:
            self._partition_table.resize_(max_node + 1)
            if self._max_node == 0:
                self._partition_table[:] = self.UNASSIGNED
            else:
                self._partition_table[self._max_node + 1:] = self.UNASSIGNED
            self._max_node = max_node

        # update edges partitioned
        self._edges_partitioned = self._edges_partitioned + len(src_nodes)
        # update the capacity
        upsilon = 1.1
        self._partition_capacity = (max_node * upsilon) / self._num_partitions

        # dispatch edges to already assigned source nodes
        partitions = []
        for i in range(self._num_partitions):
            mask = self._partition_table[src_nodes] == i

            # enable memory
            for src_id, dst_id, tstmp in zip(src_nodes[mask], dst_nodes[mask], timestamps[mask]):
                if int(dst_id) in self._neighbor_memory.keys():
                    self._neighbor_memory[int(dst_id)][i].add(int(src_id))

                    # add the timestamp
                    self._partition_time_map[i][0] += 1
                    self._partition_time_map[i][1] += tstmp

                else:
                    self._neighbor_memory[int(dst_id)] = [set() for i in range(self._num_partitions)]
                    self._neighbor_memory[int(dst_id)][i].add(int(src_id))

                    # add the timestamp
                    self._partition_time_map[i][0] += 1
                    self._partition_time_map[i][1] += tstmp

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

    def Incr(self, vid: int):
        partition_score = []

        # hyper parameter
        alpha = (self._num_partitions ** 0.5) * self._edges_partitioned / (self._max_node ** 1.5)
        gamma = 1.5

        for i in range(self._num_partitions):
            partition_size = self._partition_table.tolist().count(i)

            # if partition_size >= self._partition_capacity:
            #     partition_score.append(-2147483647)
            #     continue

            neighbour_in_partition_size = 0
            if vid in self._neighbor_memory.keys():
                neighbour_in_partition_size = len(self._neighbor_memory[vid][i])

            # time factor log(time_avg + 1)
            timefctr = 0.0
            if self._partition_time_map[i][0] != 0:
                timefctr = 0.04 * -1.0 * math.log(1 + self._partition_time_map[i][1] / self._partition_time_map[i][0])

            partition_score.append(timefctr + neighbour_in_partition_size
                                   - 0.8 * alpha * gamma * (partition_size ** (gamma - 1)))

        partition_score = np.array(partition_score)

        return np.random.choice(np.where(partition_score == partition_score.max())[0])

    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        partition_table = torch.zeros(len(unique_src_nodes), dtype=torch.int8)
        for i in range(len(unique_src_nodes)):
            pid = self.Incr(int(unique_src_nodes[i]))
            partition_table[i] = pid
            self._partition_table[int(unique_src_nodes[i])] = pid

            # update time factor
            for timest in timestamps_list[i]:
                self._partition_time_map[pid][0] += 1
                self._partition_time_map[pid][1] += timest

            for dst_nid in dst_nodes_list[i]:
                # update memory table
                if int(dst_nid) in self._neighbor_memory.keys():
                    self._neighbor_memory[int(dst_nid)][pid].add(int(unique_src_nodes[i]))

                else:
                    self._neighbor_memory[int(dst_nid)] = [set() for i in range(self._num_partitions)]
                    self._neighbor_memory[int(dst_nid)][pid].add(int(unique_src_nodes[i]))

        return partition_table


# SOTA Partitoner
class LDGLightPartitioner(Partitioner):
    """
    Linear Deterministic Greedy (LDG) Partiton Algorithm
    paper: http://www.vldb.org/pvldb/vol11/p1590-abbas.pdf
    """

    def __init__(self, num_partitions: int, assign_with_dst_node: bool = False):
        super().__init__(num_partitions, assign_with_dst_node)

        # key: NID -> value: List[num_partitions]
        self._neighbor_memory = torch.zeros(num_partitions, 30000)

        self._edges_partitioned_num_list = torch.zeros(num_partitions, dtype=torch.int32)

        # ideal partition capacity
        self._partition_capacity = 0
        # edges partitioned
        self._edges_partitioned = 0

    # LDG Partition
    def partition(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                  timestamps: torch.Tensor, eids: torch.Tensor) -> List[Partition]:
        # resize the partition table if necessary
        max_node = int(torch.max(torch.max(src_nodes), torch.max(dst_nodes)))
        if max_node > self._max_node:
            self._partition_table.resize_(max_node + 1)
            if self._max_node == 0:
                self._partition_table[:] = self.UNASSIGNED
            else:
                self._partition_table[self._max_node + 1:] = self.UNASSIGNED
            self._max_node = max_node

        # update edges partitioned
        self._edges_partitioned = self._edges_partitioned + len(src_nodes)
        # update the capacity
        upsilon = 1.1
        self._partition_capacity = (max_node * upsilon) / self._num_partitions

        # dispatch edges to already assigned source nodes
        partitions = []
        for i in range(self._num_partitions):
            mask = self._partition_table[src_nodes] == i

            # enable memory
            self._neighbor_memory[i][dst_nodes[mask]] = self._neighbor_memory[i][dst_nodes[mask]] + 1

            self._edges_partitioned_num_list[i] += len(src_nodes[mask])

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

    def LDGLight(self, vid: int):
        partition_score = []

        # hyper parameter
        alpha = (self._num_partitions ** 0.5) * self._edges_partitioned / (self._max_node ** 1.5)
        gamma = 1.5

        for i in range(self._num_partitions):
            partition_size = self._partition_table.tolist().count(i)

            # if partition_size >= self._partition_capacity:
            #     partition_score.append(-2147483647)
            #     continue

            if self._edges_partitioned_num_list[i] > 1.50 * (self._edges_partitioned / self._num_partitions):
                partition_score.append(-2147483646)
                continue

            neighbour_in_partition_size = self._neighbor_memory[i][vid]

            partition_score.append(neighbour_in_partition_size - 0.8 * alpha * gamma * (partition_size ** (gamma - 1)))

        partition_score = np.array(partition_score)

        return np.random.choice(np.where(partition_score == partition_score.max())[0])
        # return np.argmax(partition_score)

    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        partition_table = torch.zeros(len(unique_src_nodes), dtype=torch.int8)
        for i in range(len(unique_src_nodes)):
            pid = self.LDGLight(int(unique_src_nodes[i]))
            partition_table[i] = pid
            self._partition_table[int(unique_src_nodes[i])] = pid

            self._neighbor_memory[pid][dst_nodes_list[i]] = self._neighbor_memory[pid][dst_nodes_list[i]] + 1

            self._edges_partitioned_num_list[pid] += len(dst_nodes_list[i])

        return partition_table


# Own designed partitioner
class IncrLightPartitioner(Partitioner):
    """
    Partitioner With Incremental Time-Evolving Graph
    """

    def __init__(self, num_partitions: int, assign_with_dst_node: bool = False):
        super().__init__(num_partitions, assign_with_dst_node)

        # key: NID -> value: List[num_partitions]
        self._neighbor_memory = torch.zeros(num_partitions, 30000)
        # ideal partition capacity
        self._partition_capacity = 0
        # edges partitioned
        self._edges_partitioned = 0

        # pid -> [num_edges, time_sum]
        self._partition_time_map = {}

        # init
        for i in range(num_partitions):
            self._partition_time_map[i] = [0, 0.0]

    # partition
    def partition(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                  timestamps: torch.Tensor, eids: torch.Tensor) -> List[Partition]:
        # resize the partition table if necessary
        max_node = int(torch.max(torch.max(src_nodes), torch.max(dst_nodes)))
        if max_node > self._max_node:
            self._partition_table.resize_(max_node + 1)
            if self._max_node == 0:
                self._partition_table[:] = self.UNASSIGNED
            else:
                self._partition_table[self._max_node + 1:] = self.UNASSIGNED
            self._max_node = max_node

        # update edges partitioned
        self._edges_partitioned = self._edges_partitioned + len(src_nodes)
        # update the capacity
        upsilon = 1.1
        self._partition_capacity = (max_node * upsilon) / self._num_partitions

        # dispatch edges to already assigned source nodes
        partitions = []
        for i in range(self._num_partitions):
            mask = self._partition_table[src_nodes] == i

            # enable memory
            self._partition_time_map[i][0] += len(timestamps[mask])
            self._partition_time_map[i][1] += torch.sum(timestamps[mask]).item()

            # enable memory
            self._neighbor_memory[i][dst_nodes[mask]] = self._neighbor_memory[i][dst_nodes[mask]] + 1

            partitions.append(Partition(
                src_nodes[mask], dst_nodes[mask], timestamps[mask], eids[mask]))

        # partition the edges for the unseen source nodes
        unassigned_mask = self._partition_table[src_nodes] == self.UNASSIGNED

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

    def IncrLight(self, vid: int):
        partition_score = []

        # hyper parameter
        alpha = (self._num_partitions ** 0.5) * self._edges_partitioned / (self._max_node ** 1.5)
        gamma = 1.5

        for i in range(self._num_partitions):
            partition_size = self._partition_table.tolist().count(i)

            if partition_size >= self._partition_capacity:
                partition_score.append(-2147483647)
                continue

            neighbour_in_partition_size = self._neighbor_memory[i][vid]

            # time factor log(time_avg + 1)
            timefctr = 0.0
            if self._partition_time_map[i][0] != 0:
                timefctr = 0.04 * -1.5 * math.log(1 + self._partition_time_map[i][1] / self._partition_time_map[i][0])

            partition_score.append(timefctr + neighbour_in_partition_size
                                   - 0.6 * alpha * gamma * (partition_size ** (gamma - 1)))

        partition_score = np.array(partition_score)

        return np.random.choice(np.where(partition_score == partition_score.max())[0])

    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        partition_table = torch.zeros(len(unique_src_nodes), dtype=torch.int8)
        for i in range(len(unique_src_nodes)):
            pid = self.IncrLight(int(unique_src_nodes[i]))
            partition_table[i] = pid
            self._partition_table[int(unique_src_nodes[i])] = pid

            # update time factor
            self._partition_time_map[pid][0] += len(timestamps_list[i])
            self._partition_time_map[pid][1] += torch.sum(timestamps_list[i]).item()

            self._neighbor_memory[pid][dst_nodes_list[i]] = self._neighbor_memory[pid][dst_nodes_list[i]] + 1

        return partition_table


# SOTA Partitoner
# Own designed partitioner
class Suc_Partitioner(Partitioner):
    """
    Partitioner With Incremental Time-Evolving Graph
    """

    def __init__(self, num_partitions: int, assign_with_dst_node: bool = False):
        super().__init__(num_partitions, assign_with_dst_node)

        # key: NID -> value: List[num_partitions]
        self._neighbor_memory = torch.zeros(num_partitions, 30000)
        # ideal partition capacity
        self._partition_capacity = 0
        # edges partitioned
        self._edges_partitioned = 0

        # pid -> [num_edges, time_sum]
        self._partition_time_map = {}

        # init
        for i in range(num_partitions):
            self._partition_time_map[i] = [0, 0.0]

    # partition
    def partition(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                  timestamps: torch.Tensor, eids: torch.Tensor) -> List[Partition]:
        # resize the partition table if necessary
        max_node = int(torch.max(torch.max(src_nodes), torch.max(dst_nodes)))
        if max_node > self._max_node:
            self._partition_table.resize_(max_node + 1)
            if self._max_node == 0:
                self._partition_table[:] = self.UNASSIGNED
            else:
                self._partition_table[self._max_node + 1:] = self.UNASSIGNED
            self._max_node = max_node

        # update edges partitioned
        self._edges_partitioned = self._edges_partitioned + len(src_nodes)
        # update the capacity
        upsilon = 1.1
        self._partition_capacity = (max_node * upsilon) / self._num_partitions

        # dispatch edges to already assigned source nodes
        partitions = []
        for i in range(self._num_partitions):
            mask = self._partition_table[src_nodes] == i

            # enable memory
            self._partition_time_map[i][0] += len(timestamps[mask])
            self._partition_time_map[i][1] += torch.sum(timestamps[mask]).item()

            # enable memory
            self._neighbor_memory[i][dst_nodes[mask]] = self._neighbor_memory[i][dst_nodes[mask]] + 1

            partitions.append(Partition(
                src_nodes[mask], dst_nodes[mask], timestamps[mask], eids[mask]))

        # partition the edges for the unseen source nodes
        unassigned_mask = self._partition_table[src_nodes] == self.UNASSIGNED

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

    def Suc(self, vid: int):
        partition_score = []

        # hyper parameter
        alpha = (self._num_partitions ** 0.5) * self._edges_partitioned / (self._max_node ** 1.5)
        gamma = 1.5

        for i in range(self._num_partitions):
            partition_size = self._partition_table.tolist().count(i)

            if partition_size >= self._partition_capacity:
                partition_score.append(-2147483647)
                continue

            neighbour_in_partition_size = self._neighbor_memory[i][vid]

            # time factor log(time_avg + 1)
            timefctr = 0.0
            tot_time_avg = 0.0
            for j in range(self._num_partitions):
                if self._partition_time_map[j][1] == 0:
                    continue
                tot_time_avg += (self._partition_time_map[j][1] / self._partition_time_map[j][0])
            tot_time_avg /= self._num_partitions

            pt_time_avg = 0.0
            if self._partition_time_map[i][1] != 0:
                pt_time_avg = self._partition_time_map[i][1] / self._partition_time_map[i][0]

            if tot_time_avg != 0:
                timefctr = pt_time_avg / tot_time_avg



            partition_score.append(-1.0 * timefctr + neighbour_in_partition_size
                                   - 0.6 * alpha * gamma * (partition_size ** (gamma - 1)))

        partition_score = np.array(partition_score)

        return np.random.choice(np.where(partition_score == partition_score.max())[0])

    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        partition_table = torch.zeros(len(unique_src_nodes), dtype=torch.int8)
        for i in range(len(unique_src_nodes)):
            pid = self.Suc(int(unique_src_nodes[i]))
            partition_table[i] = pid
            self._partition_table[int(unique_src_nodes[i])] = pid

            # update time factor
            self._partition_time_map[pid][0] += len(timestamps_list[i])
            self._partition_time_map[pid][1] += torch.sum(timestamps_list[i]).item()

            self._neighbor_memory[pid][dst_nodes_list[i]] = self._neighbor_memory[pid][dst_nodes_list[i]] + 1

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
    elif partition_strategy == "incr":
        return IncrPartitioner(num_partitions, assign_with_dst_node)
    elif partition_strategy == "one":
        return OnePartitioner(num_partitions, assign_with_dst_node)
    elif partition_strategy == "ldglight":
        return LDGLightPartitioner(num_partitions, assign_with_dst_node)
    elif partition_strategy == "incrlight":
        return IncrLightPartitioner(num_partitions, assign_with_dst_node)
    elif partition_strategy == "suc":
        return Suc_Partitioner(num_partitions, assign_with_dst_node)
    else:
        raise ValueError("Invalid partition strategy: %s" % partition_strategy)
