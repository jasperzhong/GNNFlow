from copy import deepcopy
from typing import List, NamedTuple

import numpy as np
import torch

from gnnflow.utils import load_partition_table


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

    def __init__(self, num_partitions: int, local_world_size: int, dataset_name: str, assign_with_dst_node: bool = False):
        """
        Initialize the partitioner.

        Args:
            num_partitions (int): The number of partitions.
            local_world_size (int): The number of processes in the local world.
            dataset_name (str): the Name of the dataset
            assign_with_dst_node (bool): Whether to assign the edges to the partition of the
                asigned destination node. Default: False.
        """
        self._dataset_name = dataset_name

        self._num_partitions = num_partitions
        self._local_world_size = local_world_size
        self._assign_with_dst_node = assign_with_dst_node

        self._max_node = 0
        # NID -> partition ID, maximum 128 partitions
        self._partition_table = torch.empty(self._max_node, dtype=torch.int8)

        pt = load_partition_table(self._dataset_name)
        if pt is not None:
            self._partition_table = pt

    def get_num_partitions(self) -> int:
        """
        Get the number of partitions.

        Returns:
            int: The number of partitions.
        """
        return self._num_partitions

    def partition(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                  timestamps: torch.Tensor, eids: torch.Tensor,
                  return_evenly_dataset: bool = False):
        """
        Partition the dataset into multiple partitions.

        Args:
            src_nodes (torch.Tensor): The source nodes of the edges.
            dst_nodes (torch.Tensor): The destination nodes of the edges.
            timestamps (torch.Tensor): The timestamps of the edges.
            eids (torch.Tensor): The edge IDs of the edges.
            return_evenly_dataset (bool): Whether to return the evenly partitioned dataset.
        Returns:
            A list of partitions
            A evenly partitioned dataset if return_evenly_dataset is True

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

        partitions = []

        unassigned_mask = self._partition_table[src_nodes] == self.UNASSIGNED

        if self._assign_with_dst_node:
            # assign the edges to the partition of the assined destination node

            # group by src_nodes
            src_nodes_unassigned = src_nodes[unassigned_mask].clone()
            dst_nodes_unassigned = dst_nodes[unassigned_mask].clone()
            timestamps_unassigned = timestamps[unassigned_mask].clone()
            eids_unassigned = eids[unassigned_mask].clone()

            sorted_idx = torch.argsort(src_nodes_unassigned)
            unique_src_nodes, inverse_idx, counts = torch.unique(
                src_nodes_unassigned[sorted_idx], sorted=False, return_inverse=True, return_counts=True)
            split_idx = torch.split(sorted_idx, tuple(counts.tolist()))

            dst_nodes_list = [dst_nodes_unassigned[idx] for idx in split_idx]
            timestamps_list = [timestamps_unassigned[idx] for idx in split_idx]
            eids_list = [eids_unassigned[idx] for idx in split_idx]

            for i in range(len(unique_src_nodes)):
                dst_partition_list = self._partition_table[dst_nodes_list[i]]
                geq_zero_pt = dst_partition_list >= 0
                dst_partition_list = dst_partition_list[geq_zero_pt]

                mode_pt = -1
                if len(dst_partition_list) == 0:
                    # new edge, use user selected logic to partition
                    mode_pt = -1
                else:
                    mode_pt = torch.mode(dst_partition_list).values.item()

                if mode_pt == -1:
                    continue

                self._partition_table[unique_src_nodes[i]] = mode_pt

            # update: partition the edges for the unseen source nodes after assign_with_dst_nodes
            unassigned_mask = self._partition_table[src_nodes] == self.UNASSIGNED

        # dispatch edges to already assigned source nodes
        for i in range(self._num_partitions):
            mask = self._partition_table[src_nodes] == i

            partitions.append(Partition(
                src_nodes[mask], dst_nodes[mask], timestamps[mask], eids[mask]))

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

        evenly_partitioned_dataset = None
        if return_evenly_dataset:
            # make the partitions evenly
            evenly_partitioned_dataset = self._make_partitions_evenly(
                partitions)

        return partitions, evenly_partitioned_dataset

    def _make_partitions_evenly(self, partitions: List[Partition]):
        """
        Make the partitions evenly.

        Args:
            partitions (List[Partition]): The partitions.

        Returns:
            A evenly partitioned dataset.
        """
        # average number of edges in each partition
        avg_num_edges = sum([len(p.src_nodes) for p in partitions]
                            ) // self._num_partitions

        # sort the partitions by the number of edges use argsort
        sorted_idx = torch.argsort(
            torch.tensor([len(p.src_nodes) for p in partitions])).tolist()
        sorted_partitions = [deepcopy(partitions[i]) for i in sorted_idx]

        # move addtional edges from the last partition to the next partition
        for i in reversed(range(1, self._num_partitions)):
            sorted_partitions[i - 1] = Partition(
                torch.cat([sorted_partitions[i - 1].src_nodes,
                           sorted_partitions[i].src_nodes[avg_num_edges:]]),
                torch.cat([sorted_partitions[i - 1].dst_nodes,
                           sorted_partitions[i].dst_nodes[avg_num_edges:]]),
                torch.cat([sorted_partitions[i - 1].timestamps,
                           sorted_partitions[i].timestamps[avg_num_edges:]]),
                torch.cat([sorted_partitions[i - 1].eids,
                           sorted_partitions[i].eids[avg_num_edges:]]))

            sorted_partitions[i] = Partition(
                sorted_partitions[i].src_nodes[:avg_num_edges],
                sorted_partitions[i].dst_nodes[:avg_num_edges],
                sorted_partitions[i].timestamps[:avg_num_edges],
                sorted_partitions[i].eids[:avg_num_edges])

        sorted_partitions[0] = Partition(
            sorted_partitions[0].src_nodes[:avg_num_edges],
            sorted_partitions[0].dst_nodes[:avg_num_edges],
            sorted_partitions[0].timestamps[:avg_num_edges],
            sorted_partitions[0].eids[:avg_num_edges])

        # check the number of edges in each partition
        for i in range(self._num_partitions):
            assert len(sorted_partitions[i].src_nodes) == avg_num_edges, "The \
                    number of edges in partition {} is {}, but the average \
                    number of edges is {}".format(i, len(sorted_partitions[i].src_nodes), avg_num_edges)

        # restore the order of partitions
        restored_partitions = [None] * self._num_partitions
        for i in range(self._num_partitions):
            restored_partitions[sorted_idx[i]] = sorted_partitions[i]

        # for each partition, divide the edges into evenly sized chunks to multiple workers (local_world_size) interleavely
        evenly_partitioned_dataset = []
        for i in range(self._num_partitions):
            num_edges_per_partition = len(restored_partitions[i].src_nodes)
            # discard the last few edges if the number of edges is not divisible by local_world_size
            num_edges_per_partition = num_edges_per_partition - (
                num_edges_per_partition % self._local_world_size)

            src_nodes = restored_partitions[i].src_nodes[:num_edges_per_partition]
            dst_nodes = restored_partitions[i].dst_nodes[:num_edges_per_partition]
            timestamps = restored_partitions[i].timestamps[:num_edges_per_partition]
            eids = restored_partitions[i].eids[:num_edges_per_partition]

            partitioned_dataset = []
            for j in range(self._local_world_size):
                partitioned_dataset.append(Partition(
                    src_nodes[j::self._local_world_size],
                    dst_nodes[j::self._local_world_size],
                    timestamps[j::self._local_world_size],
                    eids[j::self._local_world_size]))

            evenly_partitioned_dataset.append(partitioned_dataset)

        # check the number of edges in each worker are the same
        for i in range(self._num_partitions):
            for j in range(self._local_world_size):
                assert len(evenly_partitioned_dataset[i][j].src_nodes) == len(
                    evenly_partitioned_dataset[0][0].src_nodes), "The number of \
                            edges in partition {} worker {} is {}, but the number \
                            of edges in partition 0 worker 0 is {}".format(
                    i, j, len(evenly_partitioned_dataset[i][j].src_nodes),
                    len(evenly_partitioned_dataset[0][0].src_nodes))

        return evenly_partitioned_dataset

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
            src_nodes[sorted_idx], sorted=True, return_inverse=True, return_counts=True)
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

    def __init__(self, num_partitions: int, local_world_size: int, dataset_name: str, assign_with_dst_node: bool = False):
        super().__init__(num_partitions, local_world_size, dataset_name, assign_with_dst_node)
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

    def __init__(self, num_partitions: int, local_world_size: int, dataset_name: str, assign_with_dst_node: bool = False):
        super().__init__(num_partitions, local_world_size, dataset_name, assign_with_dst_node)
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
class FennelPartitioner(Partitioner):
    """
    Fennel - A revised version of Linear Deterministic Greedy (LDG) Partiton Algorithm
    paper: http://www.vldb.org/pvldb/vol11/p1590-abbas.pdf
    """

    def __init__(self, num_partitions: int, local_world_size: int, dataset_name: str, assign_with_dst_node: bool = False, upsilon: float = 1.1, gamma: float = 1.5):
        super().__init__(num_partitions, local_world_size, dataset_name, assign_with_dst_node)

        # ideal partition capacity
        self._partition_capacity = 0
        # edges partitioned
        self._edges_partitioned = 0

        self._upsilon = upsilon
        self._gamma = gamma

    # Fennel Partition
    def partition(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                  timestamps: torch.Tensor, eids: torch.Tensor,
                  return_evenly_dataset: bool = False):

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
        self._partition_capacity = (
            max_node * self._upsilon) / self._num_partitions

        partitions = []

        # partition the edges for the unseen source nodes
        unassigned_mask = self._partition_table[src_nodes] == self.UNASSIGNED

        # dispatch edges to already assigned source nodes
        for i in range(self._num_partitions):
            mask = self._partition_table[src_nodes] == i
            partitions.append(Partition(
                src_nodes[mask], dst_nodes[mask], timestamps[mask], eids[mask]))

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

        evenly_partitioned_dataset = None
        if return_evenly_dataset:
            # make the partitions evenly
            evenly_partitioned_dataset = self._make_partitions_evenly(
                partitions)

        return partitions, evenly_partitioned_dataset

    def Fennel(self, vid: int, dst_nodes: torch.Tensor):
        partition_score = []

        # hyper parameter
        alpha = (self._num_partitions ** 0.5) * \
            self._edges_partitioned / (self._max_node ** 1.5)

        local_partition_table = self._partition_table[dst_nodes]

        for i in range(self._num_partitions):
            partition_size = (self._partition_table == i).sum().item()

            if partition_size >= self._partition_capacity:
                partition_score.append(-1)
                continue

            # calculate the neighbor in partition i
            neighbour_in_partition_size = (
                local_partition_table == i).sum().item()

            partition_score.append(
                neighbour_in_partition_size - alpha * self._gamma * (partition_size ** (self._gamma - 1)))

        partition_score = np.array(partition_score)

        return int(np.argmax(partition_score))

    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        partition_table = torch.zeros(len(unique_src_nodes), dtype=torch.int8)
        for i in range(len(unique_src_nodes)):
            pid = self.Fennel(int(unique_src_nodes[i]), dst_nodes_list[i])
            partition_table[i] = pid
            self._partition_table[int(unique_src_nodes[i])] = pid

        return partition_table


class FennelEdgePartitioner(Partitioner):
    """
    FennelEdge: Our Designed Partition algorithm by using edge in Fennel
    """

    def __init__(self, num_partitions: int, local_world_size: int, dataset_name: str, assign_with_dst_node: bool = False):
        super().__init__(num_partitions, local_world_size, dataset_name, assign_with_dst_node)

        # neighbor_memory (_num_partition * max_node)
        self._out_degree = torch.zeros(self._max_node, dtype=torch.long)
        # edges partitioned
        self._edges_partitioned = 0
        # edges partitioned w.r.t. partitions
        self._edges_partitioned_num_list = torch.zeros(
            num_partitions, dtype=torch.long)

    # Fennel Edge Partition
    def partition(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor,
                  timestamps: torch.Tensor, eids: torch.Tensor,
                  return_evenly_dataset: bool = False):

        # resize the partition table and node's out degree if necessary
        max_node = int(torch.max(torch.max(src_nodes), torch.max(dst_nodes)))
        if max_node > self._max_node:
            self._partition_table.resize_(max_node + 1)
            self._out_degree.resize_(max_node + 1)
            if self._max_node == 0:
                self._partition_table[:] = self.UNASSIGNED
                self._out_degree[:] = 0
            else:
                self._partition_table[self._max_node + 1:] = self.UNASSIGNED
                self._out_degree[self._max_node + 1:] = 0
            self._max_node = max_node

        # update edges partitioned
        self._edges_partitioned = self._edges_partitioned + len(src_nodes)

        partitions = []

        # partition the edges for the unseen source nodes
        unassigned_mask = self._partition_table[src_nodes] == self.UNASSIGNED

        # dispatch edges to already assigned source nodes
        for i in range(self._num_partitions):
            mask = self._partition_table[src_nodes] == i

            # update the out degree
            local_src_nodes_mask = src_nodes[mask].clone()
            unique_src_node_mask_id_list, cnt_seq = torch.unique(
                local_src_nodes_mask, return_counts=True)

            for src_id_idx in range(len(unique_src_node_mask_id_list)):
                self._out_degree[unique_src_node_mask_id_list[src_id_idx]
                                 ] += cnt_seq[src_id_idx]

            # add to partition
            self._edges_partitioned_num_list[i] += len(src_nodes[mask])

            partitions.append(Partition(
                src_nodes[mask], dst_nodes[mask], timestamps[mask], eids[mask]))

        # Use Normal Insertion Method
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

        evenly_partitioned_dataset = None
        if return_evenly_dataset:
            # make the partitions evenly
            evenly_partitioned_dataset = self._make_partitions_evenly(
                partitions)

        return partitions, evenly_partitioned_dataset

    def edge_set_normalize(self, arr, t_min, t_max):
        # explicit function to normalize array
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)
        for i in arr:
            temp = (((i - min(arr)) * diff) / diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr

    def fennelEdge(self, vid: int, dst_nodes: torch.Tensor):
        partition_score = []
        debug_map = {}

        local_partition_table = self._partition_table[dst_nodes]

        # unique to get the neighbor quant
        npt_list, npt_cnt = torch.unique(
            local_partition_table, return_counts=True)
        npt_map = {}
        for i in range(len(npt_list)):
            npt_map[npt_list[i]] = npt_cnt[i]

        # eliminate the influence of -1
        neighbour_in_partition_size_list = []

        for i in range(self._num_partitions):
            if i not in npt_map:
                neighbour_in_partition_size_list.append(0)
                continue

            neighbour_in_partition_size_list.append(npt_map[i])

        load_balance_score = []

        for i in range(self._num_partitions):
            if self._edges_partitioned_num_list[i] + len(dst_nodes) > 1.25 * (self._edges_partitioned / self._num_partitions):
                partition_score.append(-np.inf)
                load_balance_score.append(-10)
                continue

            # calculate the neighbor in partition i
            neighbour_in_partition_size = neighbour_in_partition_size_list[i]

            # calculate neighbor's out degree sum
            neighbour_in_partition_mask = local_partition_table == i
            neighbour_in_partition_id = torch.unique(
                dst_nodes[neighbour_in_partition_mask])

            out_degree_sum = 0
            if len(neighbour_in_partition_id) != 0:
                out_degree_sum = self._out_degree[neighbour_in_partition_id].sum(
                ).item()

            locality_score = neighbour_in_partition_size + out_degree_sum

            # load_balance_score.append(self._edges_partitioned_num_list[i] / self._edges_partitioned)
            load_balance_score.append(self._edges_partitioned_num_list[i])
            partition_score.append(locality_score)

        # load_balance_score = self.edge_set_normalize(load_balance_score, -50, 50)

        for i in range(self._num_partitions):
            # print("locality_score[{}] = {}, load_balance_score[{}] = {}".format(i, partition_score[i], i, load_balance_score[i]))
            partition_score[i] -= load_balance_score[i]

        partition_score = np.array(partition_score)

        return int(np.random.choice(np.where(partition_score == partition_score.max())[0])), debug_map
        # return int(np.argmax(partition_score)), debug_map

    def _do_partition_for_unseen_nodes_impl(self, unique_src_nodes: torch.Tensor,
                                            dst_nodes_list: List[torch.Tensor],
                                            timestamps_list: List[torch.Tensor],
                                            eids_list: List[torch.Tensor]) -> torch.Tensor:
        partition_table = torch.zeros(len(unique_src_nodes), dtype=torch.int8)

        # sort reversely by N(v)
        neighbour_size_list = []
        for i in range(len(dst_nodes_list)):
            neighbour_size_list.append(len(dst_nodes_list[i]))

        argsort_list = np.argsort(neighbour_size_list)
        # argsort_list = np.arange(0, len(neighbour_size_list))
        # argsort_list = argsort_list[::-1]

        for i in range(len(unique_src_nodes)):
            sorted_idx = argsort_list[i]

            pid, debug_map = self.fennelEdge(
                int(unique_src_nodes[sorted_idx]), dst_nodes_list[sorted_idx])
            partition_table[sorted_idx] = pid
            self._partition_table[int(unique_src_nodes[sorted_idx])] = pid
            self._out_degree[unique_src_nodes[sorted_idx]
                             ] += len(dst_nodes_list[sorted_idx])

            # update the edge partition num_list
            self._edges_partitioned_num_list[pid] += len(
                dst_nodes_list[sorted_idx])

        return partition_table


def get_partitioner(partition_strategy: str, num_partitions: int, local_world_size: int, dataset_name: str, assign_with_dst_node: bool = False):
    """
    Get the partitioner.

    Args:
        partition_strategy (str): The partitioning strategy.
        num_partitions (int): The number of partitions to split the dataset into.
        local_world_size (int): The number of processes in the local world.
        assign_with_dst_node (bool): Whether to assign the edges to the partition of the destination node.

    Returns:
        Partitioner: The partitioner.
    """
    if partition_strategy == "hash":
        return HashPartitioner(num_partitions, local_world_size, dataset_name, assign_with_dst_node)
    elif partition_strategy == "roundrobin":
        return RoundRobinPartitioner(
            num_partitions, local_world_size, dataset_name, assign_with_dst_node)
    elif partition_strategy == "edgecount":
        return LeastLoadedPartitionerByEdgeCount(
            num_partitions, local_world_size, dataset_name, assign_with_dst_node)
    elif partition_strategy == "timestampsum":
        return LeastLoadedPartitionerByTimestampSum(
            num_partitions, local_world_size, dataset_name, assign_with_dst_node)
    elif partition_strategy == "timestampavg":
        return LeastLoadedPartitionerByTimestampAvg(
            num_partitions, local_world_size, dataset_name, assign_with_dst_node)
    elif partition_strategy == "fennel":
        return FennelPartitioner(num_partitions, local_world_size, dataset_name, assign_with_dst_node)
    elif partition_strategy == "fennel_edge":
        return FennelEdgePartitioner(num_partitions, local_world_size, dataset_name, assign_with_dst_node)
    else:
        raise ValueError("Invalid partition strategy: %s" % partition_strategy)
