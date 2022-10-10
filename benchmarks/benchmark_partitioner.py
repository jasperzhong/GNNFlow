import itertools
import time

import numpy as np
import torch

from gnnflow.distributed.partition import get_partitioner
from gnnflow.utils import load_dataset


def benchmark_partition_graph(dataset_name, partition_strategy, batch_size):
    p_stgy = partition_strategy
    num_p = 4
    ingestion_batch_size = batch_size
    undirected = True
    _, _, _, dataset = load_dataset(dataset_name)

    num_nodes = 0
    num_edges = 0

    test_partitioner = get_partitioner(p_stgy, num_p)

    overall_start = time.time()
    for i in range(0, len(dataset), ingestion_batch_size):

        print("****** Dataset Range {} to {} Begin ******".format(i,
              i + ingestion_batch_size))

        batch = dataset[i: i + ingestion_batch_size]
        src_nodes = batch["src"].values.astype(np.int64)
        dst_nodes = batch["dst"].values.astype(np.int64)
        timestamps = batch["time"].values.astype(np.float32)
        eids = batch["eid"].values.astype(np.int64)

        num_nodes = num_nodes + \
            len(np.unique(np.concatenate([src_nodes, dst_nodes])))
        num_edges = num_edges + len(eids)

        # undirected
        if undirected:
            src_nodes_ext = np.concatenate([src_nodes, dst_nodes])
            dst_nodes_ext = np.concatenate([dst_nodes, src_nodes])
            src_nodes = src_nodes_ext
            dst_nodes = dst_nodes_ext
            timestamps = np.concatenate([timestamps, timestamps])
            eids = np.concatenate([eids, eids])

        src_nodes = torch.from_numpy(src_nodes)
        dst_nodes = torch.from_numpy(dst_nodes)
        timestamps = torch.from_numpy(timestamps)
        eids = torch.from_numpy(eids)

        partition_start = time.time()
        partitions = test_partitioner.partition(
            src_nodes, dst_nodes, timestamps, eids)
        partition_end = time.time()

        print("Test Partition. Time usage: {} seconds; Speed: {} edges per sec\n"
              .format(partition_end - partition_start, ingestion_batch_size / (partition_end - partition_start)))

        for idx in range(len(partitions)):
            pt = partitions[idx]
            print("Test Partition; Dataset Name:{}; Partition ID:{}; num_edges:{}\n"
                  .format(dataset_name, idx, len(pt.eids)))

        print("Current Partition Table size is :{}\n".format(
            len(test_partitioner.get_partition_table())))

        print("====== Dataset Range {} to {} finished ======\n".format(
            i, i + ingestion_batch_size))

    # load balance
    ptable = test_partitioner.get_partition_table()
    psize_list = []
    for i in range(num_p):
        psize_list.append(ptable.tolist().count(i))
        print("Partition {} has {} nodes.\n".format(
            i, ptable.tolist().count(i)))
    load_factor = np.max(psize_list) / (np.min(psize_list)
                                        if np.min(psize_list) != 0 else 1)

    overall_end = time.time()

    # edge cut
    edge_cut = 0
    tt = 0
    for idx, row in dataset.iterrows():
        u = int(row['src'])
        v = int(row['dst'])
        if ptable[u] != -1 and ptable[v] != -1 and (ptable[u] != ptable[v]):
            edge_cut += 1

    cut_percentage = float(100.0 * float(edge_cut) / float(len(dataset)))

    print("========== All Batch Finished =========\n")
    print("Total Time Usage: {} seconds\n".format(
        overall_end - overall_start))
    print("Load factor is:{} \n".format(load_factor))
    print("Edge Cut Percentage is :{}%; Number of Edge Cut: {}; Number of Total Edge: {}\n"
          .format(cut_percentage, edge_cut, len(dataset)))
    print("========== Test Finished (DataSet:{}, Method:{}, BatchSize:{}) =========\n\n".format(
        dataset_name, p_stgy, ingestion_batch_size))


if __name__ == "__main__":
    datasets = ["WIKI", "REDDIT"]
    partition_strategies = ["hash", "roundrobin",
                            "edgecount", "timestampsum", "ldg"]
    batch_sizes = [10000, 100000]
    for dataset, partition_strategy, batch_size in itertools.product(datasets, partition_strategies, batch_sizes):
        benchmark_partition_graph(dataset, partition_strategy, batch_size)
