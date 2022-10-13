import itertools
import time
import unittest

import logging

import numpy as np
import pandas as pd
import torch
from parameterized import parameterized

from gnnflow.distributed.partition import get_partitioner

logging.basicConfig(level=logging.DEBUG)


class TestPartition(unittest.TestCase):

    @parameterized.expand(
        itertools.product(["ldg"], [100]))
    def test_partition_graph(self, partition_strategy, batch_size):

        dataset_name = 'WIKI'
        p_stgy = partition_strategy
        num_p = 4
        ingestion_batch_size = batch_size
        undirected = True
        dataset = pd.read_csv('/data/tgl/{}/edges.csv'.format(dataset_name))  # LINUX
        dataset.rename(columns={'Unnamed: 0': 'eid'}, inplace=True)

        num_nodes = 0
        num_edges = 0

        test_partitioner = get_partitioner(p_stgy, num_p)

        cut_ratio_list = []

        overall_start = time.time()
        for i in range(0, len(dataset), ingestion_batch_size):

            print("****** Dataset Range {} to {} Begin ******".format(i, i + ingestion_batch_size))

            batch = dataset[i: i + ingestion_batch_size]
            src_nodes = batch["src"].values.astype(np.int64)
            dst_nodes = batch["dst"].values.astype(np.int64)
            timestamps = batch["time"].values.astype(np.float32)
            eids = batch["eid"].values.astype(np.int64)

            num_nodes = num_nodes + len(np.unique(np.concatenate([src_nodes, dst_nodes])))
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
            partitions = test_partitioner.partition(src_nodes, dst_nodes, timestamps, eids)
            partition_end = time.time()

            print("Test Partition. Time usage: {} seconds; Speed: {} edges per sec\n"
                  .format(partition_end - partition_start, ingestion_batch_size / (partition_end - partition_start)))

            ppa = test_partitioner.get_partition_table()
            # edge cut
            edge_cut = 0
            tt = 0
            for idx, row in batch.iterrows():
                u = int(row['src'])
                v = int(row['dst'])
                if ppa[u] != -1 and ppa[v] != -1 and (ppa[u] != ppa[v]):
                    edge_cut += 1

            cut_percentage = float(100.0 * float(edge_cut) / float(len(batch)))
            cut_ratio_list.append(cut_percentage)

            print("====== Dataset Range {} to {} finished ======\n".format(i, i + ingestion_batch_size))

        # load balance
        ptable = test_partitioner.get_partition_table()
        psize_list = []
        for i in range(num_p):
            psize_list.append(ptable.tolist().count(i))
            print("Partition {} has {} nodes.\n".format(i, ptable.tolist().count(i)))
        load_factor = np.max(psize_list) / (np.min(psize_list) if np.min(psize_list) != 0 else 1)

        overall_end = time.time()

        avg_cut_p = np.average(cut_ratio_list)

        print("========== All Batch Finished =========\n")

        print("Total Time Usage: {} seconds\n".format(overall_end - overall_start))
        print("Load factor is:{} \n".format(load_factor))
        print("Edge Cut List is :{}".format(cut_ratio_list))
        print("Edge Cut Percentage is :{}%;\n"
              .format(avg_cut_p))
        print("========== Test Finished (DataSet:{}, Method:{}, BatchSize:{}) =========\n\n".format(dataset_name, p_stgy, ingestion_batch_size))