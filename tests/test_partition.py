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

    def test_partition_graph(self):

        dataset_name = 'WIKI'
        p_stgy = "hash"
        num_p = 4
        undirected = True
        dataset = pd.read_csv('/data/tgl/{}/edges.csv'.format(dataset_name))  # LINUX
        dataset.rename(columns={'Unnamed: 0': 'eid'}, inplace=True)

        num_nodes = 0
        num_edges = 0

        test_partitioner = get_partitioner(p_stgy, num_p)

        overall_start = time.time()

        i = 0


        round = 0
        while i != len(dataset):
            round = round + 1
            start_day = dataset[i:i+1]
            start_day_time = start_day["time"].values.astype(np.float32)[0]

            j = i
            while j < len(dataset) and dataset[j:j+1]["time"].values.astype(np.float32)[0] - start_day_time < 86400.0:
                j = j + 1


            print("Round:{} ****** Dataset Range {} to {} Begin ******".format(round, i, j))

            batch = dataset[i: j]
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
                  .format(partition_end - partition_start, (j - i) / (partition_end - partition_start)))

            print("Round:{} ====== Dataset Range {} to {} length:{} finished ======\n".format(round, i, j, j - i))

            i = j

        # load balance
        ptable = test_partitioner.get_partition_table()
        psize_list = []
        for i in range(num_p):
            psize_list.append(ptable.tolist().count(i))
            print("Partition {} has {} nodes.\n".format(i, ptable.tolist().count(i)))
        load_factor = np.max(psize_list) / (np.min(psize_list) if np.min(psize_list) != 0 else 1)

        overall_end = time.time()

        # edge cut
        # edge_cut = 0
        # tt = 0
        # for idx, row in dataset.iterrows():
        #     u = int(row['src'])
        #     v = int(row['dst'])
        #     if ptable[u] != -1 and ptable[v] != -1 and (ptable[u] != ptable[v]):
        #         edge_cut += 1

        # cut_percentage = float(100.0 * float(edge_cut) / float(len(dataset)))

        print("========== All Batch Finished =========\n")

        # Print Partition Table
        for i in range(len(ptable)):
            if ptable[i].item() >= num_p:
                print("Incorrect Partition Table in vid {} is:{}\n".format(i, ptable[i].item()))

        print("Ptable is {}".format(ptable))
        print("Total Time Usage: {} seconds\n".format(overall_end - overall_start))
        print("Load factor is:{} \n".format(load_factor))
        print("Edge Cut Percentage is :{}%; Number of Edge Cut: {}; Number of Total Edge: {}\n"
              .format(cut_percentage, edge_cut, len(dataset)))
        print("========== Test Finished (DataSet:{}, Method:{}, BatchSize:{}) =========\n\n".format(dataset_name, p_stgy, ingestion_batch_size))