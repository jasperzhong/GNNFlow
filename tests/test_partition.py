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
        itertools.product(["hash"], [10000], [False, True]))
    def test_partition_graph(self, partition_strategy, batch_size, assign_with_dst):

        dataset_name = 'REDDIT'
        p_stgy = partition_strategy
        num_p = 4
        ingestion_batch_size = batch_size
        undirected = True
        dataset = pd.read_csv('/home/ubuntu/data/{}/edges.csv'.format(dataset_name))  # LINUX
        dataset.rename(columns={'Unnamed: 0': 'eid'}, inplace=True)

        num_nodes = 0
        num_edges = 0

        edge_cut_list = []

        test_partitioner = get_partitioner(p_stgy, num_p, assign_with_dst)

        overall_start = time.time()

        edge_num_tot = [0 for i in range(num_p)]

        for i in range(0, len(dataset), ingestion_batch_size):

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

                # src_nodes_ext = np.array(list(itertools.chain.from_iterable(zip(src_nodes, dst_nodes))))
                # dst_nodes_ext = np.array(list(itertools.chain.from_iterable(zip(dst_nodes, src_nodes))))
                # src_nodes = src_nodes_ext
                # dst_nodes = dst_nodes_ext
                #
                # timestamps_ext = np.array(list(itertools.chain.from_iterable(zip(timestamps, timestamps))))
                # timestamps = timestamps_ext
                #
                # eids_ext = np.array(list(itertools.chain.from_iterable(zip(eids, eids))))
                # eids = eids_ext

            src_nodes = torch.from_numpy(src_nodes)
            dst_nodes = torch.from_numpy(dst_nodes)
            timestamps = torch.from_numpy(timestamps)
            eids = torch.from_numpy(eids)


            partition_start = time.time()
            partitions = test_partitioner.partition(src_nodes, dst_nodes, timestamps, eids)
            partition_end = time.time()

            for pt_idx in range(num_p):
                edge_num_tot[pt_idx] += len(partitions[pt_idx].eids)

            # edge_cut = 0
            # ptablein = test_partitioner.get_partition_table()
            # for idx, row in batch.iterrows():
            #     u = int(row['src'])
            #     v = int(row['dst'])
            #     if ptablein[u] != -1 and ptablein[v] != -1 and (ptablein[u] != ptablein[v]):
            #         edge_cut += 1
            #
            # edge_cut_list.append(float(100.0 * float(edge_cut) / float(len(batch))))
            eid_list = torch.tensor([])
            for pid in range(num_p):
                eid_list = torch.cat([eid_list, partitions[pid].eids])

            # check the rows
            for idx, row in batch.iterrows():
                u = int(row['src'])
                v = int(row['dst'])
                e = int(row['eid'])

                check_tensor = (eid_list == e).nonzero()
                if len(check_tensor) == 0:
                    print("Find UNASSIGNED edge. u:{}, v:{}, eid:{}\n".format(u, v, eid))

        # load balance
        ptable = test_partitioner.get_partition_table()
        psize_list = []
        for i in range(num_p):
            psize_list.append(ptable.tolist().count(i))
            print("Partition {} has {} nodes.\n".format(i, ptable.tolist().count(i)))
        load_factor = np.max(psize_list) / (np.min(psize_list) if np.min(psize_list) != 0 else 1)

        for i in range(num_p):
            print("Partition {} has {} edges. \n".format(i, edge_num_tot[i]))

        print("The Sum of # of edges is: {} \n".format(np.sum(edge_num_tot)))

        overall_end = time.time()

        print("========== All Batch Finished =========\n")

        # Print Partition Table
        for i in range(len(ptable)):
            if ptable[i].item() >= num_p or ptable[i].item() == -1:
                print("Incorrect Partition Table in vid {} is:{}\n".format(i, ptable[i].item()))


        print("Ptable is {}".format(ptable))
        print("Total Time Usage: {} seconds\n".format(overall_end - overall_start))
        print("Load factor is:{} \n".format(load_factor))
        # print("Edge Cut Percentage is :{}%;".format(np.average(edge_cut_list)))
        print("========== Test Finished (DataSet:{}, Method:{}, BatchSize:{}, Assign_With_Dst:{}) =========\n\n".format(dataset_name, p_stgy, ingestion_batch_size, assign_with_dst))

