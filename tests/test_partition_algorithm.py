from dgnn.distributed.partition import get_partitioner
from dgnn.distributed.partition import Partition
import torch
import unittest
import time
import pandas as pd
import numpy as np


class TestPartition(unittest.TestCase):

    def test_partition_graph(self):

        dataset_name = 'REDDIT'
        p_stgy = 'hash'
        ingestion_batch_size = 100000
        dataset = pd.read_csv('/data/tgl/{}/edges.csv'.format(dataset_name))  # LINUX

        num_nodes = 0
        num_edges = 0

        test_partitioner = get_partitioner(p_stgy, 4)

        for i in range(0, len(dataset), ingestion_batch_size):

            batch = dataset[i: i + ingestion_batch_size]
            src_nodes = batch["src"].values.astype(np.int64)
            dst_nodes = batch["dst"].values.astype(np.int64)
            timestamps = batch["time"].values.astype(np.float32)
            eids = batch["eid"].values.astype(np.int64)

            num_nodes = num_nodes + len(np.unique(np.concatenate([src_nodes, dst_nodes])))
            num_edges = num_edges + len(eids)

            src_nodes = torch.from_numpy(src_nodes)
            dst_nodes = torch.from_numpy(dst_nodes)
            timestamps = torch.from_numpy(timestamps)
            eids = torch.from_numpy(eids)

            partitions = test_partitioner.partition(src_nodes, dst_nodes, timestamps, eids)

            for idx in range(len(partitions)):
                pt = partitions[idx]
                print("Test Partition; Dataset Name:{}; Partition ID:{}; num_edges:{}\n"
                      .format(dataset_name, idx, len(pt.eids)))

            print("====== Dataset Range {} to {} finished ======".format(i, i + ingestion_batch_size))
