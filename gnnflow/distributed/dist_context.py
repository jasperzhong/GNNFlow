import datetime
import logging
import os

import psutil
import pandas as pd
import torch
import torch.distributed
import torch.distributed.rpc as rpc
import numpy as np

import gnnflow.distributed.graph_services as graph_services
from gnnflow.distributed.dispatcher import get_dispatcher
from gnnflow.distributed.kvstore import KVStoreServer
from gnnflow.utils import get_project_root_dir, load_feat, local_world_size


def initialize(rank: int, world_size: int, dataset: pd.DataFrame,
               initial_ingestion_batch_size: int,
               ingestion_batch_size: int, partition_strategy: str,
               num_partitions: int, undirected: bool, data_name: str,
               use_memory: int):
    """
    Initialize the distributed environment.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The number of processes participating in the job.
        dataset (df.DataFrame): The dataset to ingest.
        initial_ingestion_batch_size (int): The number of edges to ingest in
        ingestion_batch_size (int): The number of samples to ingest in each iteration.
        num_partitions (int): The number of partitions to split the dataset into.
        undirected (bool): Whether the graph is undirected.
        data_name (str): the dataset name of the dataset for loading features.
        use_memory (bool): if the kvstore need to initialize the memory.
    """
    # NB: disable IB according to https://github.com/pytorch/pytorch/issues/86962
    rpc.init_rpc("worker%d" % rank, rank=rank, world_size=world_size,
                 #  rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                 #      num_worker_threads=2))
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                     num_worker_threads=2,
                     rpc_timeout=1800,
                     _transports=["uv"],
                     _channels=["cma", "mpt_uv", "basic", "cuda_xth", "cuda_ipc", "cuda_basic"]))
    logging.info("Rank %d: Initialized RPC.", rank)

    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize the KVStore.
    if local_rank == 0:
        graph_services.set_kvstore_server(KVStoreServer())

    if rank == 0:
        dispatcher = get_dispatcher(partition_strategy, num_partitions)
        # load the feature only at rank 0
        node_feats, edge_feats = load_feat(data_name)
        # edge_feats = None
        # node_feats = torch.randn(100000000, 10)
        # logging.info("load feats done")
        chunk = 10
        for i in range(chunk):  # 10 chunks of data
            # train_data, val_data, test_data, full_data = load_dataset(args.data)
            logging.info("{}th chunk add edges".format(i))
            data_dir = os.path.join(get_project_root_dir(), "data")
            path = os.path.join(data_dir, 'MAG', 'edges_{}.csv'.format(i))
            dataset = pd.read_csv(path, engine='pyarrow')
            dispatcher.partition_graph(dataset, initial_ingestion_batch_size,
                                       ingestion_batch_size,
                                       undirected, node_feats, edge_feats,
                                       use_memory)
            del dataset
        # dispatch node feature and node memory here
        # dispatcher.partition_graph(dataset,  initial_ingestion_batch_size,
        #                            ingestion_batch_size,
        #                            undirected, node_feats, edge_feats,
        #                            use_memory)
        futures = []
        dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
        partition_table = graph_services.get_partition_table()
        # TODO: partition unassigned node
        # get the index of the unsigned nodes
        unassigned_nodes_index = (partition_table == -1).nonzero().squeeze()
        partition_id = torch.arange(
            len(unassigned_nodes_index), dtype=torch.int8) % dispatcher.get_num_partitions()
        partition_table[unassigned_nodes_index] = partition_id
        graph_services.set_partition_table(partition_table)
        dispatcher._partitioner._partition_table = partition_table
        dispatcher.broadcast_partition_table()
        for partition_id in range(dispatcher.get_num_partitions()):
            partition_mask = partition_table == partition_id
            assert partition_mask.sum() > 0  # should not be 0
            vertices = torch.arange(len(partition_table), dtype=torch.long)
            partition_vertices = vertices[partition_mask]
            keys = partition_vertices.contiguous()
            kvstore_rank = partition_id * local_world_size()
            if node_feats is not None:
                features = node_feats[keys]
                futures.append(rpc.rpc_async("worker%d" % kvstore_rank, graph_services.push_tensors,
                                             args=(keys, features, 'node')))
        mem = psutil.virtual_memory().percent
        logging.info("peak memory usage: {}".format(mem))
        del node_feats
        # do this separately to decrease peak memory usage
        for partition_id in range(dispatcher.get_num_partitions()):
            partition_mask = partition_table == partition_id
            assert partition_mask.sum() > 0  # should not be 0
            vertices = torch.arange(len(partition_table), dtype=torch.long)
            partition_vertices = vertices[partition_mask]
            keys = partition_vertices.contiguous()
            kvstore_rank = partition_id * local_world_size()
            if use_memory > 0:
                # use None as value and just init keys here.
                memory = torch.zeros(
                    (len(keys), use_memory), dtype=torch.float32)
                memory_ts = torch.zeros(len(keys), dtype=torch.float32)
                dim_raw_message = 2 * use_memory + dim_edge
                mailbox = torch.zeros(
                    (len(keys), dim_raw_message), dtype=torch.float32)
                mailbox_ts = torch.zeros((len(keys), ), dtype=torch.float32)
                all_mem = torch.cat((memory,
                                    memory_ts.unsqueeze(dim=1),
                                    mailbox,
                                    mailbox_ts.unsqueeze(dim=1),
                                     ), dim=1)
                futures.append(rpc.rpc_async("worker%d" % kvstore_rank, graph_services.push_tensors,
                                             args=(keys, all_mem, 'memory')))
    # check
    torch.distributed.barrier()
    logging.info("Rank %d: Number of vertices: %d, number of edges: %d",
                 rank, graph_services.num_vertices(), graph_services.num_edges())
    logging.info("Rank %d: partition table shape: %s",
                 rank, str(graph_services.get_partition_table().shape))

    # save partition table
    partition_table_numpy = graph_services.get_partition_table().numpy()
    np.savetxt('partiton_table.txt', partition_table_numpy, delimiter='\n')
    dgraph = graph_services.get_dgraph()
    logging.info("Rank %d: local number of vertices: %d, number of edges: %d",
                 rank, dgraph._dgraph.num_vertices(), dgraph._dgraph.num_edges())
    mem = psutil.virtual_memory().percent
    logging.info("build graph done memory usage: {}".format(mem))
