import logging
import os
import time

import psutil
import torch
import torch.distributed
import torch.distributed.rpc as rpc
from tqdm import tqdm

import gnnflow.distributed.graph_services as graph_services
from gnnflow.distributed.dispatcher import get_dispatcher
from gnnflow.distributed.kvstore import KVStoreServer
from gnnflow.utils import load_dataset_in_chunks, load_feat


def initialize(rank: int, world_size: int, partition_strategy: str,
               num_partitions: int, data_name: str, dim_memory: int):
    """
    Initialize the distributed environment.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The number of processes participating in the job.
        num_partitions (int): The number of partitions to split the dataset into.
        data_name (str): the dataset name of the dataset for loading features.
        dim_memory (int): the dimension of memory
    """
    # NB: disable IB according to https://github.com/pytorch/pytorch/issues/86962
    rpc.init_rpc("worker%d" % rank, rank=rank, world_size=world_size,
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                     rpc_timeout=180000,
                     num_worker_threads=32,
                     _transports=["shm", "uv"],
                     _channels=["cma", "mpt_uv", "basic", "cuda_xth", "cuda_ipc", "cuda_basic"]))
    logging.info("Rank %d: Initialized RPC.", rank)

    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize the KVStore.
    if local_rank == 0:
        node_feats, edge_feats = load_feat(data_name, memmap=True)
        dim_node = 0 if node_feats is None else node_feats.shape[1]
        dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
        graph_services.set_kvstore_server(KVStoreServer(
            node_feats, edge_feats, dim_memory, dim_edge))

        if rank == 0:
            dispatcher = get_dispatcher(
                partition_strategy, num_partitions, dim_node > 0, dim_edge > 0, dim_memory > 0, data_name)
            dispatcher.broadcast_node_edge_dim(dim_node, dim_edge)

    torch.distributed.barrier()
    if rank == 0:
        logging.info("initialized done")


def dispatch_full_dataset(rank: int, data_name: str,
                          initial_ingestion_batch_size: int, ingestion_batch_size: int):
    start = time.time()
    if rank == 0:
        dispatcher = get_dispatcher()
        # read csv in chunks
        df_iterator = load_dataset_in_chunks(
            data_name, chunksize=initial_ingestion_batch_size)

        t = tqdm()
        # ingest the first chunk
        for i, dataset in enumerate(df_iterator):
            dataset.rename(columns={'Unnamed: 0': 'eid'}, inplace=True)
            if i > 0:
                for i in range(0, initial_ingestion_batch_size, ingestion_batch_size):
                    dataset_chunk = dataset.iloc[i:i + ingestion_batch_size]
                    dispatcher.partition_graph(dataset_chunk, False)
                    t.update(len(dataset_chunk))
            else:
                dispatcher.partition_graph(dataset, False)
                t.update(initial_ingestion_batch_size)
            del dataset

        t.close()
        logging.info("Rank 0: Ingestion edges done in %.2fs.",
                     time.time() - start)
        dispatcher.dispatch_node_memory()
        logging.info("Rank 0: Dispatch node memory done in %.2fs.",
                     time.time() - start)
        dispatcher.broadcast_rand_sampler()
        logging.info("Rank 0: Broadcast rand sampler done in %.2fs.",
                     time.time() - start)

    # check
    torch.distributed.barrier()
    if rank == 0:
        logging.info("Rank %d: Ingested full dataset in %f seconds.", rank,
                     time.time() - start)

    logging.info("Rank %d: Number of vertices: %d, number of edges: %d",
                 rank, graph_services.num_vertices(), graph_services.num_edges())
    logging.info("Rank %d: partition table shape: %s",
                 rank, str(graph_services.get_partition_table().shape))

    # save partition table
    dgraph = graph_services.get_dgraph()
    logging.info("Rank %d: local number of vertices: %d, number of edges: %d",
                 rank, dgraph._dgraph.num_vertices(), dgraph._dgraph.num_edges())
    mem = psutil.virtual_memory().percent
    logging.info("build graph done memory usage: {}".format(mem))
