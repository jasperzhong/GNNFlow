import sys

MB = 1 << 20
GB = 1 << 30


def get_default_config(model: str, dataset: str):
    """
    Get default configuration for a model and dataset.

    Args:
        model: Model name.
        dataset: Name of the dataset.

    Returns:
        Default configuration for the model and dataset.
    """
    model, dataset = model.lower(), dataset.lower()
    assert model in ["tgn", "tgat", "dysat", "graphsage", "gat"] and dataset in [
        "wiki", "reddit", "mooc", "lastfm", "gdelt", "mag"], "Invalid model or dataset."

    mod = sys.modules[__name__]
    return getattr(
        mod, f"_{model}_default_config"), getattr(
        mod, f"_{dataset}_default_config")


_tgn_default_config = {
    "dropout": 0.2,
    "att_head": 2,
    "att_dropout": 0.2,
    "num_layers": 1,
    "fanouts": [10],
    "sample_strategy": "recent",
    "num_snapshots": 1,
    "snapshot_time_window": 0,
    "prop_time": False,
    "use_memory": True,
    "dim_time": 100,
    "dim_embed": 100,
    "dim_memory": 100
}

_tgat_default_config = {
    "dropout": 0.1,
    "att_head": 2,
    "att_dropout": 0.1,
    "num_layers": 2,
    "fanouts": [10, 10],
    "sample_strategy": "uniform",
    "num_snapshots": 1,
    "snapshot_time_window": 0,
    "prop_time": False,
    "use_memory": False,
    "dim_time": 100,
    "dim_embed": 100
}

_dysat_default_config = {
    "dropout": 0.1,
    "att_head": 2,
    "att_head": 2,
    "att_dropout": 0.1,
    "num_layers": 2,
    "fanouts": [10, 10],
    "sample_strategy": "uniform",
    "num_snapshots": 3,
    "snapshot_time_window": 10000,
    "prop_time": True,
    "use_memory": False,
    "dim_time": 0,
    "dim_embed": 100
}

_graphsage_default_config = {
    "dim_embed": 100,
    "num_layers": 2,
    "aggregator": 'mean',
    "fanouts": [15, 10],
    "sample_strategy": "uniform",
    "num_snapshots": 1,
    "snapshot_time_window": 0,
    "prop_time": False,
    "use_memory": False,
    "is_static": True
}

_gat_default_config = {
    "dropout": 0.1,
    "att_head": 2,
    "att_dropout": 0.1,
    "num_layers": 2,
    "fanouts": [10, 10],
    "sample_strategy": "uniform",
    "num_snapshots": 1,
    "snapshot_time_window": 0,
    "prop_time": False,
    "use_memory": False,
    "dim_time": 0,
    "dim_embed": 100,
    "is_static": True
}

_wiki_default_config = {
    "initial_pool_size": 10 * MB,
    "maximum_pool_size": 30 * MB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 18,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
    "undirected": True,
    "node_feature": False,
    "edge_feature": True,
    "batch_size": 600
}

_reddit_default_config = {
    "initial_pool_size": 20 * MB,
    "maximum_pool_size": 50 * MB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 62,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
    "undirected": False,
    "node_feature": True,
    "edge_feature": True,
    "batch_size": 600
}

_mooc_default_config = {
    "initial_pool_size": 20 * MB,
    "maximum_pool_size": 50 * MB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 59,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
    "undirected": False,
    "node_feature": False,
    "edge_feature": True,
    "batch_size": 600
}

_lastfm_default_config = {
    "initial_pool_size": 50 * MB,
    "maximum_pool_size": 100 * MB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 650,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
    "undirected": False,
    "node_feature": False,
    "edge_feature": True,
    "batch_size": 600
}

_gdelt_default_config = {
    "initial_pool_size": 10*GB,
    "maximum_pool_size": 20*GB,
    "mem_resource_type": "unified",
    "minimum_block_size": 123,
    "blocks_to_preallocate": 8196,
    "insertion_policy": "insert",
    "undirected": False,
    "node_feature": True,
    "edge_feature": True,
    "batch_size": 4000
}

_mag_default_config = {
    "initial_pool_size": 50*GB,
    "maximum_pool_size": 300*GB,
    "mem_resource_type": "unified",
    "minimum_block_size": 11,
    "blocks_to_preallocate": 65536,
    "insertion_policy": "insert",
    "undirected": False,
    "node_feature": True,
    "edge_feature": False,
    "batch_size": 600
}
