import sys

MiB = 1 << 20
GiB = 1 << 30


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
    "dropout": 0.1,
    "att_head": 2,
    "att_dropout": 0.1,
    "num_layers": 1,
    "fanouts": [10],
    "sample_strategy": "recent",
    "num_snapshots": 1,
    "snapshot_time_window": 0,
    "prop_time": False,
    "use_memory": True,
    "dim_time": 100,
    "dim_embed": 100,
    "dim_memory": 172,
    "batch_size": 4000
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
    "dim_embed": 100,
    "batch_size": 600
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
    "dim_embed": 100,
    "batch_size": 4000
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
    "is_static": True,
    "batch_size": 1200
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
    "is_static": True,
    "batch_size": 600
}

_wiki_default_config = {
    "initial_pool_size": 10 * MiB,
    "maximum_pool_size": 30 * MiB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 18,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
    "undirected": True,
    "node_feature": False,
    "edge_feature": True,
}

_reddit_default_config = {
    "initial_pool_size": 20 * MiB,
    "maximum_pool_size": 1000 * MiB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 62,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
    "undirected": False,
    "node_feature": True,
    "edge_feature": True,
}

_mooc_default_config = {
    "initial_pool_size": 20 * MiB,
    "maximum_pool_size": 50 * MiB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 59,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
    "undirected": False,
    "node_feature": False,
    "edge_feature": True,
}

_lastfm_default_config = {
    "initial_pool_size": 50 * MiB,
    "maximum_pool_size": 100 * MiB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 650,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
    "undirected": False,
    "node_feature": False,
    "edge_feature": True,
}

_gdelt_default_config = {
    "initial_pool_size": 10*GiB,
    "maximum_pool_size": 20*GiB,
    "mem_resource_type": "unified",
    "minimum_block_size": 123,
    "blocks_to_preallocate": 8196,
    "insertion_policy": "insert",
    "undirected": False,
    "node_feature": True,
    "edge_feature": True,
}

_mag_default_config = {
    "initial_pool_size": 5*GiB,
    "maximum_pool_size": 50*GiB,
    "mem_resource_type": "unified",
    "minimum_block_size": 11,
    "blocks_to_preallocate": 65536,
    "insertion_policy": "insert",
    "undirected": False,
    "node_feature": True,
    "edge_feature": False,
}
