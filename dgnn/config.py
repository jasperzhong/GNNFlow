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
    assert model in ["tgn", "tgat", "dysat"] and dataset in [
        "wiki", "reddit", "mooc", "lastfm", "gdelt", "mag"], "Invalid model or dataset."

    mod = sys.modules[__name__]
    return getattr(mod, f"_{model}_default_config"), getattr(mod, f"_{dataset}_default_config")


_tgn_default_config = {
    "dropout": 0.2,
    "attn_dropout": 0.2,
    "sample_layer": 1,
    "sample_neighbor": [10],
    "sample_strategy": "recent",
    "sample_history": 1,
    "sample_duration": 0,
    "prop_time": False,
    "use_memory": True
}

_tgat_default_config = {
    "dropout": 0.1,
    "attn_dropout": 0.1,
    "sample_layer": 2,
    "sample_neighbor": [10, 10],
    "sample_strategy": "uniform",
    "sample_history": 1,
    "sample_duration": 0,
    "prop_time": False,
    "use_memory": False
}

_dysat_default_config = {
    "dropout": 0.1,
    "attn_dropout": 0.1,
    "sample_layer": 2,
    "sample_neighbor": [10, 10],
    "sample_strategy": "uniform",
    "sample_history": 3,
    "sample_duration": 0,
    "prop_time": False,
    "use_memory": False
}

_wiki_default_config = {
    "initial_pool_size": 10 * MB,
    "maximum_pool_size": 30 * MB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 64,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
    "undirected": True,
    "node_feature": False,
    "edge_feature": True
}

_reddit_default_config = {
    "initial_pool_size": 20 * MB,
    "maximum_pool_size": 50 * MB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 64,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
    "undirected": True,
    "node_feature": False,
    "edge_feature": True
}

_mooc_default_config = {
    "initial_pool_size": 20 * MB,
    "maximum_pool_size": 50 * MB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 64,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
    "undirected": True,
    "node_feature": False,
    "edge_feature": True
}

_lastfm_default_config = {
    "initial_pool_size": 50 * MB,
    "maximum_pool_size": 100 * MB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 64,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
    "undirected": True,
    "node_feature": False,
    "edge_feature": True
}

_gdelt_default_config = {
    "initial_pool_size": 5*GB,
    "maximum_pool_size": 10*GB,
    "mem_resource_type": "managed",
    "minimum_block_size": 128,
    "blocks_to_preallocate": 8196,
    "insertion_policy": "insert",
    "undirected": False,
    "node_feature": True,
    "edge_feature": True
}

_mag_default_config = {
    "initial_pool_size": 50*GB,
    "maximum_pool_size": 100*GB,
    "mem_resource_type": "managed",
    "minimum_block_size": 1024,
    "blocks_to_preallocate": 65536,
    "insertion_policy": "insert",
    "undirected": False,
    "node_feature": True,
    "edge_feature": False
}
