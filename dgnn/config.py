from typing import Dict

MB = 1 << 20
GB = 1 << 30


def get_default_config(dataset: str) -> Dict:
    """
    Get default configuration for a dataset.

    Args:
        dataset: Name of the dataset.

    Returns:
    """
    if dataset == "WIKI":
        return _wiki_default_config
    elif dataset == "REDDIT":
        return _reddit_default_config
    elif dataset == "MOOC":
        return _mooc_default_config
    elif dataset == "LASTFM":
        return _lastfm_default_config
    elif dataset == "GDELT":
        return _gdelt_default_config
    elif dataset == "MAG":
        return _mag_default_config
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


_wiki_default_config = {
    "initial_pool_size": 10 * MB,
    "maximum_pool_size": 30 * MB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 64,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
}

_reddit_default_config = {
    "initial_pool_size": 20 * MB,
    "maximum_pool_size": 50 * MB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 64,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
}

_mooc_default_config = {
    "initial_pool_size": 20 * MB,
    "maximum_pool_size": 50 * MB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 64,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",

}

_lastfm_default_config = {
    "initial_pool_size": 50 * MB,
    "maximum_pool_size": 100 * MB,
    "mem_resource_type": "cuda",
    "minimum_block_size": 64,
    "blocks_to_preallocate": 1024,
    "insertion_policy": "insert",
}

_gdelt_default_config = {
    "initial_pool_size": 5*GB,
    "maximum_pool_size": 10*GB,
    "mem_resource_type": "managed",
    "minimum_block_size": 128,
    "blocks_to_preallocate": 8196,
    "insertion_policy": "insert",
}

_mag_default_config = {
    "initial_pool_size": 50*GB,
    "maximum_pool_size": 100*GB,
    "mem_resource_type": "managed",
    "minimum_block_size": 1024,
    "blocks_to_preallocate": 65536,
    "insertion_policy": "insert",
}
