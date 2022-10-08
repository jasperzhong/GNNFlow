from typing import List

import torch


class KVStoreServer:
    """
    Key-value store server.

    NB: we let local root (i.e., local_rank == 0) to be the server.

    The value can be:
    - node feature (key: "N{node_id}", value: node feature)
    - edge feature (key: "E{edge_id}", value: edge feature)
    - memory (key: "M{node_id}", value: memory)
    """

    def __init__(self):
        # keys -> tensors
        self._map = {}

    def push(self, keys: torch.Tensor, tensors: List[torch.Tensor]):
        """
        Push tensors to the server.

        Args:
            keys (torch.Tensor): The keys.
            tensors (List[torch.Tensor]): The tensors.
        """
        assert len(keys) == len(
            tensors), "The number of keys and tensors must be the same."

        for key, tensor in zip(keys, tensors):
            self._map[key] = tensor

    def pull(self, keys: torch.Tensor) -> List[torch.Tensor]:
        """
        Pull tensors from the server.

        Args:   
            keys (torch.Tensor): The keys.

        Returns:
            List[torch.Tensor]: The tensors.
        """
        return [self._map[key] for key in keys]


class KVStoreClient:
    """
    Key-value store client.

    It is used by the trainer to push/pull tensors to/from the KVStore servers.
    """

    def __init__(self, partition_table: torch.Tensor):
        self._partition_table = partition_table

    def push(self, keys: torch.Tensor, tensors: List[torch.Tensor]):
        """
        Push tensors to the corresponding KVStore servers according to the partition table.

        Args:
            keys (torch.Tensor): The keys.
            tensors (List[torch.Tensor]): The tensors.
        """
        # TODO(guangming): rpc call to the corresponding KVStore servers (call graph_services.push_tensors)

    def pull(self, keys: torch.Tensor) -> List[torch.Tensor]:
        """
        Pull tensors from the corresponding KVStore servers according to the partition table.

        Args:
            keys (torch.Tensor): The keys.

        Returns:
            List[torch.Tensor]: The tensors.
        """
        # TODO(guangming): rpc call to the corresponding KVStore servers (call graph_services.pull_tensors)
