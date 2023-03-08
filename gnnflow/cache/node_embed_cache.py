from typing import Tuple, Union

import numpy as np
import torch


class NodeEmbedCache:
    """
    Node embedding cache on GPU
    """

    def __init__(self, node_embed_cache_ratio: int, num_nodes: int,
                 device: Union[str, torch.device],
                 dim_node_embed: int):
        self.node_embed_cache_ratio = node_embed_cache_ratio
        self.num_nodes = num_nodes
        self.device = device
        self.dim_node_embed = dim_node_embed
        self.node_capacity = int(self.num_nodes * self.node_embed_cache_ratio)

        self.cache_node_buffer = torch.zeros(
            self.node_capacity, self.dim_node_embed, device=self.device)
        self.cache_node_flag = torch.zeros(
            num_nodes, dtype=torch.bool, device=self.device)
        self.cache_node_map = torch.zeros(
            num_nodes, dtype=torch.int64, device=self.device) - 1
        self.cache_index_to_node_id = torch.zeros(
            self.node_capacity, dtype=torch.int64, device=self.device) - 1

        self.hit_count = 0
        self.miss_count = 0

    def get(self, nodes: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get node embeddings from cache

        Args:
            nodes: Node ids

        Returns:
            node_embeds: Node embeddings
            uncached_mask: Mask of uncached nodes
        """
        cache_mask = self.cache_node_flag[nodes]

        self.hit_count += cache_mask.sum().item()
        self.miss_count += (~cache_mask).sum().item()

        node_embeds = torch.zeros(
            len(nodes), self.dim_node_embed, dtype=torch.float32, device=self.device)

        cache_node_index = self.cache_node_map[nodes[cache_mask]]
        node_embeds[cache_mask] = self.cache_node_buffer[cache_node_index]
        uncached_mask = ~cache_mask
        return node_embeds, uncached_mask.cpu()

    def put(self, node_ids: np.ndarray, node_embeds: torch.Tensor) -> None:
        """
        Put node embeddings into cache
        """
        self.node_embed_cache[node_ids] = node_embeds
