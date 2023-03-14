from typing import Optional, Tuple, Union

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

        # self.cache_node_count = torch.zeros(
        #     self.node_capacity, dtype=torch.int32, device=self.device)

        # pointer to the last entry for the recent cached nodes
        self.cache_node_pointer = 0
        self.cache_edge_pointer = 0

        self.hit_count = 0
        self.total_count = 0

    def reset(self):
        """
        Reset cache
        """
        self.cache_node_flag.fill_(False)
        self.cache_node_map.fill_(-1)
        self.cache_index_to_node_id.fill_(-1)
        # self.cache_node_count.fill_(0)

    def get(self, nodes: np.ndarray) -> Tuple[Optional[torch.Tensor], np.ndarray]:
        """
        Get node embeddings from cache

        Args:
            nodes: Node ids

        Returns:
            node_embeds: Node embeddings
            uncached_mask: Mask of uncached nodes
        """
        nodes = torch.from_numpy(nodes).to(self.device)
        cache_mask = self.cache_node_flag[nodes]

        hit_count = cache_mask.sum().item()
        self.hit_count += hit_count
        self.total_count += len(cache_mask)

        if hit_count == 0:
            return None, np.ones(len(nodes), dtype=np.bool_)

        node_embeds = torch.zeros(
            len(nodes), self.dim_node_embed, dtype=torch.float32, device=self.device)

        cache_node_index = self.cache_node_map[nodes[cache_mask]]
        node_embeds[cache_mask] = self.cache_node_buffer[cache_node_index]
        uncached_mask = ~cache_mask

        # # update
        # self.cache_node_count -= 1
        # self.cache_node_count[cache_node_index] = 0

        return node_embeds, uncached_mask.cpu().numpy()

    def put(self, uncached_nodes: torch.Tensor, uncached_node_embeds: torch.Tensor):
        """
        Put node embeddings into cache

        Args:
            uncached_nodes: Node ids
            uncached_node_embeds: Node embeddings
        """
        if len(uncached_nodes) > self.node_capacity:
            num_node_to_cache = self.node_capacity
        else:
            num_node_to_cache = len(uncached_nodes)

        node_id_to_cache = uncached_nodes[:num_node_to_cache]
        node_embed_to_cache = uncached_node_embeds[:num_node_to_cache].detach()

        # # update cache
        # removing_cache_index = torch.topk(
        #     self.cache_node_count, k=num_node_to_cache, largest=False).indices
        # removing_node_id = self.cache_index_to_node_id[removing_cache_index]

        if self.cache_node_pointer + num_node_to_cache < self.node_capacity:
            removing_cache_index = torch.arange(
                self.cache_node_pointer + 1, self.cache_node_pointer + num_node_to_cache + 1)
            self.cache_node_pointer = self.cache_node_pointer + num_node_to_cache
        else:
            removing_cache_index = torch.cat([torch.arange(num_node_to_cache - (self.node_capacity - 1 - self.cache_node_pointer)),
                                             torch.arange(self.cache_node_pointer + 1, self.node_capacity)])
            self.cache_node_pointer = num_node_to_cache - \
                (self.node_capacity - 1 - self.cache_node_pointer) - 1

        removing_cache_index = removing_cache_index.to(
            device=self.device, non_blocking=True)
        removing_node_id = self.cache_index_to_node_id[removing_cache_index]

        self.cache_node_buffer[removing_cache_index] = node_embed_to_cache
        # self.cache_node_count[removing_cache_index] = 0
        self.cache_node_flag[removing_node_id] = False
        self.cache_node_flag[node_id_to_cache] = True
        self.cache_node_map[removing_node_id] = -1
        self.cache_node_map[node_id_to_cache] = removing_cache_index
        self.cache_index_to_node_id[removing_cache_index] = node_id_to_cache

    def hit_rate(self) -> float:
        """
        Get hit rate of cache
        """
        return self.hit_count / self.total_count
