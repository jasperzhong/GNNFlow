from __future__ import annotations

from typing import Union

import torch


def capacity_to_bytes(capacity: int) -> int:
    """
    Convert the capacity of a memory buffer to bytes.
    """
    return capacity * (8 + 4 + 8)


class TemporalBlock:
    """
    This class is used to store the temporal blocks in the graph.

    The blocks are stored in a linked list. The first block is the newest block.
    Each block stores the target vertices and timestamps of the edges. The target
    vertices are sorted by timestamps. The block has a maximum capacity and can
    only store a certain number of edges. The block can be moved to a different
    device.
    """

    def __init__(self, capacity: int, device: Union[torch.device, str]):
        """
        Create a new temporal block. The block is initially empty.

        Arguments:
            capacity: The maximum number of edges that can be stored in the block.
            device: The device to store the block on.
        """
        # lazy initialization
        self._target_vertices = None
        self._timestamps = None
        self._edge_ids = None
        self._capacity = capacity
        self._device = device
        self._size = 0
        self._next_block = None  # points to the next block in the linked list

    def add_edges(self, target_vertices: torch.Tensor, timestamps: torch.Tensor, edge_ids: torch.Tensor):
        """
        Add edges to the block. Assume that the edges are sorted by timestamp.

        Arguments:
            target_vertices: Tensor of shape (N,), where N is the number of edges.
            timestamps: Tensor of shape (N,), where N is the number of edges.
            edge_ids: Tensor of shape (N,), where N is the number of edges.

        Note: raise RuntimeError if the block cannot hold more edges.
        """
        assert target_vertices.shape[0] == timestamps.shape[0] == edge_ids.shape[0], \
            "The number of edges must be the same."
        assert len(target_vertices.shape) == 1 and len(timestamps.shape) == 1 and len(
            edge_ids.shape) == 1, "The target vertices, timestamps and edge ids must be 1D tensors."

        if target_vertices.dtype not in [torch.int32, torch.int64]:
            raise ValueError("Target vertices must be of type int32 or int64.")

        if edge_ids.dtype not in [torch.int32, torch.int64]:
            raise ValueError("Edge ids must be of type int32 or int64.")

        if self._size + target_vertices.size(0) > self._capacity:
            raise RuntimeError("Block can only hold {} edges more, but {} edges are added.".format(
                self._capacity - self._size, target_vertices.size(0)))

        if self.empty():
            # lazy initialization
            self._target_vertices = torch.zeros(
                self._capacity, dtype=torch.long, device=self._device)
            self._timestamps = torch.zeros(
                self._capacity, dtype=torch.float32, device=self._device)
            self._edge_ids = torch.zeros(
                self._capacity, dtype=torch.long, device=self._device)

        if target_vertices.device != self._device or timestamps.device != self._device \
                or edge_ids.device != self._device:
            # move to the correct device
            target_vertices = target_vertices.to(self._device)
            timestamps = timestamps.to(self._device)
            edge_ids = edge_ids.to(self._device)

        if target_vertices.dtype != self._target_vertices.dtype or \
                timestamps.dtype != self._timestamps.dtype or \
                edge_ids.dtype != self._edge_ids.dtype:
            # convert to correct dtype
            target_vertices = target_vertices.to(
                self._target_vertices.dtype)
            timestamps = timestamps.to(self._timestamps.dtype)
            edge_ids = edge_ids.to(self._edge_ids.dtype)

        self._target_vertices[self._size:self._size +
                              target_vertices.size(0)] = target_vertices
        self._timestamps[self._size:self._size +
                         timestamps.size(0)] = timestamps
        self._edge_ids[self._size:self._size +
                       edge_ids.size(0)] = edge_ids
        self._size += target_vertices.size(0)

    def to(self, device: Union[torch.device, str]):
        """
        Move the block to the specified device.

        Arguments:
            device: The device to move the block to.
        """
        if not self.empty():
            self._target_vertices = self._target_vertices.to(device)
            self._timestamps = self._timestamps.to(device)
            self._edge_ids = self._edge_ids.to(device)
        self._device = device
        return self

    def copy_to(self, other: TemporalBlock, device: torch.device):
        """
        Copy the block to another block.

        Arguments:
            other: The block to copy to.
            device: The device to copy the block to.
        """
        if other.capacity < self._capacity:
            raise RuntimeError("The block to copy to has a smaller capacity.")

        if not self.empty():
            if other.empty():
                # lazy initialization
                other._target_vertices = torch.zeros(
                    other._capacity, dtype=torch.long, device=device)
                other._timestamps = torch.zeros(
                    other._capacity, dtype=torch.float32, device=device)
                other._edge_ids = torch.zeros( other._capacity, dtype=torch.long, device=device)
                other._target_vertices[:self._size] = self._target_vertices[:self._size]
                other._timestamps[:self._size] = self._timestamps[:self._size]
                other._edge_ids[:self._size] = self._edge_ids[:self._size]
            else:
                other._target_vertices.to(device)
                other._target_vertices[:self._size] = self._target_vertices[:self._size]
                other._timestamps.to(device)
                other._timestamps[:self._size] = self._timestamps[:self._size]
                other._edge_ids.to(device)
                other._edge_ids[:self._size] = self._edge_ids[:self._size]

        other._size = self._size
        other._device = device

    def size_in_bytes(self):
        """
        Return the size of the block in bytes.
        """
        return capacity_to_bytes(self._capacity)

    def empty(self) -> bool:
        """
        Return whether the block is empty.
        """
        return self._size == 0

    @property
    def device(self):
        return self._device

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value

    @property
    def target_vertices(self):
        if self._target_vertices is None:
            return None
        else:
            return self._target_vertices[:self._size]

    @property
    def timestamps(self):
        if self._timestamps is None:
            return None
        else:
            return self._timestamps[:self._size]

    @property
    def edge_ids(self):
        if self._edge_ids is None:
            return None
        else:
            return self._edge_ids[:self._size]

    @property
    def next_block(self):
        return self._next_block

    @next_block.setter
    def next_block(self, block):
        self._next_block = block

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if self.empty():
            raise RuntimeError("Block is empty")
        elif index < 0 or index >= self._size:
            raise RuntimeError("Index out of range")
        else:
            return self._target_vertices[index], self._timestamps[index], self._edge_ids[index]

    def __iter__(self):
        if self.empty():
            raise RuntimeError("Block is empty")
        else:
            for i in range(self._size):
                yield self._target_vertices[i], self._timestamps[i], self._edge_ids[i]

    def __repr__(self):
        return f"TemporalBlock(capacity={self._capacity}, size={self._size},"  \
            f"device={self._device}, target_vertices={self._target_vertices}," \
            f"timestamps={self._timestamps})"

    def __str__(self):
        return f"TemporalBlock(capacity={self._capacity}, size={self._size},"  \
            f"device={self._device}, target_vertices={self._target_vertices}," \
            f"timestamps={self._timestamps})"
