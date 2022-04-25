from typing import Union

import torch


class TemporalBlock:
    def __init__(self, capacity: int):
        """
        Create a new temporal block. The block is initially empty.

        Arguments:
            capacity: The maximum number of edges that can be stored in the block.
        """
        # lazy initialization
        self.target_vertices = None
        self.timestamps = None
        self.capacity = capacity
        self.size = 0
        self.next_block = None  # points to the next block in the linked list

    def add_edges(self, target_vertices: torch.Tensor, timestamps: torch.Tensor):
        """
        Add edges to the block. Assume that the edges are sorted by timestamp.

        Arguments:
            target_vertices: Tensor of shape (N,), where N is the number of edges.
            timestamps: Tensor of shape (N,), where N is the number of edges.
        """
        assert target_vertices.shape[0] == timestamps.shape[0]
        assert len(target_vertices.shape) == 1 and len(timestamps.shape) == 1

        if self.size + target_vertices.size(0) > self.capacity:
            raise RuntimeError("Block is full")
        else:
            if self.target_vertices is None:
                self.target_vertices = torch.zeros(
                    self.capacity, dtype=torch.long)
                self.timestamps = torch.zeros(
                    self.capacity, dtype=torch.float32)

            self.target_vertices[self.size:self.size +
                                 target_vertices.size(0)] = target_vertices
            self.timestamps[self.size:self.size +
                            timestamps.size(0)] = timestamps
            self.size += target_vertices.size(0)

    def device(self) -> torch.device:
        """
        Return the device of the block.
        """
        if self.target_vertices is not None:
            return self.target_vertices.device
        else:
            return torch.device("cpu")

    def to(self, device: Union[torch.device, str]):
        """
        Move the block to the specified device.

        Arguments:
            device: The device to move the block to.
        """
        if self.target_vertices is not None:
            self.target_vertices = self.target_vertices.to(device)
            self.timestamps = self.timestamps.to(device)
        return self

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.target_vertices is None or self.timestamps is None:
            raise RuntimeError("Block is empty")
        elif index < 0 or index >= self.size:
            raise RuntimeError("Index out of range")
        else:
            return self.target_vertices[index], self.timestamps[index]

    def __iter__(self):
        if self.target_vertices is None or self.timestamps is None:
            raise RuntimeError("Block is empty")
        else:
            for i in range(self.size):
                yield self.target_vertices[i], self.timestamps[i]

    def __repr__(self):
        return f"TemporalBlock(capacity={self.capacity}, size={self.size})"

    def __str__(self):
        return f"TemporalBlock(capacity={self.capacity}, size={self.size})"
