import random
from typing import Iterable, Iterator, List, Union

from torch.utils.data import BatchSampler, Sampler


class BatchSamplerReorder(BatchSampler):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]],
                 batch_size: int, drop_last: bool,
                 num_chunks: int = 0) -> None:
        super(BatchSamplerReorder, self).__init__(
            sampler, batch_size, drop_last)
        self.num_chunks = num_chunks
        self.chunk_size = batch_size // num_chunks
        self.reorder = self.num_chunks > 1
        self.random_size = batch_size

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if self.reorder:
                if len(batch) == self.random_size:
                    yield batch
                    self.reorder = False
                    batch = []
            else:
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def reset(self):
        l = self.batch_size // self.chunk_size
        self.random_size = random.randint(0, l - 1) * self.chunk_size
        self.reorder = True
