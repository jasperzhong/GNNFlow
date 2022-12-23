import collections
import re
from typing import Iterable, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed
from torch._six import string_classes
from torch.utils.data import BatchSampler, Dataset, Sampler

from gnnflow.utils import DstRandEdgeSampler, RandEdgeSampler, local_rank

np_str_obj_array_pattern = re.compile(r'[SaUO]')


class EdgePredictionDataset(Dataset):
    """
    Edge prediction dataset.

    It samples negative edges from the given graph and returns the node ids,
    timestamps and edge ids for the positive and negative edges.

    Args:
        data: the dataframe for the dataset.
        neg_sampler: the negative sampler.
    """

    def __init__(self, data: pd.DataFrame,
                 neg_sampler: Optional[DstRandEdgeSampler] = None):
        super(EdgePredictionDataset, self).__init__()
        self.data = data
        self.length = np.max(np.array(data['dst'], dtype=int))
        self.neg_sampler = neg_sampler

    def __getitem__(self, index):
        row = self.data.iloc[index]
        if self.neg_sampler is not None:
            neg_batch = self.neg_sampler.sample(len(row.src.values))
            target_nodes = np.concatenate(
                [row.src.values, row.dst.values, neg_batch]).astype(
                np.int64)
            ts = np.concatenate(
                [row.time.values, row.time.values, row.time.values]).astype(
                np.float32)
        else:
            target_nodes = np.concatenate(
                [row.src.values, row.dst.values]).astype(np.int64)
            ts = np.concatenate(
                [row.time.values, row.time.values]).astype(np.float32)
        eid = row['eid'].values
        return (target_nodes, ts, eid)

    def __len__(self):
        return len(self.data)


class RandomStartBatchSampler(BatchSampler):
    """
    The sampler select a random start point for each epoch.
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]],
                 batch_size: int, drop_last: bool,
                 num_chunks: int = 1, world_size: int = 1):
        """
        Args:
            sampler: Base class for all Samplers.
            batch_size: Size of mini-batch.
            drop_last: Set to ``True`` to drop the last incomplete batch, if the
                dataset size is not divisible by the batch size. If ``False`` and
                the size of dataset is not divisible by the batch size, then the
                last batch will be smaller.
            num_chunks: Number of chunks to split the batch into.
            world_size: For GDELT and MAG distributed training
        """
        super(RandomStartBatchSampler, self).__init__(sampler, batch_size,
                                                      drop_last)
        assert 0 < num_chunks < batch_size, "num_chunks must be in (0, batch_size)"

        self.num_chunks = num_chunks
        self.chunk_size = batch_size // num_chunks
        self.reorder = self.num_chunks > 1
        self.random_size = batch_size
        if world_size > 1:
            self.device = torch.device('cuda', local_rank())
        else:
            self.device = torch.device('cpu')
        self.world_size = world_size

    def __iter__(self) -> Iterator[List[int]]:
        self.reset()
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
        self.reorder = self.num_chunks > 1
        l = self.batch_size // self.chunk_size
        randint = torch.randint(
            0, self.num_chunks, size=(1,), device=self.device)
        if self.world_size > 1:
            torch.distributed.broadcast(randint, src=0)
        self.random_size = int(randint) * self.chunk_size
        if self.random_size == 0:
            self.reorder = False


class DistributedBatchSampler(BatchSampler):
    """
    Distributed batch sampler.
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]],
                 batch_size: int, drop_last: bool,
                 rank: int, world_size: int,
                 num_chunks: int = 1):
        """
        Args:
            sampler: Base class for all Samplers.
            batch_size: Size of mini-batch.
            drop_last: Set to ``True`` to drop the last incomplete batch, if the
                dataset size is not divisible by the batch size. If ``False`` and
                the size of dataset is not divisible by the batch size, then the
                last batch will be smaller.
            rank: The rank of the current process.
            world_size: The number of processes.
            num_chunks: Number of chunks to split the batch into.
        """
        super(DistributedBatchSampler, self).__init__(sampler, batch_size,
                                                      drop_last)
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank()
        self.device = torch.device('cuda', self.local_rank)
        assert 0 < num_chunks < batch_size, "num_chunks must be in (0, batch_size)"

        self.num_chunks = num_chunks
        self.chunk_size = batch_size // num_chunks
        self.reorder = False
        self.random_size = batch_size

    def __iter__(self) -> Iterator[List[int]]:
        self.reset()
        batch = []
        for idx in self.sampler:
            if idx % self.world_size != self.rank:
                continue
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
        self.reorder = self.num_chunks > 1
        if self.reorder:
            if self.rank == 0:
                randint = torch.randint(
                    0, self.num_chunks, size=(1,), device=self.device)
            else:
                randint = torch.zeros(1, dtype=torch.int64, device=self.device)

            torch.distributed.broadcast(randint, src=0)
            self.random_size = int(randint.item() * self.chunk_size)
            if self.random_size == 0:
                self.reorder = False


default_collate_err_msg_format = (
    "default_collate_ndarray: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def default_collate_ndarray(batch):
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`np.ndarray`, a `Sequence` of :class:`np.ndarray`, a
        Collection of :class:`np.ndarray`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

        Here is the general input type (based on the type of the element within the batch) to output type mapping:
        * :class:`np.ndarray` -> :class:`np.ndarray` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`np.ndarray`
        * `float` -> :class:`np.ndarray`
        * `int` -> :class:`np.ndarray`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate_ndarray([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate_ndarray([V1_1, V1_2, ...]), default_collate_ndarray([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate_ndarray([V1_1, V1_2, ...]), default_collate_ndarray([V2_1, V2_2, ...]), ...]`

        Args:
            batch: a single batch to be collated

        Examples:
            >>> # Example with a batch of `int`s:
            >>> default_collate_ndarray([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> default_collate_ndarray(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> default_collate_ndarray([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_collate_ndarray([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> default_collate_ndarray([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> default_collate_ndarray([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, np.ndarray):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.size for x in batch)
            out = np.zeros(numel, dtype=elem.dtype).reshape(
                len(batch), elem.size)
        return np.stack(batch, 0, out=out).flatten('F')  # column major
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    default_collate_err_msg_format.format(elem.dtype))

            return default_collate_ndarray([b for b in batch])
        elif elem.shape == ():  # scalars
            return np.array(batch)
    elif isinstance(elem, float):
        return np.array(batch, dtype=np.float64)
    elif isinstance(elem, int):
        return np.array(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type(
                {key: default_collate_ndarray([d[key] for d in batch])
                 for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate_ndarray([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*
                         (default_collate_ndarray(samples)
                          for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        # It may be accessed twice, so we use a list.
        transposed = list(zip(*batch))

        if isinstance(elem, tuple):
            # Backwards compatibility.
            return [default_collate_ndarray(samples) for samples in transposed]
        else:
            try:
                return elem_type([default_collate_ndarray(samples)
                                  for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate_ndarray(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
