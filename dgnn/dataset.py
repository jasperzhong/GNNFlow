import torch
import math
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
import numpy as np
import re
import collections
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')


class DynamicGraphDataset(Dataset):
    def __init__(self, df):
        super(DynamicGraphDataset).__init__()
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]
        length = np.max(np.array(self.df['dst'], dtype=int))
        target_nodes = np.array(
            [row.src, row.dst, np.random.randint(length)]).astype(
            np.int64)
        ts = np.array(
            [row.time, row.time, row.time]).astype(
            np.float32)
        eid = row['Unnamed: 0']
        return (target_nodes, ts, eid)

    def __len__(self):
        return len(self.df)


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
            out = np.zeros(numel).resize(len(batch), elem.size)
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
            return elem_type({key: default_collate_ndarray([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate_ndarray([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate_ndarray(samples) for samples in zip(*batch)))
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
                return elem_type([default_collate_ndarray(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate_ndarray(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
