import time
import unittest

import numpy as np
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from gnnflow.data import (EdgePredictionDataset, RandomStartBatchSampler,
                       default_collate_ndarray)
from gnnflow.utils import get_batch, load_dataset


class TestDataset(unittest.TestCase):
    def test_loader(self, num_workers=0):
        train_df, val_df, test_df, df = load_dataset('REDDIT')

        ds = EdgePredictionDataset(train_df)

        sampler = BatchSampler(SequentialSampler(
            ds), batch_size=600, drop_last=False)

        a = DataLoader(dataset=ds, sampler=sampler,
                       collate_fn=default_collate_ndarray,
                       num_workers=num_workers)
        ti = 0
        ite = iter(get_batch(train_df, batch_size=600))
        start_loader = 0
        avg_loader = 0
        avg_batch = 0
        for target, ts, eid in a:
            end_loader = time.time()
            loader_time = end_loader - start_loader
            start_batch = time.time()
            target1, ts1, eid1 = ite.__next__()
            end_batch = time.time()
            batch_time = end_batch - start_batch
            self.assertTrue(np.array_equal(target[:1200], target1[:1200]))
            self.assertTrue(np.array_equal(ts[:1200], ts1[:1200]))
            self.assertTrue(np.array_equal(eid[:1200], eid1[:1200]))
            ti = ti + 1
            if ti > 10:
                break
            if ti > 1:
                avg_loader += loader_time
                avg_batch += batch_time
            start_loader = time.time()

        print("avg loader time with {} workers: {}".format(
            num_workers, avg_loader / 10))
        print("avg batch time: {}".format(avg_batch / 10))

    def test_sampler(self):
        train_df, val_df, test_df, df = load_dataset('REDDIT')

        ds = EdgePredictionDataset(train_df)

        sampler = RandomStartBatchSampler(SequentialSampler(
            ds), batch_size=600, drop_last=False, num_chunks=8)

        a = DataLoader(dataset=ds, sampler=sampler,
                       collate_fn=default_collate_ndarray,
                       num_workers=0)
        ti = 0
        ite = iter(get_batch(train_df, batch_size=600))
        start_loader = 0
        avg_loader = 0
        avg_batch = 0
        sampler.reset()
        for target, ts, eid in a:
            end_loader = time.time()
            loader_time = end_loader - start_loader
            start_batch = time.time()
            target1, ts1, eid1 = ite.__next__()
            end_batch = time.time()
            batch_time = end_batch - start_batch

            ti = ti + 1
            if ti > 10:
                break
            if ti > 1:
                avg_loader += loader_time
                avg_batch += batch_time
            start_loader = time.time()

        print("avg loader time: {}".format(
            avg_loader / 10))
        print("avg batch time: {}".format(avg_batch / 10))


if __name__ == "__main__":
    unittest.main()
