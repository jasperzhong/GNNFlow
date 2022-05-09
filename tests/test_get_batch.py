import unittest
import numpy as np
import random
from dgnn.build_graph import load_graph

class TestGetBatch(unittest.TestCase):
    def test_get_batch(self, batch_size=600):
        df = load_graph(None, 'REDDIT')
        df = df[0]
        group_indexes = list()
        i = 0
        group_indexes.append(np.array(df.index // batch_size))
        for _, rows in df.groupby(group_indexes[random.randint(0, len(group_indexes) - 1)]):
            i += 1
            print()
            # np.random.randint(self.num_nodes, size=n)
            # TODO: wrap a neglink sampler
            length = np.max(np.array(rows.src.values, dtype=int))
            # TODO: eliminate np to tensor
            target_nodes = np.concatenate([rows.src.values, rows.dst.values, np.random.randint(length, size=len(rows.src.values))]).astype(int)
            ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
            if i == 785:
                print()
                print(target_nodes.shape)
                print(ts.shape)
            self.assertEqual(target_nodes.shape[0], ts.shape[0])