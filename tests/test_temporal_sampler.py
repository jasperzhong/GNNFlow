import unittest

import numpy as np

from dgnn import DynamicGraph
from dgnn import TemporalSampler
from dgnn.utils import load_dataset, build_dynamic_graph


class TestTemporalSampler(unittest.TestCase):

    @unittest.skip("debug")
    def test_sample_layer(self):
        # build the dynamic graph
        dgraph = DynamicGraph()
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)

        # sample 1-hop neighbors
        sampler = TemporalSampler(dgraph, [2])
        target_vertices = np.array([0, 1, 2])
        blocks = sampler.sample(target_vertices,
                                np.array([1.5, 1.5, 1.5]))
        blocks = blocks[0]

        block = blocks[0]
        self.assertEqual(block.srcdata['ID'].tolist(), [
            0, 1, 2,
            2, 1, 2, 1, 2, 1])
        self.assertEqual(block.srcdata['ts'].tolist(), [
            1.5, 1.5, 1.5,
            1, 0, 1, 0, 1, 0])
        self.assertEqual(block.edata['dt'].tolist(), [
            0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
        self.assertEqual(block.edata['ID'].tolist(), [1, 0, 4, 3, 7, 6])
        self.assertEqual(block.num_src_nodes(), 9)
        self.assertEqual(block.num_dst_nodes(), 3)
        self.assertEqual(block.edges()[0].tolist(), [3, 4, 5, 6, 7, 8])
        self.assertEqual(block.edges()[1].tolist(), [0, 0, 1, 1, 2, 2])

        print("Test sample_layer passed")

    @unittest.skip("debug")
    def test_sample_layer_uniform(self):
        # build the dynamic graph
        dgraph = DynamicGraph()
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)

        # sample 1-hop neighbors
        sampler = TemporalSampler(dgraph, [2], strategy='uniform')
        target_vertices = np.array([0, 1, 2])
        blocks = sampler.sample(target_vertices,
                                np.array([3, 3, 3]))
        blocks = blocks[0]

        block = blocks[0]
        self.assertEqual(block.num_src_nodes(), 9)
        self.assertEqual(block.num_dst_nodes(), 3)

        print("Test sample_layer uniform passed")

    @unittest.skip("debug")
    def test_sample_layer_with_multiple_blocks(self):
        # build the dynamic graph
        dgraph = DynamicGraph(min_block_size=4)
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)

        # add more edges
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([3, 4, 5, 3, 4, 5, 3, 4, 5])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)

        # sample 1-hop neighbors
        sampler = TemporalSampler(dgraph, [2])
        target_vertices = np.array([0, 1, 2])
        blocks = sampler.sample(target_vertices,
                                np.array([1.5, 1.5, 1.5]))
        blocks = blocks[0]

        block = blocks[0]
        self.assertEqual(block.srcdata['ID'].tolist(), [
            0, 1, 2,
            2, 1, 2, 1, 2, 1])
        self.assertEqual(block.srcdata['ts'].tolist(), [
            1.5, 1.5, 1.5,
            1, 0, 1, 0, 1, 0])
        self.assertEqual(block.edata['dt'].tolist(), [
            0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
        self.assertEqual(block.edata['ID'].tolist(), [1, 0, 4, 3, 7, 6])
        self.assertEqual(block.num_src_nodes(), 9)
        self.assertEqual(block.num_dst_nodes(), 3)
        self.assertEqual(block.edges()[0].tolist(), [3, 4, 5, 6, 7, 8])
        self.assertEqual(block.edges()[1].tolist(), [0, 0, 1, 1, 2, 2])

        print("Test sample_layer passed")

    @unittest.skip("debug")
    def test_sampler_layer_with_duplicate_vertices(self):
        # build the dynamic graph
        dgraph = DynamicGraph()
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)

        # sample 1-hop neighbors
        sampler = TemporalSampler(dgraph, [2])
        target_vertices = np.array([0, 1, 2, 0])
        blocks = sampler.sample(target_vertices,
                                np.array([1.5, 1.5, 1.5, 1.5]))
        blocks = blocks[0]

        block = blocks[0]
        self.assertEqual(block.srcdata['ID'].tolist(), [
            0, 1, 2, 0,
            2, 1, 2, 1, 2, 1, 2, 1])
        self.assertEqual(block.srcdata['ts'].tolist(), [
            1.5, 1.5, 1.5, 1.5,
            1, 0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(block.edata['dt'].tolist(), [
            0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
        self.assertEqual(block.edata['ID'].tolist(), [1, 0, 4, 3, 7, 6, 1, 0])
        self.assertEqual(block.num_src_nodes(), 12)
        self.assertEqual(block.num_dst_nodes(), 4)
        self.assertEqual(block.edges()[0].tolist(), [4, 5, 6, 7, 8, 9, 10, 11])
        self.assertEqual(block.edges()[1].tolist(), [0, 0, 1, 1, 2, 2, 3, 3])

        print("Test sampler_layer_with_duplicate_vertices passed")

    @unittest.skip("debug")
    def test_sample_multi_layers(self):
        # build the dynamic graph
        dgraph = DynamicGraph()
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)

        # sample 2-hop neighbors
        sampler = TemporalSampler(dgraph, [2, 2])
        target_vertices = np.array([0, 1, 2])
        blocks = sampler.sample(target_vertices,
                                np.array([1.5, 1.5, 1.5]))

        block = blocks[1][0]
        self.assertEqual(block.srcdata['ID'].tolist(), [
            0, 1, 2,
            2, 1, 2, 1, 2, 1])
        self.assertEqual(block.srcdata['ts'].tolist(), [
            1.5, 1.5, 1.5,
            1, 0, 1, 0, 1, 0])
        self.assertEqual(block.edata['dt'].tolist(), [
            0.5, 1.5, 0.5, 1.5, 0.5, 1.5])
        self.assertEqual(block.edata['ID'].tolist(), [1, 0, 4, 3, 7, 6])
        self.assertEqual(block.num_src_nodes(), 9)
        self.assertEqual(block.num_dst_nodes(), 3)
        self.assertEqual(block.edges()[0].tolist(), [3, 4, 5, 6, 7, 8])
        self.assertEqual(block.edges()[1].tolist(), [0, 0, 1, 1, 2, 2])

        block = blocks[0][0]
        self.assertEqual(block.srcdata['ID'].tolist(), [
            0, 1, 2, 2, 1, 2, 1, 2, 1,
            2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1])
        self.assertEqual(block.srcdata['ts'].tolist(), [
            1.5, 1.5, 1.5, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
        self.assertEqual(block.edata['dt'].tolist(), [
            0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0, 1, 0, 0, 1, 0, 0, 1, 0])
        self.assertEqual(block.edata['ID'].tolist(), [
            1, 0, 4, 3, 7, 6, 7, 6, 3, 7, 6, 3, 7, 6, 3])
        self.assertEqual(block.num_src_nodes(), 24)
        self.assertEqual(block.num_dst_nodes(), 9)
        self.assertEqual(block.edges()[0].tolist(), [
            9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        self.assertEqual(block.edges()[1].tolist(), [
            0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8])

        print("Test sample_multi_layers passed")

    @unittest.skip("debug")
    def test_sample_multi_snapshots(self):
        # build the dynamic graph
        dgraph = DynamicGraph()
        source_vertices = np.array(
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        target_vertices = np.array(
            [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
        timestamps = np.array(
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)

        # sample 1-hop neighbors with two snapshots
        sampler = TemporalSampler(dgraph, [2], num_snapshots=2,
                                  snapshot_time_window=1)
        target_vertices = np.array([0, 1, 2])
        blocks = sampler.sample(target_vertices,
                                np.array([5, 5, 5]))
        blocks = blocks[0]

        block = blocks[1]  # timestamp range: [4, 5]
        self.assertEqual(block.srcdata['ID'].tolist(), [
            0, 1, 2,
            6, 5, 6, 5, 6, 5])
        self.assertEqual(block.srcdata['ts'].tolist(), [
            5, 5, 5,
            5, 4, 5, 4, 5, 4])
        self.assertEqual(block.edata['dt'].tolist(), [
            0, 1, 0, 1, 0, 1])
        self.assertEqual(block.edata['ID'].tolist(), [
            5, 4, 11, 10, 17, 16])
        self.assertEqual(block.num_src_nodes(), 9)
        self.assertEqual(block.num_dst_nodes(), 3)
        self.assertEqual(block.edges()[0].tolist(), [
            3, 4, 5, 6, 7, 8])
        self.assertEqual(block.edges()[1].tolist(), [
            0, 0, 1, 1, 2, 2])

        block = blocks[0]  # timestamp range: [3, 4]
        self.assertEqual(block.srcdata['ID'].tolist(), [
            0, 1, 2,
            5, 4, 5, 4, 5, 4])
        self.assertEqual(block.srcdata['ts'].tolist(), [
            5, 5, 5,
            4, 3, 4, 3, 4, 3])
        self.assertEqual(block.edata['dt'].tolist(), [
            1, 2, 1, 2, 1, 2])
        self.assertEqual(block.edata['ID'].tolist(), [
            4, 3, 10, 9, 16, 15])
        self.assertEqual(block.num_src_nodes(), 9)
        self.assertEqual(block.num_dst_nodes(), 3)
        self.assertEqual(block.edges()[0].tolist(), [
            3, 4, 5, 6, 7, 8])
        self.assertEqual(block.edges()[1].tolist(), [
            0, 0, 1, 1, 2, 2])

        print("Test sample_multi_snapshots passed")

    @unittest.skip("debug")
    def test_sample_multi_layers_multi_snapshots(self):
        # build the dynamic graph
        dgraph = DynamicGraph()
        source_vertices = np.array(
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        target_vertices = np.array(
            [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
        timestamps = np.array(
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)

        # sample 2-hop neighbors with two snapshots
        sampler = TemporalSampler(dgraph, [2, 2], num_snapshots=2,
                                  snapshot_time_window=1)
        target_vertices = np.array([0, 1, 2])
        blocks = sampler.sample(target_vertices,
                                np.array([5, 5, 5]))

        # root -> layer 1, timestamp range: [4, 5]
        block = blocks[1][1]
        self.assertEqual(block.srcdata['ID'].tolist(), [
            0, 1, 2,
            6, 5, 6, 5, 6, 5])
        self.assertEqual(block.srcdata['ts'].tolist(), [
            5, 5, 5,
            5, 4, 5, 4, 5, 4])
        self.assertEqual(block.edata['dt'].tolist(), [
            0, 1, 0, 1, 0, 1])
        self.assertEqual(block.edata['ID'].tolist(), [
            5, 4, 11, 10, 17, 16])
        self.assertEqual(block.num_src_nodes(), 9)
        self.assertEqual(block.num_dst_nodes(), 3)
        self.assertEqual(block.edges()[0].tolist(), [
            3, 4, 5, 6, 7, 8])
        self.assertEqual(block.edges()[1].tolist(), [
            0, 0, 1, 1, 2, 2])

        # root -> layer 1, timestamp range: [3, 4]
        block = blocks[1][0]
        self.assertEqual(block.srcdata['ID'].tolist(), [
            0, 1, 2,
            5, 4, 5, 4, 5, 4])
        self.assertEqual(block.srcdata['ts'].tolist(), [
            5, 5, 5,
            4, 3, 4, 3, 4, 3])
        self.assertEqual(block.edata['dt'].tolist(), [
            1, 2, 1, 2, 1, 2])
        self.assertEqual(block.edata['ID'].tolist(), [
            4, 3, 10, 9, 16, 15])
        self.assertEqual(block.num_src_nodes(), 9)
        self.assertEqual(block.num_dst_nodes(), 3)
        self.assertEqual(block.edges()[0].tolist(), [
            3, 4, 5, 6, 7, 8])
        self.assertEqual(block.edges()[1].tolist(), [
            0, 0, 1, 1, 2, 2])

        # root -> layer 0, timestamp range: [4, 5]
        block = blocks[0][1]
        self.assertEqual(block.srcdata['ID'].tolist(), [
            0, 1, 2, 6, 5, 6, 5, 6, 5,
            6, 5, 6, 5, 6, 5])
        self.assertEqual(block.srcdata['ts'].tolist(), [
            5, 5, 5, 5, 4, 5, 4, 5, 4,
            5, 4, 5, 4, 5, 4])
        self.assertEqual(block.edata['dt'].tolist(), [0, 1, 0, 1, 0, 1])
        self.assertEqual(block.edata['ID'].tolist(), [5, 4, 11, 10, 17, 16])
        self.assertEqual(block.num_src_nodes(), 15)
        self.assertEqual(block.num_dst_nodes(), 9)
        self.assertEqual(block.edges()[0].tolist(), [9, 10, 11, 12, 13, 14])
        self.assertEqual(block.edges()[1].tolist(), [0, 0, 1, 1, 2, 2])

        # root -> layer 0, timestamp range: [3, 4]
        block = blocks[0][0]
        self.assertEqual(block.srcdata['ID'].tolist(), [
            0, 1, 2, 5, 4, 5, 4, 5, 4,
            5, 4, 5, 4, 5, 4])
        self.assertEqual(block.srcdata['ts'].tolist(), [
            5, 5, 5, 4, 3, 4, 3, 4, 3,
            4, 3, 4, 3, 4, 3])
        self.assertEqual(block.edata['dt'].tolist(), [1, 2, 1, 2, 1, 2])
        self.assertEqual(block.edata['ID'].tolist(), [4, 3, 10, 9, 16, 15])
        self.assertEqual(block.num_src_nodes(), 15)
        self.assertEqual(block.num_dst_nodes(), 9)
        self.assertEqual(block.edges()[0].tolist(), [9, 10, 11, 12, 13, 14])
        self.assertEqual(block.edges()[1].tolist(), [0, 0, 1, 1, 2, 2])

        print("Test sample_multi_layers_multi_snapshots passed")

    @unittest.skip("debug")
    def test_sample_layer_with_different_batch_size(self):
        # build the dynamic graph
        dgraph = DynamicGraph()
        source_vertices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        target_vertices = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        timestamps = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        dgraph.add_edges(source_vertices, target_vertices,
                         timestamps, add_reverse=False)

        # sample 1-hop neighbors
        sampler = TemporalSampler(dgraph, [2])
        for bs in range(0, 100, 10):
            target_vertices = np.random.randint(0, 3, bs)
            timestamps = np.random.randint(0, 3, bs)
            sampler.sample(target_vertices,
                           timestamps)

        print("Test sample_layer_with_different_batch_size passed")

    def test_sampler_use_df(self):
        train_df, _, _, df = load_dataset(dataset="REDDIT")
        train_edge_end = df[df['ext_roll'].gt(0)].index[0]
        df = df[:train_edge_end]
        df = df.astype({'time': np.float32})
        dgraph = build_dynamic_graph(train_df, add_reverse=True)
        sampler = TemporalSampler(
            dgraph, fanouts=[10], strategy="recent")

        for _, rows in df.groupby(df.index // 600):
            root_nodes = np.concatenate(
                [rows.src.values, rows.dst.values]).astype(np.int64)
            ts = np.concatenate(
                [rows.time.values, rows.time.values]).astype(
                np.float32)

            try:
                for i in range(len(root_nodes)):
                    block = sampler.sample(root_nodes[i:i+1], ts[i:i+1])[0][0]
                    df_temp = df[df['src'] == root_nodes[i]]
                    if df_temp.empty:
                        df_temp = df[df['dst'] == root_nodes[i]]
                    time = np.array(df_temp[df_temp['time'] < ts[i]]['time'])
                    time = np.flip(time)[:10]
                    origin_id = np.array(df_temp[df_temp['time'] < ts[i]]['dst'])
                    if len(origin_id) == 0:
                        origin_id = np.array(df_temp[df_temp['time'] < ts[i]]['src'])
                    self.assertEqual(len(block.srcdata['ID'][1:]), len(time))
                    self.assertTrue(np.allclose(block.srcdata['ts'][1:], time))
            except AssertionError:
                print("root_nodes: {}".format(root_nodes[i]))
                print("ts: {}".format(ts[i]))
                print("sample ID: {}".format(block.srcdata['ID']))
                print("origin ID: {}".format(origin_id))
                print("sample time: {}".format(block.srcdata['ts']))
                print("orgin time: {}".format(time))

if __name__ == '__main__':
    unittest.main()
