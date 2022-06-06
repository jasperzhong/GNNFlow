import unittest

import torch
import numpy as np

import dgnn
from dgnn import DynamicGraph


class TestDynamicGraph(unittest.TestCase):
    def test_build_graph(self):
        max_gpu_pool_size = 1 << 30
        alignment = 16
        dgraph = DynamicGraph(max_gpu_pool_size, alignment,
                              dgnn.InsertionPolicy.insert)

        # Add Edges
        source_vertices = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        target_vertices = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        timestamps = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        dgraph.add_edges(source_vertices, target_vertices, timestamps, True)
