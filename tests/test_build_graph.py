from distutils.command.build import build
import unittest

from dgnn.build_graph import build_dynamic_graph

class TestBuildGraph(unittest.TestCase):
    def test_build_graph(self):
        """
        Test building a dynamic graph from edges.csv(REDDIT)
        Only use training data to build a graph
        """
        dgraph = build_dynamic_graph("REDDIT")
        print("done")
    