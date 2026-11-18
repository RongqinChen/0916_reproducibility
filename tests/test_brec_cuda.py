"""Compare CUDA results with the reference at the original precision."""
import unittest

import networkx as nx
import numpy as np

from rebuttal_experiments.brec_cuda import CudaProducts
from rebuttal_experiments.encodings import polynomial_bank
from rebuttal_experiments.triple_exact import TripleKernel


class CudaParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
        except ImportError as exc:
            raise unittest.SkipTest('CuPy is unavailable') from exc
        if not cupy.cuda.is_available():
            raise unittest.SkipTest('No CUDA device is available')
        cls.products = CudaProducts('cuda:0')

    def test_exact_rational_polynomial_banks(self):
        for graph in (nx.empty_graph(4), nx.complete_graph(20)):
            adjacency = nx.to_numpy_array(graph, dtype=np.int64)
            for basis in ('monomial', 'chebyshev'):
                with self.subTest(nodes=len(adjacency), basis=basis):
                    expected = polynomial_bank(adjacency, 16, basis, 'exact')
                    actual = polynomial_bank(adjacency, 16, basis, 'exact', accelerator=self.products)
                    np.testing.assert_array_equal(actual, expected)

    def test_interval_endpoints_at_each_precision(self):
        graph = nx.disjoint_union(nx.path_graph(5), nx.empty_graph(1))
        adjacency = nx.to_numpy_array(graph, dtype=np.int64)
        for bits in (128, 130, 256, 512):
            for basis in ('monomial', 'chebyshev'):
                with self.subTest(bits=bits, basis=basis):
                    expected = polynomial_bank(adjacency, 16, basis, 'interval', bits)
                    actual = polynomial_bank(adjacency, 16, basis, 'interval', bits,
                                             accelerator=self.products)
                    np.testing.assert_array_equal(actual, expected)

    def test_exact_joint_columns_with_bounded_and_wide_integers(self):
        graph = nx.cycle_graph(5)
        graph.add_edge(0, 5)
        adjacency = nx.to_numpy_array(graph, dtype=np.int64)
        triples = np.asarray(list(np.ndindex((len(graph),) * 3)))
        for basis in ('monomial', 'chebyshev'):
            reference = TripleKernel(adjacency, 16, basis)
            gpu = TripleKernel(adjacency, 16, basis, accelerator=self.products)
            for degrees in ((1, 2, 3), (8, 16, 1), (16, 16, 16)):
                with self.subTest(basis=basis, degrees=degrees):
                    key = ('joint', *degrees)
                    self.assertEqual(gpu.column(triples, key), reference.column(triples, key))
