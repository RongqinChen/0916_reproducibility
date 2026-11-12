"""Exact regression checks for polynomial bases and interval dispatch."""
from fractions import Fraction
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from rebuttal_experiments.brec import condition_record
from rebuttal_experiments.encodings import descriptors, feature_matrix, polynomial_bank
from rebuttal_experiments.intervals import Interval


class PolynomialBackendTests(unittest.TestCase):
    def test_chebyshev_interval_path_matches_analytic_values(self):
        # The normalized three-node path has M^3=M. Hence T_(odd)(M)=M,
        # T_(4k)(M)=I and T_(4k+2)(M) swaps the two end vertices.
        adjacency = nx.to_numpy_array(nx.path_graph(3), dtype=np.int64)
        for bits in (128, 256, 512):
            bank = polynomial_bank(adjacency, 16, 'chebyshev', 'interval', bits)
            for degree in range(17):
                for u, v in np.ndindex(3, 3):
                    with self.subTest(bits=bits, degree=degree, u=u, v=v):
                        value = bank[degree, u, v]
                        self.assertIsInstance(value, Interval)
                        lo, hi = value.lower(), value.upper()
                        if degree % 2 and adjacency[u, v]:
                            # The exact entry is 1/sqrt(2); compare squares
                            # of rational bounds without introducing floats.
                            self.assertGreaterEqual(lo, 0)
                            self.assertLessEqual(lo * lo, Fraction(1, 2))
                            self.assertGreaterEqual(hi * hi, Fraction(1, 2))
                        else:
                            expected = (int(u == v) if degree % 4 == 0
                                        else int(u == 2 - v) if degree % 4 == 2 else 0)
                            self.assertLessEqual(lo, expected)
                            self.assertGreaterEqual(hi, expected)

    def test_regular_graph_exact_interval_features_agree(self):
        adjacency = nx.to_numpy_array(nx.cycle_graph(4), dtype=np.int64)
        for basis in ('monomial', 'chebyshev'):
            for variant in ('polynomial', 'endpoint'):
                for q in (2, 4, 8, 16):
                    with self.subTest(basis=basis, variant=variant, q=q):
                        exact = feature_matrix(adjacency, q, 2, variant, basis, 'exact')
                        interval = feature_matrix(adjacency, q, 2, variant, basis, 'interval', 128)
                        self.assertEqual(exact.shape, (16, len(descriptors(q, 2, variant))))
                        self.assertEqual(exact.shape, interval.shape)
                        for rational, enclosure in zip(exact.flat, interval.flat):
                            self.assertTrue(enclosure.is_exact())
                            self.assertEqual(enclosure.lower(), rational)

    def test_implementation_error_retains_traceback(self):
        task = {'record': {'pair_id': 345, 'basis': 'chebyshev', 'variant': 'endpoint',
                           'q': 16, 'readout': 'mean', 'p': 4}}
        with patch('rebuttal_experiments.brec.evaluate_condition',
                   side_effect=TypeError('regression diagnostic')):
            record = condition_record(None, {'seconds_per_condition': None}, task)
        self.assertEqual(record['status'], 'implementation_error')
        self.assertIn('TypeError', record['reason'])
        self.assertIn('Traceback (most recent call last)', record['traceback'])
        self.assertIn('TypeError: regression diagnostic', record['traceback'])


if __name__ == '__main__':
    unittest.main()
