"""Regression checks for CPU exact and interval projection optimizations."""
from fractions import Fraction
import itertools
import random
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from rebuttal_experiments.encodings import (feature_matrix, moment_blocks,
                                           scaled_exact_feature_matrix)
from rebuttal_experiments.intervals import Interval, number_json
from rebuttal_experiments.readouts import (exact_multiset, mean_decision,
                                           interval_multiset, projection_decision)
from rebuttal_experiments.scaled_exact import (ScaledIntegerArray,
                                               scaled_array_equal,
                                               scaled_mean_decision)


def legacy_projection(left, right, max_cuts=32):
    """The pre-optimization implementation, retained as a test oracle."""
    width = len(left[0])
    probes = [[int(i == j) for i in range(width)] for j in range(width)]
    probes += [[(-1 if (i + probe) % 2 else 1) * (1 + (i * (probe + 1)) % 7)
                for i in range(width)] for probe in range(4)]
    for weights in probes:
        first = [sum((x * w for x, w in zip(row, weights) if w), 0) for row in left]
        second = [sum((x * w for x, w in zip(row, weights) if w), 0) for row in right]
        centers = sorted(set(x.midpoint() for x in first + second))
        count = min(max_cuts, max(0, len(centers) - 1))
        cuts = [Fraction(0)]
        for index in range(count):
            position = index * (len(centers) - 1) // count
            cuts.append((centers[position] + centers[position + 1]) / 2)
        for cut in cuts:
            a = (sum(x.upper() < cut for x in first),
                 sum(x.lower() < cut for x in first))
            b = (sum(x.upper() < cut for x in second),
                 sum(x.lower() < cut for x in second))
            if a[0] > b[1] or b[0] > a[1]:
                return {
                    'status': 'certified_different',
                    'evidence': {
                        'kind': 'projection_count',
                        'weights': weights,
                        'threshold': number_json(cut),
                        'left_count_bounds': list(a),
                        'right_count_bounds': list(b),
                    },
                }
    return None


class ExactIntegerStorageTests(unittest.TestCase):
    def test_scaled_features_match_fraction_reference(self):
        adjacency = nx.to_numpy_array(nx.cycle_graph(6), dtype=np.int64)
        for basis in ('monomial', 'chebyshev'):
            for variant in ('polynomial', 'endpoint'):
                for order, q in ((2, 4), (3, 2)):
                    with self.subTest(basis=basis, variant=variant, order=order, q=q):
                        expected = feature_matrix(
                            adjacency, q, order, variant, basis, 'exact')
                        actual = scaled_exact_feature_matrix(
                            adjacency, q, order, variant, basis, 'exact')
                        self.assertIsInstance(actual, ScaledIntegerArray)
                        converted = np.vectorize(
                            lambda value: Fraction(int(value), actual.denominator),
                            otypes=[object])(np.asarray(actual))
                        self.assertTrue(np.array_equal(expected, converted))

    def test_scaled_multiset_normalizes_different_denominators(self):
        half = ScaledIntegerArray([[1], [2]], 2)
        quarter = ScaledIntegerArray([[2], [4]], 4)
        self.assertEqual(exact_multiset(half, quarter)['status'], 'certified_equal')
        self.assertTrue(scaled_array_equal(half, quarter))

    def test_scaled_mean_decision_matches_fraction_path(self):
        left_rows = ScaledIntegerArray([[1, 3], [2, 5]], 4)
        right_rows = ScaledIntegerArray([[2, 6], [4, 10]], 8)
        left_moments = [
            ScaledIntegerArray(block, left_rows.denominator**power * len(left_rows))
            for power, block in enumerate(itertools.islice(moment_blocks(left_rows), 2), 1)
        ]
        right_moments = [
            ScaledIntegerArray(block, right_rows.denominator**power * len(right_rows))
            for power, block in enumerate(itertools.islice(moment_blocks(right_rows), 2), 1)
        ]
        actual = scaled_mean_decision(tuple(zip(left_moments, right_moments)), range(2))
        expected = mean_decision(
            [Fraction(int(value), block.denominator)
             for block in left_moments for value in block],
            [Fraction(int(value), block.denominator)
             for block in right_moments for value in block],
        )
        self.assertEqual(actual, expected)


class ProjectionOptimizationTests(unittest.TestCase):
    def test_equal_interval_multiset_skips_projection(self):
        value = Interval(1, 3, 32)
        with patch('rebuttal_experiments.readouts.projection_decision',
                   side_effect=AssertionError('projection should be skipped')):
            result = interval_multiset([[value], [value]], [[value], [value]])
        self.assertEqual(result['status'], 'unresolved')
        self.assertEqual(result['reason'], 'compatible_perfect_matching_exists')

    def test_weighted_unique_projection_matches_legacy_oracle(self):
        rng = random.Random(20261119)
        for case in range(50):
            width = rng.randrange(1, 8)
            length = rng.randrange(2, 25)
            bits = rng.choice((8, 16, 128))

            def rows():
                result = []
                for _ in range(length):
                    row = []
                    for _ in range(width):
                        lower = rng.randrange(-20, 20)
                        row.append(Interval(lower, lower + rng.randrange(0, 8), bits))
                    result.append(row)
                return result

            left = rows()
            right = rows()
            if case % 3 == 0:
                right = [row[:] for row in left]
            with self.subTest(case=case):
                self.assertEqual(
                    projection_decision(
                        np.asarray(left, dtype=object),
                        np.asarray(right, dtype=object),
                    ),
                    legacy_projection(left, right),
                )


if __name__ == '__main__':
    unittest.main()
