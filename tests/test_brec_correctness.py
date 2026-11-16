"""Independent arithmetic and classification oracles for supplementary BREC."""
from collections import Counter
from fractions import Fraction
import itertools
from math import isqrt
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from rebuttal_experiments.brec import (aligned_control, pending_conditions, reliability_control,
                                       reliability_mapping, validate_configuration)
from rebuttal_experiments.common import Budget
from rebuttal_experiments.encodings import feature_matrix, matmul, moments
from rebuttal_experiments.intervals import Interval
from rebuttal_experiments.readouts import (
    DIFFERENT, EQUAL, UNRESOLVED, exact_multiset, interval_multiset, mean_decision)


def multiply(left, right):
    n = len(left)
    return [[sum((left[i][k] * right[k][j] for k in range(n)), Fraction(0))
             for j in range(n)] for i in range(n)]


def oracle_bank(adjacency, q, basis):
    """Pure Fraction oracle: evaluate polynomial coefficients on matrix powers."""
    n = len(adjacency)
    degrees = [int(sum(row)) for row in adjacency]
    eye = [[Fraction(i == j) for j in range(n)] for i in range(n)]
    matrix = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for i, j in itertools.product(range(n), repeat=2):
        if adjacency[i, j]:
            denominator = isqrt(degrees[i] * degrees[j])
            assert denominator**2 == degrees[i] * degrees[j]
            matrix[i][j] = Fraction(1, denominator)
    powers = [eye]
    for _ in range(q):
        powers.append(multiply(powers[-1], matrix))
    if basis == 'monomial':
        return powers
    if basis == 'shifted':
        assert q == 1
        return [eye, [[eye[i][j] + matrix[i][j] for j in range(n)] for i in range(n)]]
    coefficients = [[1], [0, 1]]
    for degree in range(2, q + 1):
        current = [0] + [2 * value for value in coefficients[-1]]
        for index, value in enumerate(coefficients[-2]):
            current[index] -= value
        coefficients.append(current)
    return [[[sum((coefficient * powers[power][i][j]
                   for power, coefficient in enumerate(coefficients[degree])), Fraction(0))
              for j in range(n)] for i in range(n)] for degree in range(q + 1)]


def oracle_rows(adjacency, bank, q, order, variant):
    n = len(adjacency)
    positions = [(0, 1)] if order == 2 else list(itertools.permutations(range(3), 2))
    h = [[sum(row, Fraction(0)) for row in matrix] for matrix in bank[:q + 1]]
    result = []
    for coordinates in itertools.product(range(n), repeat=order):
        row = [Fraction(1)]
        matrices = [bank[0], adjacency, *bank[1:q + 1]]
        row += [(matrix[coordinates[i]][coordinates[j]]
                 if isinstance(matrix[coordinates[i]][coordinates[j]], Fraction)
                 else Fraction(int(matrix[coordinates[i]][coordinates[j]])))
                for matrix in matrices for i, j in positions]
        if variant != 'polynomial':
            row += [h[a][coordinates[i]] * h[b][coordinates[j]]
                    for i, j in positions for a in range(q) for b in range(q - a)
                    if a + b > 0]
        if variant == 'joint':
            u, v, w = coordinates
            row += [sum((bank[a][u][t] * bank[b][v][t] * bank[c][w][t]
                         for t in range(n)), Fraction(0))
                    for a, b, c in itertools.product(range(q + 1), repeat=3)]
        result.append(row)
    return result


class IntervalArithmeticTests(unittest.TestCase):
    def assert_encloses(self, interval, value):
        self.assertLessEqual(interval.lower(), value)
        self.assertGreaterEqual(interval.upper(), value)

    def test_primitives_against_exact_rationals(self):
        rng = random.Random(7421)
        for bits in (8, 128, 256, 512):
            for _ in range(80):
                a = Fraction(rng.randrange(-1000, 1001), rng.randrange(1, 100))
                b = Fraction(rng.randrange(-1000, 1001), rng.randrange(1, 100))
                left, right = Interval.exact(a, bits), Interval.exact(b, bits)
                self.assert_encloses(left, a)
                for interval, expected in ((left + right, a + b), (left - right, a - b),
                                           (left * right, a * b), (-left, -a)):
                    self.assert_encloses(interval, expected)
                for power in range(5):
                    self.assert_encloses(left**power, a**power)
                if not right.contains_zero():
                    self.assert_encloses(left / right, a / b)
                    self.assert_encloses(right.reciprocal(), 1 / b)
        with self.assertRaises(ZeroDivisionError):
            Interval(-1, 1, 128).reciprocal()

    def test_square_roots_have_exact_integer_bounds(self):
        for bits in (8, 128, 512):
            for value in (*range(100), 2**100 - 1, 2**100, 2**100 + 1):
                result = Interval.sqrt_integer(value, bits)
                self.assertLessEqual(result.lower()**2, value)
                self.assertGreaterEqual(result.upper()**2, value)

    def test_wide_intervals_enclose_endpoint_extrema(self):
        rng = random.Random(1753)
        for _ in range(100):
            lo, hi = sorted(rng.sample(range(-100, 101), 2))
            other_lo, other_hi = sorted(rng.sample(range(-100, 101), 2))
            left, right = Interval(lo, hi, 8), Interval(other_lo, other_hi, 8)
            for a, b in itertools.product((left.lower(), left.upper()),
                                           (right.lower(), right.upper())):
                self.assert_encloses(left + right, a + b)
                self.assert_encloses(left - right, a - b)
                self.assert_encloses(left * right, a * b)
                if not right.contains_zero():
                    self.assert_encloses(left / right, a / b)
            values = [left.lower(), left.upper()]
            if left.contains_zero():
                values.append(Fraction(0))
            for power in range(5):
                for value in values:
                    self.assert_encloses(left**power, value**power)


class IndependentEncodingTests(unittest.TestCase):
    def test_sparse_matmul_matches_dense_endpoints_exactly(self):
        rng = random.Random(903)
        for mode in ('fraction', 'interval'):
            for bits in (8, 128, 512):
                left = np.empty((7, 9), dtype=object)
                right = np.empty((9, 5), dtype=object)
                make = Fraction if mode == 'fraction' else lambda x: Interval.exact(x, bits)
                for i, j in np.ndindex(left.shape):
                    left[i, j] = make(Fraction(rng.randrange(-5, 6), 3) if (
                        i > 0 and (i >= 4 or j % 4 == 0)) else 0)
                for i, j in np.ndindex(right.shape):
                    right[i, j] = make(Fraction(rng.randrange(-5, 6), 7))
                if mode == 'interval':
                    # An interval containing zero must still participate.
                    left[1, 1] = Interval(-2, 3, bits)
                    right[1, 3] = Interval(-7, 9, bits)
                original_left, original_right = left.copy(), right.copy()
                expected = left @ right
                actual = matmul(left, right)
                self.assertTrue(np.array_equal(actual, expected))
                self.assertTrue(np.array_equal(left, original_left))
                self.assertTrue(np.array_equal(right, original_right))
                self.assertTrue(np.array_equal(matmul(left, right[:, :1]), expected[:, :1]))
        self.assertTrue(np.array_equal(matmul(np.empty((3, 0), dtype=object),
                                              np.empty((0, 4), dtype=object)),
                                       np.zeros((3, 4))))

    def test_all_budgets_and_moments_against_fraction_oracle(self):
        graphs = [nx.star_graph(4), nx.disjoint_union(nx.cycle_graph(3), nx.path_graph(2)),
                  nx.complete_graph(4), nx.complete_bipartite_graph(3, 3)]
        for graph in graphs:
            graph.add_nodes_from(range(6))
            adjacency = nx.to_numpy_array(graph, nodelist=range(6), dtype=np.int64)
            for basis in ('monomial', 'chebyshev'):
                bank = oracle_bank(adjacency, 16, basis)
                for variant, q, bits in itertools.product(
                        ('polynomial', 'endpoint'), (2, 4, 8, 16), (128, 512)):
                    with self.subTest(edges=graph.number_of_edges(), basis=basis,
                                      variant=variant, q=q, bits=bits):
                        expected = oracle_rows(adjacency, bank, q, 2, variant)
                        actual = feature_matrix(adjacency, q, 2, variant, basis, 'interval', bits)
                        self.assertEqual(actual.shape, (len(expected), len(expected[0])))
                        for enclosure, rational in zip(actual.flat, itertools.chain.from_iterable(expected)):
                            self.assertLessEqual(enclosure.lower(), rational)
                            self.assertGreaterEqual(enclosure.upper(), rational)
                        width = len(expected[0])
                        actual_moments = moments(actual, 4)
                        for power, column in itertools.product(range(1, 5), range(width)):
                            value = sum((row[column]**power for row in expected), Fraction(0)) / len(expected)
                            enclosure = actual_moments[(power - 1) * width + column]
                            self.assertLessEqual(enclosure.lower(), value)
                            self.assertGreaterEqual(enclosure.upper(), value)

    def test_triple_joint_and_shifted_features_against_oracle(self):
        adjacency = nx.to_numpy_array(nx.cycle_graph(3), dtype=np.int64)
        for basis, q in (('monomial', 2), ('chebyshev', 2), ('shifted', 1)):
            bank = oracle_bank(adjacency, q, basis)
            for variant in ('polynomial', 'endpoint', 'joint'):
                expected = oracle_rows(adjacency, bank, q, 3, variant)
                actual = feature_matrix(adjacency, q, 3, variant, basis, 'exact')
                self.assertEqual(actual.tolist(), expected)

    def test_non_involutory_relabel_alignment(self):
        adjacency = nx.to_numpy_array(nx.cycle_graph(6), dtype=np.int64)
        permutation = np.array([2, 0, 4, 5, 1, 3])
        self.assertFalse(np.array_equal(permutation[permutation], np.arange(6)))
        changed = adjacency[np.ix_(permutation, permutation)]
        for basis, mode in itertools.product(('monomial', 'chebyshev'), ('exact', 'interval')):
            original = feature_matrix(adjacency, 4, 2, 'endpoint', basis, mode)
            relabelled = feature_matrix(changed, 4, 2, 'endpoint', basis, mode)
            aligned_control(original, relabelled, permutation, mode)
            relabelled[0, 0] = Fraction(99) if mode == 'exact' else Interval.exact(99, 128)
            with self.assertRaises(AssertionError):
                aligned_control(original, relabelled, permutation, mode)

    def test_official_reliability_mapping_direction(self):
        adjacency = nx.to_numpy_array(nx.cycle_graph(6), dtype=np.int64)
        permutation = np.array([2, 0, 4, 5, 1, 3])
        changed = adjacency[np.ix_(permutation, permutation)]
        self.assertFalse(np.array_equal(adjacency, changed))
        old_to_new = np.argsort(permutation)
        original6 = nx.to_graph6_bytes(nx.from_numpy_array(adjacency), header=False).strip()
        changed6 = nx.to_graph6_bytes(nx.from_numpy_array(changed), header=False).strip()
        data = np.full(51200, original6, dtype='S16')
        data[400 * 64 + 1] = changed6
        config = {'seed': 0, 'reliability_relabels': 2}
        with patch('pynauty.canon_label', side_effect=[list(range(6)), old_to_new.tolist()]):
            checks = reliability_control(data, 0, 4, 'chebyshev', 'endpoint',
                                         'exact', 128, config, Budget(None, 3))
        self.assertEqual(checks[0]['control_old_to_new'], old_to_new.tolist())
        self.assertEqual(len(checks), 3)
        self.assertTrue(checks[0]['official_representatives']['aligned_exact_equal'])

    def test_reliability_mapping_is_verified_for_multiple_graph_families(self):
        rng = random.Random(913)
        graphs = [nx.empty_graph(6), nx.path_graph(6), nx.cycle_graph(6),
                  nx.star_graph(5), nx.complete_graph(6), nx.complete_bipartite_graph(3, 3)]
        for graph in graphs:
            adjacency = nx.to_numpy_array(graph, dtype=np.int64)
            for _ in range(4):
                permutation = list(range(len(graph)))
                rng.shuffle(permutation)
                changed = adjacency[np.ix_(permutation, permutation)]
                # Falling back to VF2 here reintroduces an unbounded symmetric-graph search.
                with patch('networkx.algorithms.isomorphism.GraphMatcher',
                           side_effect=AssertionError('VF2 must not be used for control alignment')):
                    mapping = reliability_mapping(adjacency, changed)
                self.assertEqual(sorted(mapping.tolist()), list(range(len(graph))))
                inverse = np.argsort(mapping)
                self.assertTrue(np.array_equal(adjacency[np.ix_(inverse, inverse)], changed))

    def test_nonisomorphic_reliability_graphs_are_rejected(self):
        # Same node/edge count and degree sequence; a permutation cannot align them.
        first = nx.to_numpy_array(nx.cycle_graph(6), dtype=np.int64)
        second = nx.to_numpy_array(nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3)), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, 'not isomorphic'):
            reliability_mapping(first, second)

    def test_bad_canonical_orders_are_rejected(self):
        adjacency = nx.to_numpy_array(nx.path_graph(4), dtype=np.int64)
        with patch('pynauty.canon_label', return_value=[0, 0, 1, 2]):
            with self.assertRaisesRegex(ValueError, 'canonical vertex ordering'):
                reliability_mapping(adjacency, adjacency)

    def test_preflight_checks_controls_without_testing_main_pair_isomorphism(self):
        first, other = [nx.to_graph6_bytes(graph, header=False).strip() for graph in (
            nx.cycle_graph(6), nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3)))]
        data = np.full(51200, first, dtype='S16')
        data[1] = other  # A valid nonisomorphic main pair must still be scheduled.
        config = {'bases': ['monomial'], 'variants': ['polynomial'], 'q_values': [2],
                  'mean_p_values': [1], 'max_candidate_edges': 10000}
        with tempfile.TemporaryDirectory() as directory:
            tasks = list(pending_conditions(data, [0], config, Path(directory), 'test'))
            self.assertEqual(len(tasks), 2)
            data[400 * 64 + 1] = other  # Nonisomorphic control must fail before scheduling.
            pending = pending_conditions(data, [0], config, Path(directory), 'test')
            with self.assertRaisesRegex(ValueError, 'Reliability preflight failed for pair000'):
                next(pending)


class ReadoutSoundnessTests(unittest.TestCase):
    def test_multisets_preserve_multiplicities(self):
        self.assertEqual(exact_multiset([[1], [1], [2]], [[1], [2], [2]])['status'], DIFFERENT)
        self.assertEqual(exact_multiset([[1], [1], [2]], [[2], [1], [1]])['status'], EQUAL)

    def test_matching_uses_row_capacities(self):
        a, b = Interval(0, 2, 8), Interval(6, 8, 8)
        c, d = Interval(1, 3, 8), Interval(7, 9, 8)
        with patch('rebuttal_experiments.readouts.projection_decision', return_value=None):
            different = interval_multiset([[a], [a], [b]], [[c], [d], [d]])
            self.assertEqual(different['status'], DIFFERENT)
            self.assertEqual(different['evidence']['matched_objects'], 2)
            self.assertEqual(different['evidence']['required_objects'], 3)
            unresolved = interval_multiset([[a], [a], [b]], [[c], [c], [d]])
            self.assertEqual(unresolved['status'], UNRESOLVED)
            limited = interval_multiset([[a], [b]], [[c], [d]], max_candidate_edges=1)
            self.assertEqual(limited['reason'], 'matching_edge_limit')

    def test_interval_decisions_never_overstate_rational_truth(self):
        rng = random.Random(823)
        for _ in range(100):
            left = [[Fraction(rng.randrange(-4, 5), 3) for _ in range(3)] for _ in range(4)]
            right = [row[:] for row in reversed(left)] if rng.randrange(2) else [
                [Fraction(rng.randrange(-4, 5), 3) for _ in range(3)] for _ in range(4)]
            first = [[Interval.exact(x, 8) for x in row] for row in left]
            second = [[Interval.exact(x, 8) for x in row] for row in right]
            exact_equal = Counter(map(tuple, left)) == Counter(map(tuple, right))
            decision = interval_multiset(first, second)
            if decision['status'] == DIFFERENT:
                self.assertFalse(exact_equal)
            elif decision['status'] == EQUAL:
                self.assertTrue(exact_equal)
            exact_means = [[sum(row[column] for row in values) / len(values)
                            for column in range(3)] for values in (left, right)]
            interval_means = [[sum(row[column] for row in values) / len(values)
                               for column in range(3)] for values in (first, second)]
            result = mean_decision(*interval_means)
            if result['status'] == DIFFERENT:
                self.assertNotEqual(*exact_means)
            elif result['status'] == EQUAL:
                self.assertEqual(*exact_means)


class ConfigurationTests(unittest.TestCase):
    def test_invalid_protocols_fail_early(self):
        from pathlib import Path
        import json
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/rebuttal/brec.json').read_text())
        validate_configuration(config)
        invalid = {'bases': [[], ['invalid'], ['monomial', 'monomial']],
                   'variants': [[], ['joint']], 'q_values': [[], [0], [2, 2], [True]],
                   'mean_p_values': [[], [-1]], 'precision_bits': [[], [1], [256, 128]],
                   'relabels': [0, 1.5], 'reliability_relabels': [-1], 'seed': [None],
                   'max_candidate_edges': [0], 'memory_gib': [0, float('inf'), True],
                   'seconds_per_condition': [0, -1, float('nan'), 'unlimited']}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_configuration({**config, key: value})


if __name__ == '__main__':
    unittest.main()
