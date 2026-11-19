from fractions import Fraction
from math import lcm
import unittest

import networkx as nx
import numpy as np

from brec.encodings import exact_fwl_pair
from brec.fwl4_encoding import baseline_rows, exact_four_fwl
from brec.fwl4_numeric import MODULI, atomic_types, extend_quadruples, pivot_profiles, quadruple_orbits


class QuadrupleEncodingTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.to_numpy_array(nx.house_graph(), dtype=np.int64)
        self.orbit = quadruple_orbits(self.graph)
        self.initial = atomic_types(self.graph, self.orbit.rows)

    def test_orbit_prefix_matches_literal_ordered_shared_pivot_update(self):
        np.testing.assert_array_equal(self.orbit.weights,
                                      np.bincount(self.orbit.lookup.ravel()))
        state = self.initial[self.orbit.lookup]
        result = extend_quadruples(self.orbit, self.initial[:, None], 1)
        np.testing.assert_array_equal(result[:, 0], self.initial)
        for j, modulus in enumerate(MODULI):
            shifts = (7853+104729+7919*j, 65537+130363+6511*j,
                      196613+171007+10427*j, 262147+196613+13007*j)
            expected = np.empty_like(state)
            for u, v, w, z in np.ndindex(state.shape):
                total = int(state[u, v, w, z])
                for x in range(len(self.graph)):
                    total += (
                        pow(int(state[x, v, w, z])+shifts[0], 5, modulus)
                        * pow(int(state[u, x, w, z])+shifts[1], 7, modulus)
                        * pow(int(state[u, v, x, z])+shifts[2], 11, modulus)
                        * pow(int(state[u, v, w, x])+shifts[3], 13, modulus))
                expected[u, v, w, z] = total % modulus
            np.testing.assert_array_equal(result[:, 1+j][self.orbit.lookup], expected)

    def test_reference_matches_dense_reference_after_relabeling(self):
        order = [2, 4, 0, 1, 3]
        changed = self.graph[np.ix_(order, order)]
        other = quadruple_orbits(changed)
        atomic_other = atomic_types(changed, other.rows)
        result = exact_four_fwl((self.orbit, other), (self.initial, atomic_other))
        self.assertEqual(result['different'], exact_fwl_pair(self.graph, changed, 4))
        self.assertIsNotNone(result.get('stable_after_round'))
        colours = self.initial*10000
        radix = int(colours.max())+1
        a, b, c, d = self.orbit.rows[0]
        joined = pivot_profiles(self.orbit.rows[:1], self.orbit.lookup, colours,
                                np.uint64(radix), False)[0]
        dense = colours[self.orbit.lookup]
        for x, (high, low) in enumerate(joined):
            self.assertEqual(divmod(int(high), radix),
                             (dense[x, b, c, d], dense[a, x, c, d]))
            self.assertEqual(divmod(int(low), radix),
                             (dense[a, b, x, d], dense[a, b, c, x]))

    def test_exact_joint_baseline_uses_common_denominator(self):
        degrees = self.graph.sum(axis=1).tolist()
        scale = lcm(*degrees)
        compact = baseline_rows(self.graph, self.orbit.rows, self.initial, scale)
        for row, packed in zip(self.orbit.rows, compact, strict=True):
            total = sum((Fraction(1, degrees[x]**2)
                         for x in range(len(self.graph)) if self.graph[row, x].all()),
                        Fraction())
            denominator = 1
            for vertex in row:
                denominator *= degrees[vertex]
            self.assertEqual(Fraction(int(packed[-1]), scale**8), total**2/denominator)

    def test_wider_joint_numerator_preserves_the_exact_feature(self):
        narrow_scale = lcm(*self.graph.sum(axis=1).tolist())
        wide_scale = 3060
        narrow = baseline_rows(self.graph, self.orbit.rows, self.initial, narrow_scale)
        wide = baseline_rows(self.graph, self.orbit.rows, self.initial, wide_scale)
        self.assertEqual(wide.shape[1], narrow.shape[1]+1)
        self.assertTrue(np.any(wide[:, -2]))
        for narrow_row, wide_row in zip(narrow, wide, strict=True):
            value = (int(wide_row[-2]) << 64) | int(wide_row[-1])
            self.assertEqual(Fraction(value, wide_scale**8),
                             Fraction(int(narrow_row[-1]), narrow_scale**8))
