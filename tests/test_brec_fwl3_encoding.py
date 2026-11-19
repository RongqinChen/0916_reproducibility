import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from brec.encodings import exact_fwl_pair
from brec.fwl2_encoding import compare_encodings, encode_pair_colours
from brec.fwl3_encoding import _check_official, compare_progressively, exact_three_fwl
from brec.fwl3_numeric import MODULI, atomic_types, extend_triples, triple_orbits
from brec.suite_data import mechanism_graphs


class TripleEncodingTests(unittest.TestCase):
    def test_orbit_reduced_contraction_matches_dense_integer_definition(self):
        graph = nx.to_numpy_array(nx.cycle_graph(5), dtype=np.int64)
        orbits = triple_orbits(graph)
        np.testing.assert_array_equal(orbits.weights, np.bincount(orbits.lookup.ravel()))
        self.assertEqual(int(orbits.weights.sum()), len(graph)**3)
        prefix = atomic_types(graph, orbits.rows)[:, None]
        dense = [prefix[:, 0][orbits.lookup]] * 2
        for depth in (1, 2):
            extended = extend_triples(orbits, prefix, depth)
            np.testing.assert_array_equal(extended[:, :prefix.shape[1]], prefix)
            for j, modulus in enumerate(MODULI):
                a = 7853 + 104729*depth + 7919*j
                b = 65537 + 130363*depth + 6511*j
                c = 196613 + 171007*depth + 10427*j
                prior = dense[j]
                next_values = np.empty_like(prior)
                for u, v, w in np.ndindex(prior.shape):
                    contracted = sum(
                        pow(int(prior[x, v, w]) + a, 5, modulus)
                        * pow(int(prior[u, x, w]) + b, 7, modulus)
                        * pow(int(prior[u, v, x]) + c, 11, modulus)
                        for x in range(len(graph)))
                    next_values[u, v, w] = (int(prior[u, v, w]) + contracted) % modulus
                np.testing.assert_array_equal(extended[:, -2+j][orbits.lookup], next_values)
                dense[j] = next_values
            prefix = extended

    def test_exact_reference_and_baseline_separate_rook_and_shrikhande(self):
        graphs = mechanism_graphs()
        self.assertFalse(compare_encodings(*(encode_pair_colours(g) for g in graphs))['different'])
        orbits = tuple(triple_orbits(graph) for graph in graphs)
        atomic = tuple(atomic_types(graph, orbit.rows) for graph, orbit in zip(graphs, orbits))
        reference = exact_three_fwl(orbits, atomic)
        self.assertEqual(reference['different'], exact_fwl_pair(*graphs, 3))
        self.assertEqual(reference['round'], 1)
        with patch('brec.fwl3_numeric.extend_triples') as extend:
            result, prefixes = compare_progressively(graphs, orbits, atomic, 5)
            extend.assert_not_called()
        self.assertEqual((result['round'], result['computed_rounds']), (0, 0))
        self.assertEqual(prefixes, [None, None])

    def test_reference_and_numerical_channels_are_equivariant(self):
        graph = nx.to_numpy_array(nx.house_graph(), dtype=np.int64)
        order = [2, 4, 0, 1, 3]
        relabelled = graph[np.ix_(order, order)]
        graphs = (graph, relabelled)
        orbits = tuple(triple_orbits(g) for g in graphs)
        atomic = tuple(atomic_types(g, o.rows) for g, o in zip(graphs, orbits))
        self.assertFalse(exact_three_fwl(orbits, atomic)['different'])
        result, prefixes = compare_progressively(graphs, orbits, atomic, 2)
        self.assertEqual((result['status'], result['computed_rounds']), ('unresolved', 2))
        first = prefixes[0][orbits[0].lookup]
        np.testing.assert_array_equal(first[np.ix_(order, order, order)],
                                      prefixes[1][orbits[1].lookup])
        _check_official(graphs, 2, {})
