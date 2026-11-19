import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from brec.fwl2_encoding import _progressive_summary, compare_encodings, encode_pair_colours
from brec.fwl2_numeric import compare_numeric_encodings, encode_numeric_pairs, extend_numeric_pairs
from brec.fwl2_progressive import check_control, compare_progressively
from brec.suite_data import mechanism_graphs, two_fwl


def adjacency(graph):
    return nx.to_numpy_array(graph, dtype=np.int64)


class PairColourEncodingTests(unittest.TestCase):
    def test_canonical_colour_dictionaries_follow_a_relabeling(self):
        graph = adjacency(nx.house_graph())
        perm = [2, 4, 0, 1, 3]
        first = encode_pair_colours(graph)
        second = encode_pair_colours(graph[np.ix_(perm, perm)])
        self.assertFalse(compare_encodings(first, second)['different'])
        np.testing.assert_array_equal(first.colours[np.ix_(perm, perm)], second.colours)

    def test_joint_pivot_counts_separate_graphs_with_the_same_initial_counts(self):
        first = adjacency(nx.cycle_graph(6))
        second = adjacency(nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3)))
        encodings = [encode_pair_colours(graph) for graph in (first, second)]
        np.testing.assert_array_equal(*(encoding.rounds[0].counts for encoding in encodings))
        result = compare_encodings(*encodings)
        self.assertTrue(result['different'])
        self.assertEqual(result['round'], two_fwl(first, second)['rounds'])
        witness = result['witness']
        self.assertNotEqual(*witness['multiplicities'])
        self.assertEqual(sum(count for _, _, count in witness['joint_colour_counts']), len(first))

    def test_rook_and_shrikhande_have_equal_encodings(self):
        self.assertFalse(compare_encodings(
            *(encode_pair_colours(graph) for graph in mechanism_graphs())
        )['different'])

    def test_numeric_pair_channels_share_relabeling_and_depth_prefixes(self):
        graph = adjacency(nx.house_graph())
        perm = [2, 4, 0, 1, 3]
        full = encode_numeric_pairs(graph)
        shallow = encode_numeric_pairs(graph, rounds=2)
        relabeled = encode_numeric_pairs(graph[np.ix_(perm, perm)])
        self.assertEqual(full.shape, (5, 5, 11))
        np.testing.assert_array_equal(full[:, :, :5], shallow)
        np.testing.assert_array_equal(extend_numeric_pairs(shallow, 5), full)
        np.testing.assert_array_equal(full[np.ix_(perm, perm)], relabeled)
        self.assertFalse(compare_numeric_encodings(full, relabeled)['different'])

    def test_numeric_readout_uses_joint_counts(self):
        cycle = adjacency(nx.cycle_graph(6))
        triangles = adjacency(nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3)))
        result = compare_numeric_encodings(encode_numeric_pairs(cycle), encode_numeric_pairs(triangles))
        self.assertTrue(result['different'])
        self.assertEqual(result['round'], 1)
        self.assertNotEqual(*result['witness']['multiplicities'])

    def test_numeric_readout_agrees_on_rook_and_shrikhande(self):
        self.assertFalse(compare_numeric_encodings(
            *(encode_numeric_pairs(graph) for graph in mechanism_graphs())
        )['different'])


class ProgressiveEncodingTests(unittest.TestCase):
    def setUp(self):
        self.cycle = adjacency(nx.cycle_graph(6))
        self.triangles = adjacency(nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3)))
        self.options = dict(basis='monomial', variant='endpoint', max_rounds=5)

    def test_baseline_difference_skips_every_numerical_round(self):
        with patch('brec.fwl2_progressive.encode_numeric_pairs') as initialize:
            record, prefixes = compare_progressively(self.cycle, self.triangles, q=2, **self.options)
        initialize.assert_not_called()
        self.assertEqual((record['status'], record['round'], record['computed_rounds']),
                         ('certified_different', 0, 0))
        self.assertEqual(prefixes, [None, None])

    def test_unresolved_baseline_extends_once_before_stopping(self):
        with patch('brec.fwl2_progressive.extend_numeric_pairs', wraps=extend_numeric_pairs) as extend:
            record, prefixes = compare_progressively(self.cycle, self.triangles, q=1, **self.options)
        self.assertEqual([call.args[1] for call in extend.call_args_list], [1, 1])
        self.assertEqual((record['round'], record['computed_rounds']), (1, 1))
        self.assertNotEqual(*record['witness']['multiplicities'])
        for graph, prefix in zip((self.cycle, self.triangles), prefixes, strict=True):
            np.testing.assert_array_equal(prefix, encode_numeric_pairs(graph, rounds=1))
        summary = _progressive_summary([record], 5)
        self.assertEqual([layer['evaluated'] for layer in summary['layers']], [1, 1, 0, 0, 0, 0])
        self.assertEqual(summary['numerical_pair_rounds_computed'], 1)

    def test_matched_prefixes_remain_unresolved_at_depth_limit(self):
        record, prefixes = compare_progressively(*mechanism_graphs(), q=2, **self.options)
        self.assertEqual((record['status'], record['round'], record['computed_rounds']),
                         ('unresolved', None, 5))
        self.assertTrue(all(prefix.shape[-1] == 11 for prefix in prefixes))

    def test_official_control_uses_only_the_requested_prefix(self):
        graph = adjacency(nx.house_graph())
        permutation = [2, 4, 0, 1, 3]
        relabeling = graph[np.ix_(permutation, permutation)]
        with patch('brec.fwl2_progressive.encode_numeric_pairs', wraps=encode_numeric_pairs) as encode:
            check_control(graph, relabeling, permutation, q=3, basis='chebyshev', depth=0)
            encode.assert_not_called()
            check_control(graph, relabeling, permutation, q=3, basis='chebyshev', depth=2)
        self.assertEqual([call.kwargs['rounds'] for call in encode.call_args_list], [2, 2])
