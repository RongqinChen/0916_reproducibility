"""Exact radicals, orbit multiplicities and comparison against full enumeration."""
from collections import Counter
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import networkx as nx
import numpy as np

from rebuttal_experiments.common import sha256
from rebuttal_experiments.encodings import constructed_graphs, descriptors, feature_matrix, moments
from rebuttal_experiments.readouts import exact_multiset, mean_decision
from rebuttal_experiments.triple_encoding_compare import compare_kernels
from rebuttal_experiments.triple_encoding_controls import aligned_kernel_control
from rebuttal_experiments.triple_exact import (ONE, ZERO, Radical, TripleKernel,
                                              ordered_triple_orbits, weighted_counts, weighted_moment)

ROOT = Path(__file__).resolve().parents[1]


class TripleExactTests(unittest.TestCase):
    def test_radical_arithmetic_is_canonical_and_exact(self):
        root2 = Radical.make(((2,1),))
        self.assertEqual(Radical.make(((8,1),)),root2.scaled(2))
        self.assertEqual(root2*root2,Radical.integer(2))
        self.assertEqual((ONE+root2)**2,Radical.integer(3)+root2.scaled(2))
        self.assertEqual((ONE+root2)*(ONE-root2),Radical.integer(-1))
        self.assertEqual((ONE+root2)**16,(ONE+root2)**8*(ONE+root2)**8)
        self.assertEqual(root2-root2,ZERO)

    def test_all_features_and_sums_against_dense_fraction_backend(self):
        for graph in (nx.cycle_graph(4),nx.complete_graph(4),nx.empty_graph(3)):
            adjacency = nx.to_numpy_array(graph,dtype='int64')
            triples = list(np.ndindex((len(graph),)*3))
            for basis in ('monomial','chebyshev'):
                for q in (1,2,4):
                    kernel = TripleKernel(adjacency,q,basis)
                    expected = feature_matrix(adjacency,q,3,'joint',basis,'exact')
                    for column,key in enumerate(descriptors(q,3,'joint')):
                        actual = kernel.column(triples,key)
                        self.assertEqual(actual,[Radical.make(((1,x),)) for x in expected[:,column]])
                        self.assertEqual(kernel.mean_sum(key),sum(actual,ZERO))

    def test_irrational_features_against_independent_sympy_matrix_powers(self):
        import sympy as sp
        adjacency = nx.to_numpy_array(nx.path_graph(3),dtype='int64')
        degree = adjacency.sum(1).tolist()
        matrix = sp.Matrix([[sp.Integer(int(adjacency[i,j]))/sp.sqrt(degree[i]*degree[j])
                             for j in range(3)] for i in range(3)])
        bank = [sp.eye(3),matrix,matrix*matrix]
        kernel = TripleKernel(adjacency,2,'monomial')
        def scalar(value):
            return sum(sp.Rational(c.numerator,c.denominator)*sp.sqrt(s) for s,c in value.terms)
        for triple in ((0,0,0),(0,1,2),(1,1,1),(2,0,1)):
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        expected = sum(bank[a][triple[0],t]*bank[b][triple[1],t]*bank[c][triple[2],t]
                                       for t in range(3))
                        self.assertEqual(sp.simplify(scalar(kernel.value(triple,('joint',a,b,c)))-expected),0)

    def test_orbit_weighted_histograms_equal_full_ordered_enumeration(self):
        for graph in (nx.path_graph(4),nx.star_graph(4),nx.cycle_graph(4),nx.complete_graph(4)):
            adjacency = nx.to_numpy_array(graph,dtype='int64')
            representatives,weights = ordered_triple_orbits(adjacency)
            self.assertEqual(int(weights.sum()),len(graph)**3)
            kernel = TripleKernel(adjacency,2,'chebyshev')
            keys = descriptors(2,3,'joint')
            full = Counter(tuple(kernel.value(triple,key) for key in keys)
                           for triple in np.ndindex((len(graph),)*3))
            compressed = weighted_counts([tuple(kernel.value(triple,key) for key in keys)
                                          for triple in representatives],weights)
            self.assertEqual(full,compressed)

    def test_mixed_degree_q4_and_high_moments_against_sympy(self):
        import sympy as sp
        adjacency = nx.to_numpy_array(nx.Graph([(0,1),(1,2),(2,0),(0,3)]),dtype='int64')
        degree = adjacency.sum(1).tolist()
        matrix = sp.Matrix([[sp.Integer(int(adjacency[i,j]))/sp.sqrt(degree[i]*degree[j])
                             for j in range(4)] for i in range(4)])
        triples,weights = ordered_triple_orbits(adjacency)
        def scalar(value):
            return sum(sp.Rational(c.numerator,c.denominator)*sp.sqrt(s) for s,c in value.terms)
        for basis in ('monomial','chebyshev'):
            bank = [sp.eye(4),matrix]
            for _ in range(2,5):
                bank.append(matrix*bank[-1] if basis=='monomial' else 2*matrix*bank[-1]-bank[-2])
            kernel = TripleKernel(adjacency,4,basis)
            keys = [('joint',1,2,4),('joint',0,2,4),('joint',4,4,4),('endpoint',1,2,0,1)]
            for key in keys:
                def expected(triple):
                    u,v,w = map(int,triple)
                    if key[0]=='endpoint':
                        return sp.expand(sum(bank[1][u,t] for t in range(4))*
                                         sum(bank[2][v,t] for t in range(4)))
                    a,b,c = key[1:]
                    return sp.expand(sum(bank[a][u,t]*bank[b][v,t]*bank[c][w,t] for t in range(4)))
                values = [expected(triple) for triple in triples]
                actual = kernel.column(triples,key)
                for triple,left,right in zip(triples,actual,values):
                    self.assertEqual(sp.expand(scalar(left)-right),0)
                    self.assertEqual(kernel.value(triple,key),left)
                full_sum = sum(expected(triple) for triple in np.ndindex((4,4,4)))
                self.assertEqual(sp.expand(scalar(kernel.mean_sum(key))-full_sum),0)
                for power in (2,4,8,16):
                    oracle = sum(sp.expand(value**power)*int(weight) for value,weight in zip(values,weights))
                    self.assertEqual(sp.expand(scalar(weighted_moment(actual,weights,power))-oracle),0)

    def test_large_integer_joint_avoids_native_integer_overflow(self):
        graph = nx.gnp_random_graph(20,0.4,seed=4)
        kernel = TripleKernel(nx.to_numpy_array(graph,dtype='int64'),4,'chebyshev')
        key = ('joint',4,4,4)
        self.assertGreater(kernel.n*kernel.max_abs[4]**3,np.iinfo('int64').max)
        triples = [(0,1,2),(3,3,3),(0,4,19)]
        self.assertEqual(kernel.column(triples,key),[kernel.value(triple,key) for triple in triples])
        self.assertEqual(kernel.integer_banks,{})

    def test_comparisons_match_dense_multisets_and_cumulative_moments(self):
        graphs = [nx.cycle_graph(6),nx.disjoint_union(nx.cycle_graph(3),nx.cycle_graph(3))]
        adjs = [nx.to_numpy_array(graph,dtype='int64') for graph in graphs]
        compressed = [ordered_triple_orbits(a) for a in adjs]
        triples,weights = list(zip(*compressed))
        for basis in ('monomial','chebyshev'):
            kernels = [TripleKernel(a,2,basis) for a in adjs]
            result = compare_kernels(kernels,triples,weights,2,basis,[1,2,4,8,16])
            for variant in ('baseline','joint'):
                rows = [feature_matrix(a,2,3,'endpoint' if variant=='baseline' else 'joint',basis,'exact')
                        for a in adjs]
                self.assertEqual(result[variant]['multiset']['status'],exact_multiset(*rows)['status'])
                for p in (1,2,4,8,16):
                    self.assertEqual(result[variant][f'mean_p{p}']['status'],
                                     mean_decision(*(moments(x,p) for x in rows))['status'])

    def test_joint_distinguishes_when_baseline_multiset_is_equal(self):
        adjs = list(constructed_graphs())
        compressed = [ordered_triple_orbits(a) for a in adjs]
        triples,weights = list(zip(*compressed))
        for basis in ('monomial','chebyshev'):
            result = compare_kernels([TripleKernel(a,1,basis) for a in adjs],
                                     triples,weights,1,basis,[1,2,4,8,16])
            for variant in ('baseline','joint'):
                rows = [feature_matrix(a,1,3,'endpoint' if variant=='baseline' else 'joint',
                                       basis,'exact') for a in adjs]
                self.assertEqual(result[variant]['multiset']['status'],exact_multiset(*rows)['status'])
            self.assertEqual(result['baseline']['multiset']['status'],'certified_equal')
            self.assertEqual(result['joint']['multiset']['status'],'certified_different')

    def test_nonlinear_controls_use_correct_noninvolutory_mapping(self):
        adjacency = nx.to_numpy_array(nx.path_graph(4),dtype='int64')
        permutation = [2,0,3,1]
        changed = adjacency[np.ix_(permutation,permutation)]
        result = aligned_kernel_control(TripleKernel(adjacency,2,'chebyshev'),
                    TripleKernel(changed,2,'chebyshev'),np.argsort(permutation),
                    [(0,1,2),(1,1,3)],descriptors(2,3,'joint'))
        self.assertTrue(result['all_bank_entries_exact_equal'])
        self.assertTrue(result['sampled_full_rows_exact_equal'])

    def test_encoding_only_cli_and_partial_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = nx.to_graph6_bytes(nx.path_graph(3),header=False).strip()
            other = nx.to_graph6_bytes(nx.cycle_graph(3),header=False).strip()
            data = np.full(51200,graph,dtype='S16')
            data[1] = other
            source = root/'brec.npy'
            np.save(source,data,allow_pickle=False)
            config = json.loads((ROOT/'configs/rebuttal/triple_encodings.json').read_text())
            del config['mean_p_values']
            config.update(expected_brec_sha256=sha256(source),q_values=[1,2],
                          mean_p_values_by_q={'1':[1,2,4],'2':[1,2]},
                          relabels=1,reliability_relabels=0,control_triples=2,memory_gib=4)
            config_path = root/'config.json'
            config_path.write_text(json.dumps(config))
            command = [sys.executable,'-m','rebuttal_experiments','triple-encodings',
                       '--brec-file',str(source),'--config',str(config_path),'--out',str(root/'run'),
                       '--pair-ids','0','--workers','1','--reserve-gib','0']
            env = dict(os.environ,PYTHONHASHSEED='0')
            result = subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            records = list((root/'run/records').glob('*.json'))
            self.assertEqual(len(records),28)
            self.assertEqual(sum('_q1_mean_p4_' in path.name for path in records),4)
            self.assertFalse(any('_q2_mean_p4_' in path.name for path in records))
            self.assertTrue(all(json.loads(path.read_text())['status'] in
                                ('certified_different','certified_equal') for path in records))
            before = {path.name:path.read_bytes() for path in records}
            records[0].unlink()
            resumed = subprocess.run(command+['--resume'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
            self.assertEqual(resumed.returncode,0,resumed.stdout+resumed.stderr)
            for path in records[1:]:
                self.assertEqual(path.read_bytes(),before[path.name])
            summary = json.loads((root/'run/summary.json').read_text())
            self.assertEqual(len(summary['settings']),28)
            self.assertTrue(all(sum(row['counts'].values())==400 for row in summary['settings']))
            row = json.loads(records[0].read_text())
            row['q'] += 1
            records[0].write_text(json.dumps(row))
            invalid = subprocess.run(command+['--resume'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
            self.assertNotEqual(invalid.returncode,0)
            self.assertIn('record condition or identity mismatch',invalid.stdout+invalid.stderr)


if __name__=='__main__':
    unittest.main()
