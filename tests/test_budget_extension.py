"""Requested budget coverage, mean-only execution, and sequential supervision."""
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from brec.reproducibility import snapshot
from rebuttal_experiments.brec import (evaluate_condition, experiment_settings,
                                       reliability_control, validate_configuration)
from rebuttal_experiments.common import Budget, sha256
from rebuttal_experiments.encodings import polynomial_bank
from rebuttal_experiments import polynomial_banks
from rebuttal_experiments.triple_encodings import settings as triple_settings, validate as validate_triples
from rebuttal_experiments.triple_encoding_controls import controls as triple_controls
from rebuttal_experiments.triple_exact import TripleKernel
from tools import rebuttal_budget_extension as extension
from tools.rebuttal_progress import describe

ROOT = Path(__file__).resolve().parents[1]


class BudgetExtensionTests(unittest.TestCase):
    def test_full_run_uses_all_adjacent_official_controls(self):
        directory = ROOT/'configs/rebuttal'
        pair_config, triple_config = (json.loads((directory/name).read_text()) for name in
                                      ('brec_full.json', 'triple_encodings_full.json'))
        adjacency = nx.to_numpy_array(nx.path_graph(4), dtype=np.int64)
        perm = [2, 0, 3, 1]
        original, changed = (nx.to_graph6_bytes(nx.from_numpy_array(graph), header=False).strip()
                             for graph in (adjacency, adjacency[np.ix_(perm, perm)]))
        data = np.full(51200, original, dtype='S16')
        pid = 7
        offset = (pid + 400)*64
        data[offset+1:offset+64:2] = changed
        indices = [[i, i+1] for i in range(offset, offset+64, 2)]
        kernel = TripleKernel(adjacency, 2, 'monomial')
        fail = AssertionError('Unexpected synthetic relabeling')
        with (patch('rebuttal_experiments.brec.permutations', side_effect=fail),
              patch('rebuttal_experiments.triple_encoding_controls.permutations', side_effect=fail)):
            result = evaluate_condition(data, pid, 'monomial', 'endpoint', 2, 'mean', 1, pair_config)
            self.assertEqual(result['relabel_controls'], [])
            pair_checks = result['official_reliability_controls']
            triple_checks = triple_controls(data, pid, [kernel, kernel], 2, 'monomial',
                                            triple_config, Budget(None, 3))
        self.assertEqual([row['dataset_indices'] for row in pair_checks], indices)
        self.assertEqual([row['dataset_indices'] for row in triple_checks], indices)
        self.assertTrue(all(row['official_representatives']['aligned_enclosures_overlap']
                            for row in pair_checks))
        self.assertTrue(all(row['all_bank_entries_exact_equal'] and
                            row['sampled_full_rows_exact_equal'] for row in triple_checks))
        data[offset+63] = nx.to_graph6_bytes(nx.cycle_graph(4), header=False).strip()
        with self.assertRaisesRegex(ValueError, 'not isomorphic'):
            reliability_control(data, pid, 2, 'monomial', 'endpoint', 'interval', 128,
                                pair_config, Budget(None, 3))
        with self.assertRaisesRegex(ValueError, 'not isomorphic'):
            triple_controls(data, pid, [kernel, kernel], 2, 'monomial', triple_config, Budget(None, 3))

    def test_full_configs_equal_the_existing_union_and_progress_totals(self):
        directory = ROOT/'configs/rebuttal'
        grids = [
            ('brec', 'brec.json', 'brec_p8.json', 'brec_full.json',
             experiment_settings, validate_configuration, 80),
            ('triple-encodings', 'triple_encodings.json', 'triple_encodings_q8_q16.json',
             'triple_encodings_full.json', triple_settings, validate_triples, 112),
        ]
        for command, default, supplement, full, settings, validate, count in grids:
            with self.subTest(command=command):
                baseline, extra, merged = (json.loads((directory/name).read_text())
                                           for name in (default, supplement, full))
                validate(merged)
                grid = list(settings(merged))
                self.assertEqual(len(grid), count)
                self.assertEqual(set(grid), set(settings(baseline)) | set(settings(extra)))
                task = describe([sys.executable, '-m', 'rebuttal_experiments', command,
                                 '--config', str(directory/full), '--out', 'unused'])
                self.assertEqual(task.total, count*400)

    def test_triple_config_rejects_incomplete_or_ambiguous_per_q_powers(self):
        config = json.loads((ROOT/'configs/rebuttal/triple_encodings_full.json').read_text())
        for change in (
            {'mean_p_values': [1]},
            {'mean_p_values_by_q': {'1': [1]}},
            {'mean_p_values_by_q': {**config['mean_p_values_by_q'], '16': [1, 1]}},
        ):
            with self.assertRaises(ValueError):
                validate_triples({**config, **change})

    def test_triple_exact_bank_extends_pair_cache_and_preserves_wide_integers(self):
        adjacency = nx.to_numpy_array(nx.complete_graph(20), dtype='int64')
        with tempfile.TemporaryDirectory() as directory:
            for basis in ('monomial', 'chebyshev'):
                reference = polynomial_bank(adjacency, 4, basis, 'exact')
                lower = polynomial_bank(adjacency, 4, basis, 'exact', bank_cache=directory)
                np.testing.assert_array_equal(lower, reference)
                lengths = []
                extend = polynomial_banks._extend

                def observe(bank, *args):
                    lengths.append(len(bank))
                    return extend(bank, *args)

                with patch.object(polynomial_banks, '_extend', side_effect=observe):
                    high = TripleKernel(adjacency, 16, basis, bank_cache=directory)
                    cached = TripleKernel(adjacency, 16, basis, bank_cache=directory)
                    self.assertEqual(lengths, [5])
                self.assertGreater(high.max_abs[16], np.iinfo(np.int64).max)
                np.testing.assert_array_equal(cached.banks, high.banks)
                np.testing.assert_array_equal(high.banks, TripleKernel(adjacency, 16, basis).banks)

    def test_configs_cover_requested_grid_without_repeating_pair_multisets(self):
        config = json.loads((ROOT/'configs/rebuttal/brec_p8.json').read_text())
        validate_configuration(config)
        settings = list(experiment_settings(config))
        self.assertEqual(len(settings),16)
        self.assertEqual({(q,r,p) for _,_,q,r,p in settings},
                         {(q,'mean',8) for q in (2,4,8,16)})
        task = describe([sys.executable,'-m','rebuttal_experiments','brec',
                         '--config',str(ROOT/'configs/rebuttal/brec_p8.json'),'--out','unused'])
        self.assertEqual(task.total,6400)
        for bad in ([],['mean','mean'],['sum']):
            with self.assertRaises(ValueError):
                validate_configuration({**config,'readouts':bad})
        for bad in (0,-1,1.1,float('inf'),True):
            with self.assertRaises(ValueError):
                validate_configuration({**config,'memory_reservation_scale':bad})
        triples = json.loads((ROOT/'configs/rebuttal/triple_encodings_q8_q16.json').read_text())
        self.assertEqual(triples['q_values'],[8,16])
        self.assertEqual(triples['mean_p_values'],[1,2,4,8])

    def test_mean_p8_cli_replays_and_resumes_without_raw_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = nx.to_graph6_bytes(nx.cycle_graph(3),header=False).strip()
            data = np.full(51200,graph,dtype='S16')
            data[1] = nx.to_graph6_bytes(nx.empty_graph(3),header=False).strip()
            source = root/'brec.npy'
            np.save(source,data,allow_pickle=False)
            config = json.loads((ROOT/'configs/rebuttal/brec_p8.json').read_text())
            config.update(expected_brec_sha256=sha256(source),relabels=1,
                          reliability_relabels=0,memory_gib=4)
            path = root/'config.json'
            path.write_text(json.dumps(config))
            command = [sys.executable,'-m','rebuttal_experiments','brec','--config',str(path),
                       '--brec-file',str(source),'--out',str(root/'run'),
                       '--pair-ids','0','--workers','2','--reserve-gib','0']
            env = dict(os.environ,PYTHONHASHSEED='0')
            result = subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            before = snapshot(root/'run')
            self.assertEqual(len(before['records']),16)
            self.assertTrue(all('_mean_p8_' in name for name in before['records']))
            result = subprocess.run(command+['--resume'],cwd=ROOT,env=env,
                                    capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(snapshot(root/'run'),before)

    def test_high_q_features_against_independent_polynomial_coefficients(self):
        import sympy as sp
        adjacency = nx.to_numpy_array(nx.Graph([(0,1),(1,2),(2,0),(0,3)]),dtype='int64')
        degree = adjacency.sum(1).tolist()
        matrix = sp.Matrix([[sp.Integer(int(adjacency[i,j]))/sp.sqrt(degree[i]*degree[j])
                             for j in range(4)] for i in range(4)])
        powers = [sp.eye(4)]
        for _ in range(16):
            powers.append((matrix*powers[-1]).applyfunc(sp.expand))
        x = sp.Symbol('x')
        for q in (8,16):
            for basis in ('monomial','chebyshev'):
                if basis=='monomial':
                    bank = powers[:q+1]
                else:
                    bank = []
                    for order in range(q+1):
                        poly = sp.Poly(sp.chebyshevt(order,x),x)
                        value = sp.zeros(4)
                        for (power,),coefficient in poly.terms():
                            value += coefficient*powers[power]
                        bank.append(value.applyfunc(sp.expand))
                kernel = TripleKernel(adjacency,q,basis)
                keys = [('pair',f'B{q}',0,1),('endpoint',q-1,0,0,1),
                        ('joint',q,q,q),('joint',0,q,q),('joint',q,q-1,1)]
                triples = [(0,1,2),(0,0,3),(3,3,3)]
                for key in keys:
                    for triple,actual in zip(triples,kernel.column(triples,key)):
                        u,v,w = triple
                        if key[0]=='pair':
                            expected = bank[q][u,v]
                        elif key[0]=='endpoint':
                            expected = sum(bank[q-1][u,t] for t in range(4))
                        else:
                            a,b,c = key[1:]
                            expected = sum(bank[a][u,t]*bank[b][v,t]*bank[c][w,t] for t in range(4))
                        scalar = sum(sp.Rational(c.numerator,c.denominator)*sp.sqrt(s) for s,c in actual.terms)
                        self.assertEqual(sp.expand(scalar-expected),0,(q,basis,key,triple))

    def check_supervisor(self,first_code=0,finished=True,resources=()):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'run'
            calls = []
            class Process:
                pid = 999999
                def __init__(self,command,**kwargs):
                    calls.append(command)
                    self.code = first_code if len(calls)==1 else 0
                    stage_out = Path(command[command.index('--out')+1])
                    stage_out.mkdir(parents=True)
                    (stage_out/'manifest.json').write_text(json.dumps({
                        'status':'execution_finished' if finished else 'running'}))
                def wait(self): return self.code
                def poll(self): return self.code
            with patch.object(extension.subprocess,'Popen',Process):
                code = extension.main(['--out',str(out),'--brec-file',str(out/'input.npy'),
                                       '--workers','28',*resources])
            state = json.loads((out/'extension.json').read_text())
            return code,calls,state

    def test_supervisor_runs_both_stages_in_order(self):
        code,calls,state = self.check_supervisor(resources=('--device','cuda:0'))
        self.assertEqual(code,0)
        self.assertEqual(len(calls),2)
        self.assertIn('brec_p8.json',calls[0][calls[0].index('--config')+1])
        self.assertIn('triple_encodings_q8_q16.json',calls[1][calls[1].index('--config')+1])
        directories = [call[call.index('--bank-cache')+1] for call in calls]
        self.assertEqual(directories[0], directories[1])
        self.assertTrue(all(call[call.index('--device')+1] == 'cuda:0' for call in calls))
        self.assertEqual(state['completed_stages'],['brec','triple-encodings'])

    def test_supervisor_passes_100_workers_and_110_gib_to_both_stages(self):
        code,calls,state = self.check_supervisor(resources=('--workers','100','--memory-gib','110'))
        self.assertEqual(code,0)
        self.assertEqual(len(calls),2)
        for command in calls:
            self.assertEqual(command[command.index('--workers')+1],'100')
            self.assertEqual(float(command[command.index('--memory-gib')+1]),110)
        self.assertEqual(state['memory_gib_override'],110)

    def test_full_supervisor_selects_merged_grids(self):
        devices = ('cuda:0', 'cuda:1', 'cuda:2', 'cuda:3')
        code, calls, state = self.check_supervisor(resources=('--full', '--resume', '--devices', *devices))
        self.assertEqual(code, 0)
        self.assertEqual([Path(call[call.index('--config')+1]).name for call in calls],
                         ['brec_full.json', 'triple_encodings_full.json'])
        self.assertTrue(all('--resume' in call for call in calls))
        for call in calls:
            start = call.index('--devices') + 1
            self.assertEqual(call[start:start+4], list(devices))
        self.assertEqual(state['scope'], 'full')

    def test_real_resource_overrides_are_recorded_in_both_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = nx.to_graph6_bytes(nx.cycle_graph(3),header=False).strip()
            source = root/'brec.npy'
            np.save(source,np.full(51200,graph,dtype='S16'),allow_pickle=False)
            cache = root/'banks'
            bank_times = None
            for experiment,config_name in (('brec','brec_p8.json'),
                                           ('triple-encodings','triple_encodings_q8_q16.json')):
                config = json.loads((ROOT/'configs/rebuttal'/config_name).read_text())
                config.update(expected_brec_sha256=sha256(source),q_values=[2],
                              relabels=1,reliability_relabels=0,memory_gib=4)
                path = root/f'{experiment}.json'
                path.write_text(json.dumps(config))
                command = [sys.executable,'-m','rebuttal_experiments',experiment,
                    '--config',str(path),'--brec-file',str(source),'--out',str(root/experiment),
                    '--bank-cache',str(cache),
                    '--pair-ids','0','--workers','100','--memory-gib','110','--reserve-gib','0.25']
                result = subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONHASHSEED='0'),
                                        capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                manifest = json.loads((root/experiment/'manifest.json').read_text())
                self.assertEqual(manifest['status'],'execution_finished')
                recorded = manifest['identity']['configuration']
                self.assertEqual(recorded['memory_gib'],110)
                self.assertEqual(recorded['scheduling']['workers'],100)
                self.assertEqual(recorded['scheduling']['reserve_gib'],0.25)
                actual = {p.name: p.stat().st_mtime_ns for p in cache.glob('*.npz')}
                self.assertEqual(len(actual), 2)
                if bank_times is not None:
                    self.assertEqual(actual, bank_times)
                bank_times = actual

    def test_supervisor_does_not_continue_after_failure_or_existing_writer(self):
        for first_code,finished in ((7,True),(0,False)):
            code,calls,state = self.check_supervisor(first_code,finished)
            self.assertNotEqual(code,0)
            self.assertEqual(len(calls),1)
            self.assertEqual(state['status'],'failed')


if __name__ == '__main__':
    unittest.main()
