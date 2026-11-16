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
from rebuttal_experiments.brec import experiment_settings, validate_configuration
from rebuttal_experiments.common import sha256
from rebuttal_experiments.triple_exact import TripleKernel
from tools import rebuttal_budget_extension as extension
from tools.rebuttal_progress import describe

ROOT = Path(__file__).resolve().parents[1]


class BudgetExtensionTests(unittest.TestCase):
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
        code,calls,state = self.check_supervisor()
        self.assertEqual(code,0)
        self.assertEqual(len(calls),2)
        self.assertIn('brec_p8.json',calls[0][calls[0].index('--config')+1])
        self.assertIn('triple_encodings_q8_q16.json',calls[1][calls[1].index('--config')+1])
        self.assertEqual(state['completed_stages'],['brec','triple-encodings'])

    def test_supervisor_passes_100_workers_and_110_gib_to_both_stages(self):
        code,calls,state = self.check_supervisor(resources=('--workers','100','--memory-gib','110'))
        self.assertEqual(code,0)
        self.assertEqual(len(calls),2)
        for command in calls:
            self.assertEqual(command[command.index('--workers')+1],'100')
            self.assertEqual(float(command[command.index('--memory-gib')+1]),110)
        self.assertEqual(state['memory_gib_override'],110)

    def test_real_resource_overrides_are_recorded_in_both_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = nx.to_graph6_bytes(nx.cycle_graph(3),header=False).strip()
            source = root/'brec.npy'
            np.save(source,np.full(51200,graph,dtype='S16'),allow_pickle=False)
            for experiment,config_name in (('brec','brec_p8.json'),
                                           ('triple-encodings','triple_encodings_q8_q16.json')):
                config = json.loads((ROOT/'configs/rebuttal'/config_name).read_text())
                config.update(expected_brec_sha256=sha256(source),q_values=[2],
                              relabels=1,reliability_relabels=0,memory_gib=4)
                path = root/f'{experiment}.json'
                path.write_text(json.dumps(config))
                command = [sys.executable,'-m','rebuttal_experiments',experiment,
                    '--config',str(path),'--brec-file',str(source),'--out',str(root/experiment),
                    '--pair-ids','0','--workers','100','--memory-gib','110','--reserve-gib','8']
                result = subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONHASHSEED='0'),
                                        capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                manifest = json.loads((root/experiment/'manifest.json').read_text())
                self.assertEqual(manifest['status'],'execution_finished')
                recorded = manifest['identity']['configuration']
                self.assertEqual(recorded['memory_gib'],110)
                self.assertEqual(recorded['scheduling']['workers'],100)

    def test_supervisor_does_not_continue_after_failure_or_existing_writer(self):
        for first_code,finished in ((7,True),(0,False)):
            code,calls,state = self.check_supervisor(first_code,finished)
            self.assertNotEqual(code,0)
            self.assertEqual(len(calls),1)
            self.assertEqual(state['status'],'failed')


if __name__ == '__main__':
    unittest.main()
