"""Regression checks for scheduling, resource limits and scientific parity."""
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from rebuttal_experiments.common import Budget, ResourceLimit, hard_wall_limit, sha256
from rebuttal_experiments.scheduling import run_tasks
from brec.reproducibility import snapshot

ROOT = Path(__file__).resolve().parents[1]


def example_task(index, estimated_bytes=192 * 2**20):
    return {'stem': f'task{index}', 'record': {'task_id': index},
            'estimated_bytes': estimated_bytes, 'work': 1}


def sleeping_worker(task):
    started = time.monotonic()
    time.sleep(0.3)
    return {**task['record'], 'status': 'certified_equal',
            'started': started, 'ended': time.monotonic(), 'pid': os.getpid()}


class ParallelSchedulingTests(unittest.TestCase):
    def test_workers_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            run_tasks([example_task(i) for i in range(2)], sleeping_worker, out,
                      workers=2, memory_gib=3, reserve_gib=0, seed=0)
            a, b = [json.loads((out / 'records' / f'task{i}.json').read_text()) for i in range(2)]
            self.assertNotEqual(a['pid'], b['pid'])
            self.assertLess(max(a['started'], b['started']), min(a['ended'], b['ended']))

    def test_global_memory_watchdog(self):
        parent = os.getpid()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch('rebuttal_experiments.scheduling.process_rss',
                       side_effect=lambda pid: 0 if pid == parent else 2 * 2**30):
                run_tasks([example_task(i) for i in range(2)], sleeping_worker, out,
                          workers=2, memory_gib=1, reserve_gib=0, seed=0)
            for index in range(2):
                record = json.loads((out / 'records' / f'task{index}.json').read_text())
                self.assertEqual(record['status'], 'resource_limited')
                self.assertEqual(record['reason'], 'scheduler_memory_limit')
            self.assertEqual(multiprocessing.active_children(), [])

    def test_oversized_task_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            run_tasks([example_task(0, 2 * 2**30)], sleeping_worker, out,
                      workers=2, memory_gib=1, reserve_gib=0, seed=0)
            record = json.loads((out / 'records' / 'task0.json').read_text())
            self.assertEqual(record['reason'], 'estimated_memory_limit')

    def test_interrupt_stops_children(self):
        parent, original_sleep = os.getpid(), time.sleep

        def interrupt_parent(seconds):
            if os.getpid() == parent:
                raise KeyboardInterrupt
            original_sleep(seconds)

        with tempfile.TemporaryDirectory() as directory:
            with patch('rebuttal_experiments.scheduling.time.sleep', side_effect=interrupt_parent):
                with self.assertRaises(KeyboardInterrupt):
                    run_tasks([example_task(i) for i in range(2)], sleeping_worker, Path(directory),
                              workers=2, memory_gib=3, reserve_gib=0, seed=0)
            self.assertEqual(multiprocessing.active_children(), [])

    def test_unlimited_time_keeps_memory_limit(self):
        budget = Budget(None, 52)
        self.assertIsNone(budget.deadline)
        budget.check()
        with self.assertRaisesRegex(ResourceLimit, 'estimated_memory_limit'):
            budget.check(52 * 2**30 + 1)
        with self.assertRaisesRegex(ResourceLimit, 'wall_time_limit'):
            with hard_wall_limit(0.03):
                with hard_wall_limit(None):
                    time.sleep(0.1)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)


class ScientificParityTests(unittest.TestCase):
    def test_serial_parallel_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graphs = [nx.cycle_graph(4), nx.empty_graph(4), nx.path_graph(4), nx.star_graph(3)]
            graph6 = [nx.to_graph6_bytes(graph, header=False).strip() for graph in graphs]
            data = np.full(51200, graph6[0], dtype='S16')
            data[0], data[1] = graph6[0], graph6[1]
            data[64], data[65] = graph6[2], graph6[3]
            data[401 * 64] = graph6[2]
            mapping = {i: 3 - i for i in range(4)}
            data[401 * 64 + 1] = nx.to_graph6_bytes(
                nx.relabel_nodes(graphs[2], mapping), nodes=range(4), header=False).strip()
            source = root / 'brec.npy'
            np.save(source, data, allow_pickle=False)
            config = {'expected_brec_sha256': sha256(source), 'bases': ['monomial'],
                      'variants': ['polynomial'], 'q_values': [2], 'mean_p_values': [1, 2, 4],
                      'precision_bits': [16, 32], 'relabels': 1, 'reliability_relabels': 1,
                      'seed': 0, 'seconds_per_condition': None, 'memory_gib': 3,
                      'max_candidate_edges': 10000}
            configuration = root / 'config.json'
            configuration.write_text(json.dumps(config))

            def command(out, workers, resume=False):
                return [sys.executable, '-m', 'rebuttal_experiments', 'brec',
                        '--brec-file', str(source), '--config', str(configuration),
                        '--out', str(out), '--workers', str(workers), '--reserve-gib', '0',
                        '--pair-ids', '0', '1', *(['--resume'] if resume else [])]

            outputs = [root / 'serial', root / 'parallel', root / 'parallel_repeat']
            for out, workers in zip(outputs, (1, 2, 2)):
                result = subprocess.run(command(out, workers), cwd=ROOT, capture_output=True,
                                        text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(json.loads((out / 'manifest.json').read_text())['status'],
                                 'execution_finished')

            def records(out):
                result = {}
                for path in (out / 'records').glob('*.json'):
                    record = json.loads(path.read_text())
                    result[path.name] = {key: value for key, value in record.items()
                                         if key not in ('elapsed_seconds', 'identity_sha256')}
                return result

            self.assertEqual(len(records(outputs[0])), 8)
            self.assertEqual(records(outputs[0]), records(outputs[1]))
            self.assertEqual(json.loads((outputs[0] / 'summary.json').read_text()),
                             json.loads((outputs[1] / 'summary.json').read_text()))
            self.assertEqual(snapshot(outputs[1]), snapshot(outputs[2]))
            files = list((outputs[1] / 'records').glob('*.json'))
            before = {p.name: p.read_bytes() for p in files}
            # Resume must schedule a missing condition and preserve completed records.
            missing = files[0]
            missing.unlink()
            resumed = subprocess.run(command(outputs[1], 2, True), cwd=ROOT,
                                     capture_output=True, text=True, timeout=60)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertIn('1 pending conditions', resumed.stdout)
            for path in files[1:]:
                self.assertEqual(path.read_bytes(), before[path.name])
            self.assertEqual(records(outputs[0]), records(outputs[1]))


if __name__ == '__main__':
    unittest.main()
