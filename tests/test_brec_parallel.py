"""Regression checks for scheduling, resource limits and scientific parity."""
import contextlib
import io
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

from rebuttal_experiments.common import Budget, ResourceLimit, atomic_json, hard_wall_limit, sha256
from rebuttal_experiments.brec import evaluate_budgets, evaluate_condition
from rebuttal_experiments.encodings import feature_matrix
from rebuttal_experiments.runtime import report_phase, report_to
from rebuttal_experiments.scheduling import run_tasks
from brec.reproducibility import snapshot

ROOT = Path(__file__).resolve().parents[1]


def example_task(index, estimated_bytes=192 * 2**20):
    return {'stem': f'task{index}', 'record': {'task_id': index},
            'estimated_bytes': estimated_bytes, 'work': 1}


def sleeping_worker(task):
    started = time.monotonic()
    report_phase('test_sleep', step=1)
    time.sleep(0.3)
    return {**task['record'], 'status': 'certified_equal',
            'started': started, 'ended': time.monotonic(), 'pid': os.getpid()}


def crashing_worker(task):
    os._exit(7)


def device_worker(task):
    return {**sleeping_worker(task), 'device': task['device']}


class ParallelSchedulingTests(unittest.TestCase):
    def test_four_devices_receive_overlapping_jobs(self):
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            run_tasks([example_task(i) for i in range(4)], device_worker, out, workers=4,
                      memory_gib=3, reserve_gib=0, seed=0, devices=devices)
            rows = [json.loads((out/'records'/f'task{i}.json').read_text()) for i in range(4)]
            self.assertEqual([row['device'] for row in rows], devices)
            self.assertLess(max(row['started'] for row in rows), min(row['ended'] for row in rows))

    def test_live_progress_exposes_busy_worker_phase_and_cleans_up(self):
        snapshots = []

        def capture(path, value):
            atomic_json(path, value)
            if path.name == 'progress.json':
                snapshots.append(value)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            output = io.StringIO()
            with (patch('rebuttal_experiments.scheduling.HEARTBEAT_SECONDS', 0.02),
                  patch('rebuttal_experiments.scheduling.atomic_json', side_effect=capture),
                  contextlib.redirect_stdout(output)):
                run_tasks([example_task(0)], sleeping_worker, out,
                          workers=1, memory_gib=3, reserve_gib=0, seed=0)
            busy = [row for snapshot in snapshots for row in snapshot['workers']
                    if row['phase'] == 'test_sleep']
            self.assertTrue(busy, snapshots)
            self.assertEqual(busy[0]['details'], {'step': 1})
            self.assertGreater(busy[0]['rss_bytes'], 0)
            self.assertGreaterEqual(busy[0]['phase_elapsed_seconds'], 0)
            self.assertIn('BREC heartbeat: active=1', output.getvalue())
            self.assertIn('test_sleep', output.getvalue())
            final = json.loads((out / 'progress.json').read_text())
            self.assertEqual(final['status'], 'finished')
            self.assertEqual(final['completed_this_session'], 1)
            self.assertEqual(final['workers'], [])
            self.assertFalse(list(out.glob('.brec-progress-*')))

    def test_phase_report_context_is_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = [Path(directory) / name for name in ('first.json', 'second.json')]
            report_phase('no_reporter')
            with report_to(first):
                report_phase('first', bits=128)
                with report_to(second):
                    report_phase('second', bits=512)
                report_phase('first_restored')
            before = first.read_bytes()
            report_phase('no_reporter_again')
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(json.loads(first.read_text())['phase'], 'first_restored')
            self.assertEqual(json.loads(second.read_text())['details'], {'bits': 512})

    def test_worker_crash_cannot_be_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'worker exited 7 without a result'):
                run_tasks([example_task(0)], crashing_worker, out,
                          workers=1, memory_gib=3, reserve_gib=0, seed=0)
            self.assertFalse((out / 'records' / 'task0.json').exists())
            self.assertEqual(json.loads((out / 'progress.json').read_text())['status'], 'failed')
            self.assertFalse(list(out.glob('.brec-progress-*')))
            self.assertEqual(multiprocessing.active_children(), [])

    def test_workers_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            run_tasks([example_task(i) for i in range(2)], sleeping_worker, out,
                      workers=2, memory_gib=3, reserve_gib=0, seed=0)
            a, b = [json.loads((out / 'records' / f'task{i}.json').read_text()) for i in range(2)]
            self.assertNotEqual(a['pid'], b['pid'])
            self.assertLess(max(a['started'], b['started']), min(a['ended'], b['ended']))

    def test_calibrated_reservations_admit_more_workers_with_same_rss_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch('rebuttal_experiments.scheduling.process_rss',return_value=0):
                run_tasks([example_task(i,3*2**28) for i in range(2)],sleeping_worker,out,
                          workers=2,memory_gib=1,reserve_gib=0,seed=0,reservation_scale=0.6)
            a,b = [json.loads((out/'records'/f'task{i}.json').read_text()) for i in range(2)]
            self.assertLess(max(a['started'],b['started']),min(a['ended'],b['ended']))
            progress = json.loads((out/'progress.json').read_text())
            self.assertEqual(progress['memory_budget_bytes'],2**30)
            self.assertEqual(progress['memory_reservation_scale'],0.6)

    def test_global_memory_watchdog(self):
        parent = os.getpid()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch('rebuttal_experiments.scheduling.process_rss',
                       side_effect=lambda pid: 0 if pid == parent else 2 * 2**30):
                run_tasks([example_task(i) for i in range(2)], sleeping_worker, out,
                          workers=2, memory_gib=1, reserve_gib=0, seed=0,reservation_scale=0.6)
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
            out = Path(directory)
            with patch('rebuttal_experiments.scheduling.time.sleep', side_effect=interrupt_parent):
                with self.assertRaises(KeyboardInterrupt):
                    run_tasks([example_task(i) for i in range(2)], sleeping_worker, out,
                              workers=2, memory_gib=3, reserve_gib=0, seed=0)
            self.assertEqual(multiprocessing.active_children(), [])
            self.assertEqual(json.loads((out / 'progress.json').read_text())['status'], 'interrupted')
            self.assertFalse(list(out.glob('.brec-progress-*')))

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
    def test_grouped_evaluation_preserves_each_readouts_stopping_precision(self):
        path = nx.to_graph6_bytes(nx.path_graph(4), header=False).strip()
        star = nx.to_graph6_bytes(nx.star_graph(3), header=False).strip()
        data = np.full(51200, path, dtype='S16')
        data[1] = star
        config = {'seconds_per_condition': None, 'memory_gib': 3,
                  'precision_bits': [2, 8, 128], 'max_candidate_edges': 1000,
                  'seed': 0, 'relabels': 1, 'reliability_relabels': 1}
        arguments = (data, 0, 'chebyshev', 'endpoint')
        settings = [(q, readout, p) for q in (2, 4)
                    for readout, p in [('mean', 4), ('multiset', 1), ('mean', 1), ('mean', 2)]]

        with patch('rebuttal_experiments.brec.feature_matrix', wraps=feature_matrix) as features:
            grouped = dict(evaluate_budgets(*arguments, settings, config))
            # Two distinct stopping precisions. Each needs two main features,
            # two main relabels, two reliability graphs and one reliability relabel.
            self.assertEqual(features.call_count, 14)
            self.assertEqual({call.args[1] for call in features.call_args_list}, {4})
        for q, readout, p in settings:
            independent = evaluate_condition(*arguments, q, readout, p, config)
            shared = grouped[q, readout, p]
            del independent['elapsed_seconds'], shared['elapsed_seconds']
            self.assertEqual(independent, shared)
            self.assertEqual(shared['precision_attempts'][-1]['bits'],
                             2 if readout == 'multiset' else 8)

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
            config = {'expected_brec_sha256': sha256(source), 'bases': ['monomial', 'chebyshev'],
                      'variants': ['polynomial', 'endpoint'], 'q_values': [2, 16], 'mean_p_values': [1, 2, 4],
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

            self.assertEqual(len(records(outputs[0])), 64)
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
