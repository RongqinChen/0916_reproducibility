"""Progress rendering, log preservation, resume counts and signal forwarding."""
import io
import fcntl
import json
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

from rebuttal_experiments.common import sha256, source_identity
from tools.rebuttal_progress import Progress, Task, describe

ROOT = Path(__file__).resolve().parents[1]


class ProgressTests(unittest.TestCase):
    def test_triple_encodings_totals_and_record_poll(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'records').mkdir()
            task = describe([sys.executable,'-m','rebuttal_experiments','triple-encodings',
                             '--out',str(root),'--pair-ids','0','1'])
            self.assertEqual(task.total,144)
            self.assertEqual(task.completed,0)
            (root/'records'/'condition.json').write_text('{}')
            task.poll()
            self.assertEqual(task.completed,1)
            (root/'records'/'last_condition.json').write_text('{}')
            task.finish(0)
            self.assertEqual(task.completed,2)

    def test_triple_ppgn_totals_and_scheduler_poll(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps({'seeds': [100, 200, 300],
                                          'variants': ['original', 'baseline', 'joint']}))
            task = describe([sys.executable, '-m', 'rebuttal_experiments', 'triple-ppgn',
                             '--config', str(config), '--out', str(root), '--pair-ids', '0', '1'])
            self.assertEqual(task.total, 18)
            (root / 'progress.json').write_text(json.dumps({
                'completed_records': 6, 'workers': [{}, {}], 'pending_jobs': 4}))
            task.poll()
            self.assertEqual(task.completed, 6)
            self.assertIn('2 GPU workers', task.detail)

    def test_brec_resume_counts_and_eta(self):
        task = Task('BREC', 'brec', 100, 40, 'conditions')
        output = io.StringIO()
        now = [0]
        progress = Progress(task, output, lambda: now[0])
        self.assertTrue(task.observe('[10/60] monomial_polynomial_q2_mean_p1_pair001: certified_equal'))
        now[0] = 20
        progress.refresh(force=True)
        text = output.getvalue()
        self.assertIn('50/100 conditions 50.0%', text)
        self.assertIn('elapsed 00:00:20', text)
        self.assertIn('ETA 00:01:40', text)

    def test_failure_does_not_force_completion(self):
        task = Task('BREC', 'brec', 100, 4, 'conditions')
        output = io.StringIO()
        Progress(task, output).finish(7)
        self.assertEqual(task.completed, 4)
        self.assertIn('failed (7)', output.getvalue())
        self.assertNotIn('100.0%', output.getvalue())

    def test_epoch_and_graph_logs_are_not_double_counted(self):
        task = Task('Train', 'protein-train', 100)
        line = 'fold0_seed0_joint epoch=3 val_nll=0.12\n'
        task.observe(line)
        task.observe(line)
        self.assertEqual(task.completed, 1)
        graph = Task('Prepare', 'protein-prepare', 600, 20)
        graph.observe('prepared graph 21/600')
        graph.observe('prepared graph 21/600')
        self.assertEqual(graph.completed, 21)

    def test_custom_brec_configuration_and_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps({'bases': ['monomial'], 'variants': ['polynomial'],
                                           'q_values': [2, 4], 'mean_p_values': [1, 2]}))
            (root / 'out' / 'records').mkdir(parents=True)
            (root / 'out' / 'records' / 'existing.json').write_text('{}')
            task = describe([sys.executable, '-u', '-m', 'rebuttal_experiments', 'brec',
                             '--config', str(config), '--out', str(root / 'out'),
                             '--pair-ids', '3', '4', '--workers', '2', '--resume'])
            self.assertEqual(task.total, 12)
            self.assertEqual(task.completed, 1)

    def test_interactive_bar_overwrites_one_line(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        output = Terminal()
        task = Task('BREC', 'brec', 10)
        bar = Progress(task, output)
        bar.refresh(force=True)
        task.completed = 5
        bar.refresh(force=True)
        task.completed = 10
        bar.finish(0)
        self.assertIn('50.0%', output.getvalue())
        self.assertIn('100.0%', output.getvalue())
        self.assertEqual(output.getvalue().count('\n'), 1)

    def test_narrow_terminal_keeps_elapsed_eta_and_status(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        output = Terminal()
        task = Task('BREC', 'brec', 25600)
        with patch('tools.rebuttal_progress.shutil.get_terminal_size',
                   return_value=os.terminal_size((80, 24))):
            bar = Progress(task, output)
            task.completed = 1
            bar.refresh(force=True)
        line = output.getvalue().removeprefix('\r\033[K')
        self.assertLessEqual(len(line), 79)
        self.assertIn('00:00:00', line)
        self.assertIn('ETA', line)
        self.assertIn('running', line)


class ProgressProcessTests(unittest.TestCase):
    def test_busy_output_reports_original_writer_without_launching_another(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root/'triple_encodings'
            (out/'records').mkdir(parents=True)
            manifest = out/'manifest.json'
            manifest.write_text('{"status":"running","original_writer":true}\n')
            (out/'progress.json').write_text(json.dumps({
                'scheduler_pid':os.getpid(),'workers':[{'cpu':0}]}))
            (out/'records'/'original.json').write_text('{}')
            before = {p.name:p.read_bytes() for p in out.glob('*.json')}
            sources = source_identity()
            environment = dict(os.environ,REBUTTAL_PYTHON=sys.executable,
                               BREC_FILE=str(root/'must_not_be_opened.npy'),REBUTTAL_OUT=str(root))
            command = ['bash','tools/run_rebuttal.sh','triple-encodings','--workers','1']
            with (out/'.writer.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                for enabled in ('0','1'):
                    for extra,code in (([],2),(['--resume'],0)):
                        result = subprocess.run(command+extra,cwd=ROOT,
                            env=dict(environment,REBUTTAL_PROGRESS=enabled),
                            capture_output=True,text=True,timeout=10)
                        self.assertEqual(result.returncode,code,result.stdout+result.stderr)
                        self.assertIn('实验已在运行',result.stderr)
                        self.assertIn(f'PID：{os.getpid()}',result.stderr)
                        self.assertNotIn('Traceback',result.stderr)
                        self.assertNotIn('FileNotFoundError',result.stderr)
                        self.assertEqual({p.name:p.read_bytes() for p in out.glob('*.json')},before)
                # The duplicate invocations must leave the original writer's lock intact.
                with (out/'.writer.lock').open('rb') as probe:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(probe,fcntl.LOCK_SH | fcntl.LOCK_NB)
            self.assertEqual(source_identity(),sources)

    def test_disabled_progress_preserves_output_and_exit_status(self):
        result = subprocess.run([sys.executable,'-m','tools.rebuttal_progress','--no-progress','--',
            sys.executable,'-c','import sys; print("original output"); sys.exit(7)'],
            cwd=ROOT,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,7)
        self.assertEqual(result.stdout,'original output\n')
        self.assertEqual(result.stderr,'')

    def test_logs_and_nonzero_exit_code_are_preserved(self):
        result = subprocess.run([
            sys.executable, '-m', 'tools.rebuttal_progress', '--', sys.executable, '-u', '-c',
            'import sys; print("original stdout"); print("original stderr",file=sys.stderr); sys.exit(7)'],
            cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, 'original stdout\n')
        self.assertIn('original stderr\n', result.stderr)
        self.assertIn('failed (7)', result.stderr)

    def test_partial_utf8_output_is_preserved(self):
        result = subprocess.run([
            sys.executable, '-m', 'tools.rebuttal_progress', '--', sys.executable, '-u', '-c',
            'import os; data="中文日志".encode(); [os.write(1,bytes([b])) for b in data]'],
            cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '中文日志')

    def test_sigterm_reaches_child_and_descendants(self):
        self.check_signal(signal.SIGTERM, 143)

    def test_sigint_reaches_child_and_descendants(self):
        self.check_signal(signal.SIGINT, 130)

    def check_signal(self, signum, expected_code):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'child.py'
            marker = root / 'pids.json'
            script.write_text(
                'import json,os,subprocess,sys,time\n'
                'from pathlib import Path\n'
                'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\n'
                'Path(sys.argv[1]).write_text(json.dumps([os.getpid(),child.pid]))\n'
                'time.sleep(60)\n')
            process = subprocess.Popen([sys.executable, '-m', 'tools.rebuttal_progress', '--',
                                        sys.executable, str(script), str(marker)], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists())
                pids = json.loads(marker.read_text())
                process.send_signal(signum)
                _, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, expected_code, stderr)
                for pid in pids:
                    stat = Path(f'/proc/{pid}/stat')
                    self.assertTrue(not stat.exists() or stat.read_text().split()[2] == 'Z')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
                if marker.exists():
                    for pid in json.loads(marker.read_text()):
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def test_shell_wrapper_runs_and_resumes_real_brec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph6 = nx.to_graph6_bytes(nx.cycle_graph(3), header=False).strip()
            data = np.full(51200, graph6, dtype='S16')
            data[1] = nx.to_graph6_bytes(nx.empty_graph(3), header=False).strip()
            source = root / 'brec.npy'
            np.save(source, data, allow_pickle=False)
            config = json.loads((ROOT / 'configs/rebuttal/brec.json').read_text())
            config.update(expected_brec_sha256=sha256(source), bases=['monomial'],
                          variants=['polynomial'], q_values=[2], mean_p_values=[1],
                          relabels=1, reliability_relabels=1, memory_gib=3)
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            environment = dict(os.environ, REBUTTAL_PYTHON=sys.executable,
                               BREC_FILE=str(source), REBUTTAL_OUT=str(root / 'out'),
                               REBUTTAL_PROGRESS='1')
            command = ['bash', 'tools/run_rebuttal.sh', 'brec', '--workers', '2',
                       '--reserve-gib', '0', '--pair-ids', '0', '--config', str(path)]
            before = source_identity()
            for extra in ([], ['--resume']):
                result = subprocess.run(command + extra, cwd=ROOT, env=environment,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('2/2 conditions 100.0%', result.stderr)
                self.assertIn('finished', result.stderr)
            self.assertEqual(len(list((root / 'out' / 'brec' / 'records').glob('*.json'))), 2)
            self.assertEqual(source_identity(), before)


if __name__ == '__main__':
    unittest.main()
