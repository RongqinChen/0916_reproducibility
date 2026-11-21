"""Progress rendering, log preservation, resume counts and signal forwarding."""
import io
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


from tools.rebuttal_progress import Progress, Task

ROOT = Path(__file__).resolve().parents[1]


class ProgressTests(unittest.TestCase):
    def test_resume_counts_and_eta(self):
        task = Task('Prepare', 'protein-prepare', 100, 40, 'graphs')
        output = io.StringIO()
        now = [0]
        progress = Progress(task, output, lambda: now[0])
        for index in range(41, 51):
            self.assertTrue(task.observe(f'prepared graph {index}/100'))
        now[0] = 20
        progress.refresh(force=True)
        text = output.getvalue()
        self.assertIn('50/100 graphs 50.0%', text)
        self.assertIn('elapsed 00:00:20', text)
        self.assertIn('ETA 00:01:40', text)

    def test_failure_does_not_force_completion(self):
        task = Task('Prepare', 'protein-prepare', 100, 4, 'conditions')
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

    def test_interactive_bar_overwrites_one_line(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        output = Terminal()
        task = Task('Prepare', 'protein-prepare', 10)
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
        task = Task('Prepare', 'protein-prepare', 25600)
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


if __name__ == '__main__':
    unittest.main()
