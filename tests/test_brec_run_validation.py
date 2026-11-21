"""Run identity, locking, and atomic output regression checks."""
import json
from pathlib import Path
import tempfile
import unittest

from rebuttal_experiments.common import atomic_output, run_directory




class OutputIntegrityTests(unittest.TestCase):
    def test_atomic_write_failure_keeps_original(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'record.json'
            target.write_text('original')
            with self.assertRaises(RuntimeError):
                with atomic_output(target) as temporary:
                    temporary.write_text('partial')
                    raise RuntimeError('interrupted')
            self.assertEqual(target.read_text(), 'original')
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_output_lock_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            with run_directory(root, {'test': 1}):
                with self.assertRaises(BlockingIOError):
                    with run_directory(root, {'test': 1}, resume=True):
                        self.fail('A concurrent writer acquired the lock')
                self.assertEqual(json.loads((root / 'manifest.json').read_text())['status'], 'running')
            original = (root / 'manifest.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                with run_directory(root, {'test': 2}, resume=True):
                    self.fail('Changed configuration was resumed')
            self.assertEqual((root / 'manifest.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
