"""Run identity, record coverage and atomic output regression checks."""
import json
from pathlib import Path
import tempfile
import unittest

from brec.reproducibility import snapshot
from rebuttal_experiments.common import atomic_json, atomic_output, identity_hash, run_directory


def fixture(root):
    config = {'bases': ['monomial'], 'variants': ['polynomial'], 'q_values': [2],
              'mean_p_values': [1], 'executed_pair_ids': [0],
              'determinism': {'policy': 'brec-determinism-v1'}}
    identity = {'configuration': config}
    checksum = identity_hash(identity)
    atomic_json(root / 'manifest.json', {'status': 'execution_finished', 'identity': identity,
                                         'identity_sha256': checksum})
    settings = []
    for readout in ('multiset', 'mean'):
        stem = f'monomial_polynomial_q2_{readout}_p1'
        atomic_json(root / 'records' / (stem + '_pair000.json'), {
            'basis': 'monomial', 'variant': 'polynomial', 'q': 2, 'order': 2,
            'readout': readout, 'p': 1, 'pair_id': 0, 'identity_sha256': checksum,
            'status': 'certified_equal', 'elapsed_seconds': 1})
        settings.append({'setting': stem, 'denominator': 400,
                         'counts': {'certified_equal': 1, 'not_executed': 399},
                         'certified_detection_lower_bound': 0, 'detection_upper_bound': 399})
    atomic_json(root / 'summary.json', {'settings': settings})


class ReplayValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture(self.root)

    def test_valid_requested_subset(self):
        result = snapshot(self.root)
        self.assertEqual(len(result['records']), 2)
        self.assertTrue(all('elapsed_seconds' not in r for r in result['records'].values()))

    def test_same_record_count_with_wrong_filename_is_rejected(self):
        path = next((self.root / 'records').glob('*.json'))
        path.rename(path.with_name('wrong_condition.json'))
        with self.assertRaisesRegex(ValueError, 'Unexpected condition'):
            snapshot(self.root)

    def test_wrong_condition_body_is_rejected(self):
        path = next((self.root / 'records').glob('*.json'))
        record = json.loads(path.read_text())
        record['pair_id'] = 345
        atomic_json(path, record)
        with self.assertRaisesRegex(ValueError, 'Unexpected condition'):
            snapshot(self.root)

    def test_record_identity_is_checked(self):
        path = next((self.root / 'records').glob('*.json'))
        record = json.loads(path.read_text())
        record['identity_sha256'] = 'changed'
        atomic_json(path, record)
        with self.assertRaisesRegex(ValueError, 'record identity'):
            snapshot(self.root)

    def test_manifest_identity_is_checked(self):
        path = self.root / 'manifest.json'
        record = json.loads(path.read_text())
        record['identity']['configuration']['extra'] = 'changed'
        atomic_json(path, record)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            snapshot(self.root)

    def test_summary_must_agree_with_records(self):
        path = self.root / 'summary.json'
        record = json.loads(path.read_text())
        record['settings'][0]['detection_upper_bound'] = 400
        atomic_json(path, record)
        with self.assertRaisesRegex(ValueError, 'Summary differs'):
            snapshot(self.root)

    def test_missing_condition_is_rejected(self):
        next((self.root / 'records').glob('*.json')).unlink()
        with self.assertRaisesRegex(ValueError, 'do not cover'):
            snapshot(self.root)

    def test_resource_limited_condition_cannot_pass_replay(self):
        path = next((self.root / 'records').glob('*.json'))
        record = json.loads(path.read_text())
        record.update(status='resource_limited', reason='scheduler_memory_limit')
        atomic_json(path, record)
        with self.assertRaisesRegex(ValueError, 'resource-limited'):
            snapshot(self.root)


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
