"""Write-locked checkpointing and per-pair scheduling for BREC."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import json
from multiprocessing import get_context

import numpy as np

from rebuttal_experiments.brec_data import BLOCK_SIZE, RELABELS_PER_SIDE
from rebuttal_experiments.common import atomic_json

from .protocol import EXPECTED_SHA256, TOLERANCE



def read_json(path):
    return json.loads(path.read_text())


@contextmanager
def experiment_directory(out, protocol):
    """Serialize writes and resume only records from the original protocol."""
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'An experiment is already writing to {out}') from exc
        config = out/'protocol.json'
        if config.exists() and read_json(config) != protocol:
            raise ValueError('Use a new output directory when changing the protocol')
        atomic_json(config, protocol)
        (out/'records').mkdir(exist_ok=True)
        yield


def verify_lower_order(out, ids, gap_keys):
    """Check the lower-order checkpoint when a completed run is supplied."""
    summary = read_json(out/'summary.json')
    if (summary['dataset_sha256'] != EXPECTED_SHA256 or summary['dtype'] != 'float32'
            or summary['threshold'] != TOLERANCE or summary['reference_equal_marked_different']):
        raise ValueError(f'Expected a verified lower-order float32 experiment in {out}')
    for pair_id in ids:
        record = read_json(out/'records'/f'pair-{pair_id:03d}.json')
        for key in gap_keys:
            if np.max(record[key], initial=0) > TOLERANCE:
                raise ValueError(f'Pair {pair_id} was distinguished by the lower-order encoding')


def run_cpu(out, stage, ids, function, data, workers):
    prefix = 'pair' if stage == 'main' else 'control'
    paths = [(out/'records'/f'{prefix}-{pid:03d}.json', pid) for pid in ids]
    pending = [(path, pid) for path, pid in paths if not path.exists()]
    completed = len(paths)-len(pending)

    def progress():
        atomic_json(out/f'progress-{stage}.json', {'completed': completed, 'total': len(paths)})
        if completed == len(paths) or completed % 20 == 0:
            print(f'{stage} [{completed}/{len(paths)}]', flush=True)

    progress()
    if pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('fork'),
                                 initializer=_set_worker_data, initargs=(data,)) as pool:
            jobs = {pool.submit(_evaluate, function, pid): path for path, pid in pending}
            for job in as_completed(jobs):
                atomic_json(jobs[job], job.result())
                completed += 1
                progress()
    return [read_json(path) for path, _ in paths]


def _set_worker_data(data):
    global _worker_data
    _worker_data = data


def _evaluate(function, pid):
    return function(_worker_data, pid)


def official_checks(records, pair_count):
    checks = [check for record in records for check in record['checks']]
    expected = pair_count*RELABELS_PER_SIDE
    if len(checks) != expected or len({tuple(check['indices']) for check in checks}) != expected:
        raise ValueError(f'Expected all {expected} distinct official checks')
    return checks



def pair_ids_by_size(data, ids, *, largest_first):
    direction = -1 if largest_first else 1
    return sorted(ids, key=lambda pid: (direction*len(data[BLOCK_SIZE*pid]), pid))


def summarize_official(checks, *gap_fields):
    gaps = [max(check[field] for field in gap_fields) for check in checks]
    return {
        'official_total': len(gaps),
        'official_above_threshold': sum(gap > TOLERANCE for gap in gaps),
        'official_max_aligned_gap': max(gaps),
    }
