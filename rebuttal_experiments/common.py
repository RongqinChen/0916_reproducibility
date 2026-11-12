"""Output identity, resource controls and read-only input validation."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import tempfile
import time
from pathlib import Path

from . import PROTOCOL_VERSION


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@contextlib.contextmanager
def atomic_output(path):
    """Write each artifact to a temporary file, then replace its destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        yield Path(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path, value):
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def atomic_npz(path, **arrays):
    import numpy as np
    with atomic_output(path) as temporary:
        np.savez_compressed(temporary, **arrays)


def atomic_npy(path, array):
    import numpy as np
    with atomic_output(path) as temporary:
        np.save(temporary, array, allow_pickle=False)


def configuration(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Configuration must be a JSON object')
    return value


def source_identity():
    package = Path(__file__).resolve().parent
    return {p.name: sha256(p) for p in sorted(package.glob('*.py'))}


def environment():
    versions = {}
    for name in ('numpy', 'networkx', 'torch', 'pynauty', 'psutil'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': versions}


@contextlib.contextmanager
def run_directory(out, config, inputs=(), resume=False):
    """Lock a new run, or resume only with the identical input/code/config identity."""
    import fcntl
    out = Path(out)
    identity = {'protocol': PROTOCOL_VERSION, 'configuration': config,
                'inputs': {str(Path(p).resolve()): sha256(p) for p in inputs},
                'sources': source_identity(), 'environment': environment()}
    existed = out.exists()
    if existed and not resume:
        raise FileExistsError(f'{out} exists; use a new directory or explicit --resume')
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = out / 'manifest.json'
        if existed:
            old = json.loads(manifest.read_text())
            if old['identity'] != identity:
                raise ValueError('Resume identity mismatch; use a new output directory')
        record = {'identity': identity, 'identity_sha256': identity_hash(identity),
                  'status': 'running'}
        atomic_json(manifest, record)
        try:
            yield out, record['identity_sha256']
        except BaseException as exc:
            atomic_json(manifest, {**record, 'status': 'interrupted',
                                  'exception': type(exc).__name__, 'message': str(exc)})
            raise
        else:
            atomic_json(manifest, {**record, 'status': 'execution_finished',
                                  'note': 'Inspect summary.json; unresolved items are not passes.'})


class ResourceLimit(RuntimeError):
    pass


class Budget:
    """Check memory and an optional time limit; seconds=None disables the deadline."""

    def __init__(self, seconds=300, memory_gib=4):
        if ((seconds is not None and (not math.isfinite(seconds) or seconds <= 0))
                or not math.isfinite(memory_gib) or memory_gib <= 0):
            raise ValueError('Positive finite resource limits are required; seconds may be None')
        self.started = time.monotonic()
        self.deadline = None if seconds is None else self.started + seconds
        self.memory_bytes = int(memory_gib * 2**30)

    def check(self, estimated_bytes=0):
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise ResourceLimit('wall_time_limit')
        if estimated_bytes > self.memory_bytes:
            raise ResourceLimit('estimated_memory_limit')
        # psutil is optional for exact-only runs; other dependencies declare it.
        try:
            import psutil
            if psutil.Process().memory_info().rss > self.memory_bytes:
                raise ResourceLimit('resident_memory_limit')
        except ImportError:
            pass


@contextlib.contextmanager
def hard_wall_limit(seconds):
    """Enforce a Unix wall limit, or leave existing timers alone when seconds=None."""
    if seconds is None:
        yield
        return
    import signal
    def expired(signum, frame):
        raise ResourceLimit('wall_time_limit')
    started = time.monotonic()
    previous = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    if old_timer[0] and old_timer[0] < seconds:
        signal.setitimer(signal.ITIMER_REAL, old_timer[0])
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if old_timer[0]:
            remaining = old_timer[0] - (time.monotonic() - started)
            signal.setitimer(signal.ITIMER_REAL, max(1e-6, remaining), old_timer[1])


def simple_adjacency(value):
    import numpy as np
    adjacency = np.asarray(value)
    if (adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]
            or len(adjacency) == 0 or not np.isfinite(adjacency).all()
            or not np.array_equal(adjacency, adjacency.T)
            or np.any(np.diag(adjacency))
            or not np.isin(adjacency, [0, 1]).all()):
        raise ValueError('Expected a nonempty simple undirected binary adjacency matrix')
    return adjacency.astype('int64')


def permutations(n, count, seed):
    import numpy as np
    if count < 0:
        raise ValueError('Negative permutation count')
    rng = random.Random(int(seed))
    result = []
    for _ in range(count):
        order = list(range(n))
        rng.shuffle(order)
        result.append(np.asarray(order, dtype='int64'))
    return result


def topology(path):
    """Read the existing compact topology format without pickle."""
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        counts = data['counts'].astype('int64')
        edges = data['edges'].astype('int64')
        offsets = data['offsets'].astype('int64')
        if (edges.ndim != 2 or edges.shape[0] != 2 or len(offsets) != len(counts) + 1
                or offsets[0] != 0 or offsets[-1] != edges.shape[1]
                or np.any(np.diff(offsets) < 0)):
            raise ValueError('Invalid compact topology format')
        for graph_id, n in enumerate(counts):
            local = edges[:, offsets[graph_id]:offsets[graph_id + 1]]
            adjacency = np.zeros((int(n), int(n)), dtype='int64')
            if local.size and (local.min() < 0 or local.max() >= n):
                raise ValueError('Invalid topology endpoint')
            np.add.at(adjacency, (local[0], local[1]), 1)
            yield graph_id, simple_adjacency(adjacency)
