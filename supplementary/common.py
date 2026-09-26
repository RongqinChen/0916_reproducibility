"""Result identity and data loading shared by the supplementary entry points."""
from __future__ import annotations

import hashlib
import atexit
import fcntl
import json
import os
import platform
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def begin_run(args, protocol, extra_sources=()):
    """Bind resume to the exact arguments, dataset, source and environment."""
    package = Path(__file__).resolve().parent
    sources = (list(package.glob('*.py'))
               + [Path(p).resolve() for p in extra_sources])
    identity = {
        'protocol': protocol,
        'configuration': {k: str(v) if isinstance(v, Path) else v
                          for k, v in vars(args).items()
                          if k not in ('out_dir', 'resume')},
        'dataset_sha256': checksum(args.brec_file),
        'sources': {str(p.relative_to(package.parent)): checksum(p) for p in sources},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'networkx': nx.__version__, 'platform': platform.platform()},
    }
    path = args.out_dir / 'manifest.json'
    existed = args.out_dir.exists()
    if existed and not args.resume:
        raise ValueError('Output exists; select a new directory or use --resume')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.out_dir / '.writer.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError('Another process owns this output directory')
    atexit.register(lock.close)
    if existed:
        previous = json.loads(path.read_text())
        if previous['identity'] != identity:
            raise ValueError('Resume identity mismatch; start a new output directory')
    atomic_json(path, {'identity': identity, 'status': 'running'})
    return identity


def finish_run(args, identity, statuses):
    atomic_json(args.out_dir / 'manifest.json', {
        'identity': identity, 'status': 'complete' if all(s == 'complete' for s in statuses)
        else 'finished_with_missing_measurements', 'condition_statuses': statuses,
    })


def load_brec(path, trust_pickle=False):
    raw = np.load(path, allow_pickle=trust_pickle)
    if raw.ndim != 1 or len(raw) != 51200:
        raise ValueError('Expected official 51200-entry graph6 BREC layout')
    return raw


def graph_at(raw, index):
    value = raw[index]
    value = value.encode() if isinstance(value, str) else bytes(value)
    return nx.to_numpy_array(nx.from_graph6_bytes(value.strip()), dtype=np.float64)


def pair_graphs(raw, pair_id, relabels=32):
    return ([graph_at(raw, pair_id * 64 + i) for i in range(2 * relabels)],
            [graph_at(raw, (pair_id + 400) * 64 + i) for i in range(2 * relabels)])


def validate_pair_ids(ids):
    if not ids or len(ids) != len(set(ids)) or any(i < 0 or i >= 400 for i in ids):
        raise ValueError('Specify distinct BREC pair IDs in 0..399')


def read_pair_ids(path):
    """Read whitespace/comma separated pair IDs and apply shared checks."""
    text = Path(path).read_text().replace(',', ' ')
    try:
        values = [int(token) for token in text.split()]
    except ValueError as exc:
        raise ValueError('Pair-list files must contain integer IDs') from exc
    validate_pair_ids(values)
    return values


def write_csv(path, rows):
    import csv
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
