"""Optional worker phase reports, kept separate from scientific result records."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import time

from .common import atomic_json

_target = ContextVar('brec_progress_target', default=None)


@contextmanager
def report_to(path):
    token = _target.set(Path(path))
    try:
        yield
    finally:
        _target.reset(token)


def report_phase(phase, **details):
    target = _target.get()
    if target is not None:
        atomic_json(target, {'phase': phase, 'started_monotonic': time.monotonic(),
                             'details': details})
