"""Display experiment progress without changing scientific code or run identity."""
from __future__ import annotations

import argparse
import codecs
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time


@dataclass
class Task:
    label: str = 'Experiment'
    kind: str = 'unknown'
    total: int | None = None
    completed: int = 0
    unit: str = 'steps'
    path: Path | None = None
    detail: str = ''
    initial: int = field(init=False)
    seen: set = field(default_factory=set)
    started_at: float = field(default_factory=time.time)

    def __post_init__(self):
        self.initial = self.completed

    def observe(self, line):
        """Return True for routine progress logs that a terminal bar replaces."""
        if self.kind in ('objects', 'protein-prepare'):
            pattern = r'graph (\d+):' if self.kind == 'objects' else r'prepared graph (\d+)/\d+'
            match = re.match(pattern, line)
            if match:
                if match[1] not in self.seen:
                    self.seen.add(match[1])
                    self.completed += 1
                self.detail = line.strip()
                return True
        elif self.kind == 'protein-train':
            match = re.match(r'(fold\d+_seed\d+_\w+) epoch=(\d+)\s', line)
            if match:
                key = match[1], match[2]
                if key not in self.seen:
                    self.seen.add(key)
                    self.completed += 1
                self.detail = line.strip()
                return True
        elif self.kind == 'mechanisms' and line.startswith('Exact controls saved in '):
            self.completed = 1
        return False

    def poll(self):
        if self.kind == 'protein-predict' and self.path:
            self.completed = len(list((self.path / 'jobs').glob('*/prediction.json')))

    def finish(self, returncode):
        self.poll()
        if returncode == 0 and self.kind in ('mechanisms', 'protein-report') and self.path:
            if self.path.exists() and self.path.stat().st_mtime >= int(self.started_at):
                self.completed = 1


def describe(command):
    """Read totals and existing artifacts; never create or modify run files."""
    try:
        index = command.index('-m')
        module, arguments = command[index + 1], command[index + 2:]
        if module != 'rebuttal_experiments':
            return Task()
        kind = arguments[0]
        if kind == 'protein':
            kind += '-' + arguments[1]
        parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
        for name in ('config', 'out', 'dataset', 'run'):
            parser.add_argument('--' + name)
        for name in ('folds', 'seeds', 'variants'):
            parser.add_argument('--' + name, nargs='+')
        args, _ = parser.parse_known_args(arguments)
        label = kind.replace('-', ' ').title()
        path = Path(args.run if kind == 'protein-predict' else args.out) if (
            args.run if kind == 'protein-predict' else args.out) else None
        if kind in ('mechanisms', 'protein-report'):
            artifact = path / 'summary.json' if kind == 'mechanisms' and path else path
            return Task(label, kind, 1, path=artifact)
        config_name = 'protein' if kind.startswith('protein-') else kind.replace('-', '_')
        config = json.loads(Path(args.config or f'configs/rebuttal/{config_name}.json').read_text())
        completed = 0
        if kind in ('objects', 'protein-prepare'):
            total = (config['expected_graphs'][args.dataset or config['dataset']]
                     if kind == 'protein-prepare' else config['expected_graphs'])
            completed = len(list((path / 'records').glob('*.json'))) if path else 0
            unit = 'graphs'
        elif kind == 'protein-train':
            jobs = ((len(args.folds) if args.folds else config['folds'])
                    * len(args.seeds if args.seeds else config['seeds'])
                    * len(args.variants if args.variants else ('base', 'joint', 'nonjoint')))
            total, unit = jobs * config['epochs'], 'epochs'
            completed = (len(list((path / 'jobs').glob('*/complete.json'))) * config['epochs']
                         if path else 0)
        elif kind == 'protein-predict':
            total = len(list((path / 'jobs').glob('*/complete.json'))) if path else None
            completed = len(list((path / 'jobs').glob('*/prediction.json'))) if path else 0
            unit = 'jobs'
        else:
            return Task(label, kind)
        return Task(label, kind, total, completed, unit, path)
    except (OSError, ValueError, TypeError, KeyError, IndexError, argparse.ArgumentError):
        # The experiment reports its own input/configuration errors and exit code.
        return Task()


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class Progress:
    def __init__(self, task, stream=None, clock=time.monotonic, *, enabled=True):
        self.task = task
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock
        self.started = clock()
        self.last_draw = float('-inf')
        self.enabled = enabled
        self.interactive = enabled and self.stream.isatty()
        self.displayed = False

    def clear(self):
        if self.interactive and self.displayed:
            self.stream.write('\r\033[K')
            self.stream.flush()
            self.displayed = False

    def refresh(self, force=False, status='running'):
        if not self.enabled:
            return
        now = self.clock()
        if not force and now - self.last_draw < (1 if self.interactive else 30):
            return
        task = self.task
        elapsed = now - self.started
        advanced = task.completed - task.initial
        eta = duration(elapsed / advanced * max(0, task.total - task.completed)) if (
            task.total is not None and advanced > 0) else '--:--:--'
        columns = max(40, shutil.get_terminal_size((120, 24)).columns - 1)
        width = max(5, min(20, columns - 90)) if self.interactive else 20
        if task.total is not None:
            fraction = min(1, task.completed / max(1, task.total))
            filled = int(width * fraction)
            bar = '=' * filled + '.' * (width - filled)
            count = f'{task.completed}/{task.total} {task.unit} {fraction:5.1%}'
        else:
            position = int(elapsed) % width
            bar = '.' * position + '>' + '.' * (width - position - 1)
            count = 'in progress'
        line = (f'{task.label} [{bar}] {count} | elapsed {duration(elapsed)} '
                f'| ETA {eta} | {status}')
        if self.interactive:
            if len(line) > columns and task.total is not None:
                line = (f'{task.label} [{bar}] {task.completed}/{task.total} {fraction:.1%}'
                        f' | {duration(elapsed)} | ETA {eta} | {status}')
            if task.detail:
                line += ' | ' + task.detail
            self.stream.write('\r\033[K' + line[:columns])
        else:
            self.stream.write(line + '\n')
        self.stream.flush()
        self.displayed = True
        self.last_draw = now

    def finish(self, returncode):
        self.task.finish(returncode)
        self.refresh(force=True, status='finished' if returncode == 0 else f'failed ({returncode})')
        if self.interactive:
            self.stream.write('\n')
            self.stream.flush()


def run(command, *, show_progress=True):
    task = describe(command)
    progress = Progress(task, enabled=show_progress)
    progress.refresh(force=True)
    process = None
    previous = {}
    pending_signal = None

    def forward(signum, frame):
        nonlocal pending_signal
        pending_signal = signum
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    selector = selectors.DefaultSelector()

    def emit(text, channel):
        routine = task.observe(text) if channel['stdout'] else False
        if routine and progress.interactive and sys.stdout.isatty():
            return
        if channel['destination'].isatty():
            progress.clear()
        channel['destination'].write(text)
        channel['destination'].flush()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, forward)
        if pending_signal is not None:
            progress.finish(128 + pending_signal)
            return 128 + pending_signal
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True, bufsize=0)
        if pending_signal is not None:
            forward(pending_signal, None)
        for source, destination in ((process.stdout, sys.stdout), (process.stderr, sys.stderr)):
            selector.register(source, selectors.EVENT_READ, {
                'destination': destination, 'decoder': codecs.getincrementaldecoder('utf-8')('replace'),
                'buffer': '', 'stdout': source is process.stdout})
        while selector.get_map() or process.poll() is None:
            for key, _ in selector.select(timeout=0.2):
                channel = key.data
                chunk = os.read(key.fileobj.fileno(), 65536)
                channel['buffer'] += channel['decoder'].decode(chunk, final=not chunk)
                while '\n' in channel['buffer']:
                    line, channel['buffer'] = channel['buffer'].split('\n', 1)
                    emit(line + '\n', channel)
                if not chunk:
                    if channel['buffer']:
                        emit(channel['buffer'], channel)
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
            task.poll()
            progress.refresh()
        returncode = process.wait()
        progress.finish(returncode)
        return returncode if returncode >= 0 else 128 - returncode
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        selector.close()
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        if process is not None:
            process.stdout.close()
            process.stderr.close()


def main():
    command = sys.argv[1:]
    show_progress = True
    if command[:1] == ['--no-progress']:
        show_progress = False
        command = command[1:]
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        raise SystemExit('Usage: python -m tools.rebuttal_progress -- COMMAND [ARGS...]')
    raise SystemExit(run(command, show_progress=show_progress))


if __name__ == '__main__':
    main()
