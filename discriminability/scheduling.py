"""Bounded graph scheduling and atomic, identity-checked checkpoints."""

import json
import os
import psutil
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from .pipeline import init_worker, worker


def estimated_bytes(task, config):
    n = task[3]
    domains=config.get('domains', ('nodes','edges','pairs'))
    # Edges are stored in both orientations; adjacency validation rejects duplicates.
    objects={'nodes':n, 'edges':task[2].shape[1]//2, 'pairs':n*(n-1)//2}
    m=max(objects[d] for d in domains)
    budgets=config.get('degree_budgets')
    conditions=len(budgets or config['dims'])
    routes=len(config.get('routes', ('powers','endpoints','mixed','powers_2d','endpoints_2d')))
    # The current domain's object-pair storage: evaluate_domain keeps a few dense
    # boolean matrices live (eligibility, orbit mask, WL mask, MIS scratch) and
    # stores every (condition, route) similarity mask packed at one bit per pair.
    # The earlier constant reserved 32 unpacked bytes per pair, ~5x what the
    # runner allocates, which throttled admission to a handful of workers on
    # edge-heavy datasets (PPA: E^2 dominates).
    masks=(4 + conditions*routes/8)*m*m
    if budgets:
        q=max(budgets)
        width=max(q,sum(t//2+1 for t in range(1,q)))
    else:
        width=2*max(config['dims'])
    # All selected original/relabelled features remain live. Polynomial banks
    # and exact 2-FWL still use dense n^2 / n^3 storage even for nodes alone.
    feature_objects=sum(objects[d]*(1 if d=='nodes' else 2) for d in domains)
    features=feature_objects*width*2*4*(1+config['permutation_checks'])
    banks=(width+2)*n*n*4*2
    references=32*n**3
    return int(256*1024**2+masks+features+banks+references)


def worker_private_bytes(process) -> int:
    """Resident bytes that belong to one worker alone.

    ``memory_info().rss`` also counts copy-on-write pages a forked worker still
    shares with its parent, and ``statm``/``memory_info().shared`` does not
    expose inherited anonymous pages. A large parent therefore makes every idle
    child look as big as the parent: with a many-graph topology dump (PPA keeps
    ~6 GiB of edge slices), each of the 16 workers reported ~6 GiB RSS, the
    admission heuristic concluded that live children already used the whole
    budget, and the run aborted with a spurious MemoryError before doing any
    work. ``smaps_rollup`` separates private from shared pages, so only the
    bytes a worker itself needs enter the admission decision.
    """
    try:
        with open(f'/proc/{process.pid}/smaps_rollup') as handle:
            private = 0
            for line in handle:
                if line.startswith(('Private_Clean:', 'Private_Dirty:')):
                    private += int(line.split()[1]) * 1024
        return private
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    try:
        return process.memory_info().rss
    except psutil.NoSuchProcess:
        return 0


def scheduled_results(tasks, config, checkpoint_dir=None):
    tasks = sorted(tasks, key=lambda task: (-task[3], task[0]))
    root = Path(checkpoint_dir) if checkpoint_dir else None
    if root:
        root.mkdir(parents=True, exist_ok=True)
    pending = []
    for task in tasks:
        path = root / f'{task[0]}.json' if root else None
        if path and path.exists():
            yield json.loads(path.read_text())
        else:
            pending.append(task)

    def save(task, result):
        if root:
            path = root / f'{task[0]}.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(result, allow_nan=False))
            os.replace(temporary, path)
        return result

    allowance = int(config['memory_gb'] * 1024**3)
    oversized=[task for task in pending if estimated_bytes(task,config)>allowance]
    pending=[task for task in pending if estimated_bytes(task,config)<=allowance]
    def report_oversized():
        if oversized:
            info=[dict(graph_index=t[0],nodes=t[3],estimated_gib=estimated_bytes(t,config)/1024**3) for t in oversized]
            raise ValueError(f'{len(info)} graphs exceed --memory-gb; feasible graphs were retained, '
                             f'run remains incomplete. Examples: {info[:10]}')
    if config['workers'] == 1:
        init_worker(config)
        for task in pending:
            yield save(task, worker(task))
        report_oversized()
        return
    with ProcessPoolExecutor(max_workers=config['workers'], initializer=init_worker,
                             initargs=(config,)) as pool:
        active = {}
        used = 0
        while pending or active:
            while len(active) < config['workers']:
                children=psutil.Process().children(recursive=True)
                live=0
                for child in children:
                    live+=worker_private_bytes(child)
                room=min(allowance-max(used,live),
                         psutil.virtual_memory().available-8*1024**3-max(0,used-live))
                index = next((i for i, task in enumerate(pending)
                              if estimated_bytes(task, config) <= room), None)
                if index is None:
                    break
                task = pending.pop(index)
                cost = estimated_bytes(task, config)
                active[pool.submit(worker, task)] = task, cost
                used += cost
            if not active:
                if pending:
                    raise MemoryError('No task fits current available RAM with 8 GiB system reserve. '
                                      'Completed streamed/checkpoint results are retained; resume when RAM is free.')
                break
            done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                task, cost = active.pop(future)
                used -= cost
                yield save(task, future.result())

    report_oversized()
