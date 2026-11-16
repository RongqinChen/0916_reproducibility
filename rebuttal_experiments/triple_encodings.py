"""Encoding-only exact BREC triples: no neural model or GPU training."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import partial
import json
import math
from pathlib import Path
import time
import traceback

from brec.determinism import prepare_environment, ensure_startup, start_brec
prepare_environment()
import numpy as np

from .brec import graph_at, load_brec_data
from .common import (Budget, ResourceLimit, atomic_json, atomic_npz, configuration,
                     run_directory, sha256)
from .encodings import descriptors
from .scheduling import ordered_cpus, run_tasks
from .triple_encoding_compare import compare_kernels
from .triple_encoding_controls import controls
from .triple_exact import TripleKernel, ordered_triple_orbits, clear_caches
from .triple_ppgn import freeze_sources


def settings(config,q=None):
    for degree in config['q_values'] if q is None else [q]:
        for basis in config['bases']:
            for variant in config['variants']:
                yield basis,variant,degree,'multiset',1
                for p in config['mean_p_values']:
                    yield basis,variant,degree,'mean',p


def stem(pid,basis,variant,q,readout,p):
    return f'{basis}_{variant}_q{q}_{readout}_p{p}_pair{pid:03d}'


def validate_record(row,identity,pid,basis,variant,q,readout,p,*,allow_failure=False):
    expected = {'identity_sha256':identity,'pair_id':pid,'basis':basis,'variant':variant,
                'q':q,'readout':readout,'p':p,'order':3,'arithmetic':'exact_radical'}
    if any(row.get(key)!=value for key,value in expected.items()):
        raise ValueError('Triple encoding record condition or identity mismatch')
    allowed = {'certified_different','certified_equal','resource_limited'}
    if allow_failure:
        allowed.add('implementation_error')
    if row.get('status') not in allowed:
        raise ValueError('Inspect failed triple encoding record before resuming')


def orbit_cache(out,adjacency,identity):
    digest = sha256_array(adjacency)
    path = out/'orbits'/f'{digest}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as value:
            if str(value['identity']) != identity or not np.array_equal(value['adjacency'],adjacency):
                raise ValueError('Orbit cache identity mismatch')
            triples,weights = value['triples'],value['weights']
    else:
        triples,weights = ordered_triple_orbits(adjacency)
        atomic_npz(path,identity=np.asarray(identity),adjacency=adjacency,triples=triples,weights=weights)
    if (len(triples)!=len(weights) or int(weights.sum())!=len(adjacency)**3
            or np.any(weights<=0) or np.any(triples<0) or np.any(triples>=len(adjacency))):
        raise ValueError('Invalid ordered-triple orbit cache')
    return triples,weights


def sha256_array(adjacency):
    import hashlib
    return hashlib.sha256(adjacency.astype('int64').tobytes()).hexdigest()


def evaluate_pair(data,config,out,identity,task):
    pid,q = task['record']['pair_id'],task['record']['q']
    started = time.monotonic()
    results = {}
    try:
        budget = Budget(config['seconds_per_condition'],config['memory_gib'])
        graphs = [graph_at(data,64*pid+i) for i in (0,1)]
        compressed = [orbit_cache(out,graph,identity) for graph in graphs]
        triples,weights = list(zip(*compressed))
        for basis in config['bases']:
            kernels = [TripleKernel(graph,q,basis,budget) for graph in graphs]
            result = compare_kernels(kernels,triples,weights,q,basis,config['mean_p_values'],budget)
            checks = controls(data,pid,kernels,q,basis,config,budget)
            for variant in config['variants']:
                for readout,p in [('multiset',1),*(('mean',p) for p in config['mean_p_values'])]:
                    decision = result[variant]['multiset' if readout=='multiset' else f'mean_p{p}']
                    results[basis,variant,readout,p] = {
                        **decision,'controls':checks,'orbit_representatives':[len(t) for t in triples],
                        'objects':[int(w.sum()) for w in weights],
                        'width':len(descriptors(q,3,'endpoint' if variant=='baseline' else 'joint'))}
            clear_caches()
        if len(config['bases'])==2:
            for variant in config['variants']:
                if len({results[basis,variant,'multiset',1]['status'] for basis in config['bases']})!=1:
                    raise AssertionError('Exact full-multiset decisions disagree under an invertible basis change')
        status = 'complete'
    except (ResourceLimit,MemoryError) as exc:
        status = 'resource_limited'
        error = {'status':status,'reason':str(exc)}
    except Exception as exc:
        status = 'implementation_error'
        error = {'status':status,'reason':repr(exc),'traceback':traceback.format_exc()}
    for basis,variant,degree,readout,p in settings(config,q):
        target = out/'records'/f'{stem(pid,basis,variant,degree,readout,p)}.json'
        if target.exists():
            continue
        row = {'pair_id':pid,'basis':basis,'variant':variant,'q':degree,'readout':readout,'p':p,
               'order':3,'arithmetic':'exact_radical','identity_sha256':identity,
               'elapsed_pair_seconds':time.monotonic()-started}
        row.update(results[basis,variant,readout,p] if status=='complete' else error)
        atomic_json(target,row)
    clear_caches()
    meta = {**task['record'],'status':status,'elapsed_seconds':time.monotonic()-started}
    if status!='complete':
        meta.update(error)
    return meta


def summarize(out,config,identity):
    rows = []
    for basis,variant,q,readout,p in settings(config):
        counts = Counter()
        for pid in config['executed_pair_ids']:
            path = out/'records'/f'{stem(pid,basis,variant,q,readout,p)}.json'
            if path.exists():
                row = json.loads(path.read_text())
                # Failed records remain visible in summaries; resume rejects them.
                validate_record(row,identity,pid,basis,variant,q,readout,p,allow_failure=True)
                counts[row['status']] += 1
            else:
                job = out/'jobs'/f'q{q}_pair{pid:03d}.json'
                status = json.loads(job.read_text())['status'] if job.exists() else 'not_executed'
                counts[status if status in ('resource_limited','implementation_error') else 'not_executed'] += 1
        counts['not_requested'] += 400-len(config['executed_pair_ids'])
        rows.append({'basis':basis,'variant':variant,'q':q,'readout':readout,'p':p,
                     'denominator':400,'counts':dict(counts),
                     'certified_detection_lower_bound':counts['certified_different'],
                     'detection_upper_bound':400-counts['certified_equal']})
    atomic_json(out/'summary.json',{'settings':rows,
        'scope':'Exact ordered-triple encoding comparisons, weighted over all n^3 tuples; no neural training. Controls align every bank entry and sample full derived rows.'})


def validate(config):
    for key in ('q_values','mean_p_values'):
        value = config.get(key)
        if (not value or value!=sorted(set(value)) or
                any(type(x) is not int or x<1 for x in value)):
            raise ValueError(f'Increasing distinct positive integers required: {key}')
    if config['bases']!=['monomial','chebyshev'] or config['variants']!=['baseline','joint']:
        raise ValueError('This protocol compares both bases and baseline/joint')
    for key,minimum in (('relabels',1),('reliability_relabels',0),('control_triples',1),('seed',0)):
        if type(config[key]) is not int or config[key]<minimum:
            raise ValueError(f'Invalid {key}')
    if not math.isfinite(config['memory_gib']) or config['memory_gib']<=0:
        raise ValueError('Positive finite RAM budget required')
    scale = config.get('memory_reservation_scale',1.0)
    if type(scale) not in (int,float) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('memory_reservation_scale must be in (0,1]')
    if config['seconds_per_condition'] is not None:
        raise ValueError('This frozen protocol has no per-condition deadline')


def main(argv=None):
    ensure_startup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/rebuttal/triple_encodings.json'))
    parser.add_argument('--brec-file',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pair-ids',nargs='+',type=int)
    parser.add_argument('--workers',type=int,default=max(1,len(ordered_cpus())))
    parser.add_argument('--memory-gib',type=float,
                        help='Override the total scheduler+worker RSS budget in GiB')
    parser.add_argument('--reserve-gib',type=float,default=8)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(argv)
    config = configuration(args.config)
    if args.memory_gib is not None:
        config['memory_gib'] = args.memory_gib
    validate(config)
    ids = args.pair_ids if args.pair_ids is not None else list(range(400))
    if (not ids or len(set(ids))!=len(ids) or any(pid<0 or pid>=400 for pid in ids)
            or args.workers<1 or not math.isfinite(args.reserve_gib) or args.reserve_gib<0):
        parser.error('Valid pair IDs, positive workers and nonnegative system reserve required')
    config.update(executed_pair_ids=ids,determinism=start_brec(config['seed']),
                  scheduling={'workers':args.workers,'reserve_gib':args.reserve_gib},
                  evaluation='weighted-automorphism-orbits; all ordered triples including repetitions',
                  controls='all bank entries; sampled complete derived rows under known relabelings')
    data = load_brec_data(args.brec_file,config['expected_brec_sha256'])
    with run_directory(args.out.resolve(),config,[args.config.resolve(),args.brec_file.resolve()],args.resume) as (out,identity):
        (out/'records').mkdir(exist_ok=True)
        freeze_sources(out)
        metadata = {}
        for count,pid in enumerate(ids,1):
            compressed = [orbit_cache(out,graph_at(data,64*pid+i),identity) for i in (0,1)]
            metadata[pid] = [len(rows) for rows,_ in compressed]
            print(f'prepared triple orbits {count}/{len(ids)} pair{pid:03d}: {metadata[pid]}',flush=True)
        last_summary = 0
        def update_summary(task,result):
            nonlocal last_summary
            if time.monotonic()-last_summary>30:
                summarize(out,config,identity)
                last_summary = time.monotonic()
        try:
            summarize(out,config,identity)
            for q in config['q_values']:
                tasks = []
                for pid in ids:
                    conditions = list(settings(config,q))
                    targets = [out/'records'/f'{stem(pid,b,v,q,r,p)}.json'
                               for b,v,_,r,p in conditions]
                    existing = []
                    for path,(b,v,_,r,p) in zip(targets,conditions):
                        if not path.exists():
                            continue
                        row = json.loads(path.read_text())
                        validate_record(row,identity,pid,b,v,q,r,p)
                        existing.append(path)
                    if len(existing)==len(targets):
                        continue
                    (out/'jobs'/f'q{q}_pair{pid:03d}.json').unlink(missing_ok=True)
                    # Scalar columns and partition IDs are streamed; no n^3 x width table.
                    reservation = 768*2**20+sum(metadata[pid])*4096
                    tasks.append({'stem':f'q{q}_pair{pid:03d}',
                                  'record':{'pair_id':pid,'q':q,'identity_sha256':identity},
                                  'estimated_bytes':reservation,'work':sum(metadata[pid])*q**3})
                print(f'encoding-only triple stage Q={q}: {len(tasks)} pair jobs',flush=True)
                run_tasks(tasks,partial(evaluate_pair,data,config,out,identity),out,
                          workers=args.workers,memory_gib=config['memory_gib'],reserve_gib=args.reserve_gib,
                          seed=config['seed'],record_directory='jobs',on_result=update_summary,
                          reservation_scale=config.get('memory_reservation_scale',1.0))
        finally:
            summarize(out,config,identity)


if __name__=='__main__':
    main()
