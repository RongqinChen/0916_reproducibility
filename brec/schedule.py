"""Frozen, resumable, memory-aware single-machine BREC execution (POSIX)."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import psutil
import subprocess
import sys
import time
try:  # package entry point; frozen scheduler copies use the legacy flat names
    from .validation import (validate_run, validate_checkpoint, protocol_identity,
                             expected_keys)
except ImportError:
    from brec_validation import (validate_run, validate_checkpoint, protocol_identity,
                                 expected_keys)

GIB = 1024**3
FILES = ('brec_experiments.py', 'brec_encodings.py', 'budget_sweep.py',
         'brec_theory.py', 'brec_report.py', 'brec_schedule.py', 'brec_budget_cache.py',
         'brec_validation.py')
SOURCE_FILES = {
    'brec_experiments.py': 'experiments.py',
    'brec_encodings.py': 'encodings.py',
    'budget_sweep.py': 'budget_sweep.py',
    'brec_theory.py': 'theory.py',
    'brec_report.py': 'report.py',
    'brec_schedule.py': 'schedule.py',
    'brec_budget_cache.py': 'budget_cache.py',
    'brec_validation.py': 'validation.py',
}


def source_path(legacy_name):
    """Resolve a package source or its legacy name in a frozen flat snapshot."""
    directory = Path(__file__).resolve().parent
    legacy = directory / legacy_name
    return legacy if legacy.is_file() else directory / SOURCE_FILES[legacy_name]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def available_memory():
    path = Path('/proc/meminfo')
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1])*1024
    return None


def estimate(n, q, order, variants, decimals, readouts, chunk):
    # Conservative initial reservations, not measured RSS guarantees.
    try:
        from .encodings import channel_count
    except ImportError:
        from brec_encodings import channel_count
    widths = [channel_count(order, q, v) for v in variants]
    counters = 4*n**order*160*len(widths)*len(decimals) if 'multiset' in readouts else 0
    memory = (512*1024**2 + counters + chunk*max(widths)*8*12 +
              (q+1)*n*n*8*4)
    work = n**order*sum(widths)
    if 'joint' in variants:
        work += n**4*(q+1)**3
    return memory, work


def plan(args):
    import numpy as np
    import networkx as nx
    try:
        from .theory import group_for_pair, impossible
    except ImportError:
        from brec_theory import group_for_pair, impossible
    if args.evaluation_mode == 'direct':
        args.relabels = 1
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    frozen = root/'frozen'
    frozen.mkdir()
    for name in FILES:
        shutil.copyfile(source_path(name), frozen/name)
    data = args.brec_file.resolve()
    raw = np.load(data, allow_pickle=args.trust_pickle)
    if raw.ndim != 1 or len(raw) != 51200:
        raise ValueError('Expected official 51200 graph6 entries')
    common = ['--brec-file', str(data), '--p-values', *map(str,args.p_values),
              '--readouts', *args.readouts, '--decimals', *map(str,args.decimals),
              '--evaluation-mode', args.evaluation_mode,
              '--relabels', str(args.relabels), '--permutation-checks', ('0' if args.evaluation_mode=='direct' else '1'),
              '--compare-tolerance', str(args.compare_tolerance),
              '--chunk-size', str(args.chunk_size), '--joint-normalise', args.joint_normalise,
              '--no-trace-allocations',
              '--max-budget-slices' if args.max_budget_slices else '--budget-ladder']
    if args.trust_pickle:
        common.append('--trust-pickle')
    if args.joint_scale_audit:
        common.append('--joint-scale-audit')
    jobs = []
    def configuration(ids,bases,orders,variants):
        return dict(pair_ids=ids,bases=bases,orders=orders,variants=variants,
                    q_values=args.q_values,p_values=args.p_values,readouts=args.readouts,
                    decimals=args.decimals,relabels=args.relabels,permutation_checks=0 if args.evaluation_mode=="direct" else 1,
                    evaluation_mode=args.evaluation_mode,
                    compare_tolerance=args.compare_tolerance,joint_normalise=args.joint_normalise,
                    chunk_size=args.chunk_size,seed=0,budget_ladder=not args.max_budget_slices,
                    max_budget_slices=args.max_budget_slices,joint_scale_audit=args.joint_scale_audit)
    for order in args.orders:
        blocked = [i for i in range(400) if impossible(group_for_pair(i),order)]
        pairlist = root/f'excluded_order{order}.txt'
        pairlist.write_text('\n'.join(map(str,blocked)))
        variants = [v for v in args.variants if not(order==2 and v=='joint')]
        if not variants:
            continue
        # One cheap metadata-only invocation preserves all theoretical coverage.
        jobs.append(dict(id=f'excluded_o{order}', memory=512*1024**2, work=0,
                         configuration=configuration(blocked,args.bases,[order],variants),
                         command=common+['--pair-list',str(pairlist),'--orders',str(order),
                         '--bases',*args.bases,'--q-values',*map(str,args.q_values),
                         '--variants',*variants]))
        for pid in range(400):
            if pid in blocked:
                continue
            sizes=[]
            for offset in ((pid*64,) if args.evaluation_mode=="direct" else (pid*64,(pid+400)*64)):
                for i in range(args.relabels*2):
                    value=raw[offset+i]
                    value=value.encode() if isinstance(value,str) else bytes(value)
                    sizes.append(len(nx.from_graph6_bytes(value.strip())))
            n=max(sizes)
            for basis in args.bases:
                q=max(args.q_values)
                memory,work=estimate(n,q,order,variants,args.decimals,args.readouts,args.chunk_size)
                try:
                    from .encodings import channel_count
                except ImportError:
                    from brec_encodings import channel_count
                width=max(channel_count(order,q,v) for v in variants)
                # Per-graph cached Python channel keys and moment lists.
                memory+=(2 if args.evaluation_mode=="direct" else 4*args.relabels+1)*width*(400+64*max(args.p_values))
                jobs.append(dict(id=f'p{pid:03d}_{basis}_o{order}', n=n,
                                 configuration=configuration([pid],[basis],[order],variants),
                                 memory=memory,work=work*len(args.q_values)*len(args.p_values),
                                 command=common+['--pair-shard',str(pid),str(pid+1),
                                 '--orders',str(order),'--bases',basis,'--q-values',
                                 *map(str,sorted(args.q_values)),'--variants',*variants,
                                 *([] if args.max_budget_slices else ['--stage-limit','1'])]))
    for job in jobs:
        job['memory']=stage_request(job,{}, {})['bytes']
    save(root/'plan.json',dict(dataset=str(data),dataset_sha256=digest(data),
         sources={name:digest(frozen/name) for name in FILES},jobs=jobs))
    print(f'Planned {len(jobs)} jobs in {root}; no experiments launched.')


def stage_request(job, entry, calibration):
    """Request resources for the next committed-budget boundary, not Qmax."""
    config=job['configuration']
    if config.get('max_budget_slices'):
        return slice_request(job, entry, calibration)
    checkpoint=None
    if entry.get('output') and (Path(entry['output'])/'checkpoint.json').exists():
        # Integrity is checked at startup, completion, and immediately before
        # launch. Resource polling must not repeatedly hash large cache files.
        checkpoint=json.loads((Path(entry['output'])/'checkpoint.json').read_text())
    if 'n' not in job:
        return dict(bytes=512*1024**2,raw=512*1024**2,key='metadata',stage=None)
    n=job['n']
    stage=(checkpoint['next_stage'] if checkpoint else
           dict(p=min(config['p_values']),q=min(config['q_values']),order=config['orders'][0],
                requested={v:config['readouts'] for v in config['variants']}))
    # A crash after the final commit needs only a small finalization process.
    requested=stage['requested'] if stage else {}
    q=stage['q'] if stage else min(config['q_values'])
    power=stage['p'] if stage else min(config['p_values'])
    variants=list(requested)
    reads=sorted({r for values in requested.values() for r in values})
    order=config['orders'][0]
    width=0
    try:
        from .encodings import channel_count
    except ImportError:
        from brec_encodings import channel_count
    if variants:
        width=max(channel_count(order,q,v) for v in variants)
    cache_bytes=0
    if checkpoint:
        cache_bytes=(Path(entry['output'])/checkpoint['moments_file']).stat().st_size*8
    if 'mean' in reads:
        graphs=(2 if config.get('evaluation_mode')=='direct' else 4*config.get('relabels',1)+config.get('permutation_checks',1))
        cache_bytes=max(cache_bytes,graphs*width*(400+64*power))
    # Direct mean-only P<=2 does not construct tuple rows. Higher joint moments do.
    streamed='multiset' in reads or ('joint' in variants and power>2)
    buffers=128*1024**2 + (config['chunk_size']*width*8*12 if streamed else 0)
    banks=(q+1)*n*n*8*5
    floor=384*1024**2+buffers+banks+cache_bytes
    counters=(4*n**order*160*sum('multiset' in rs for rs in requested.values())
              *len(config['decimals']))
    raw=floor+counters
    # Calibration is local to order/readout/variant/power phase and size bucket.
    key=json.dumps([order,sorted((v,sorted(rs)) for v,rs in requested.items()),
                    config.get('bases',[]),q,power,((n+31)//32)*32])
    observed=calibration.get(key,[])
    # Do not use one unusually small sample to collapse reservations.
    factor=1.0
    if len(observed)>=3:
        factor=max(.25,min(4.,max(observed[-20:])*1.25))
    reservation=max(floor,raw*factor)
    return dict(bytes=reservation,raw=raw,key=key,stage=stage)


def slice_request(job, entry, calibration):
    """Reserve all Q counters plus live comparison references, not one Q only."""
    if 'n' not in job:
        return dict(bytes=128*1024**2, raw=128*1024**2, key='metadata',
                    stage=None, disk=0)
    try:
        from .encodings import channel_count
    except ImportError:
        from brec_encodings import channel_count
    config=job['configuration']
    n=job['n']; order=config['orders'][0]; q=max(config['q_values'])
    variants=config['variants']; budgets=len(config['q_values'])
    width=max(channel_count(order,q,v) for v in variants)
    multisets=len(variants)*len(config['decimals']) if 'multiset' in config['readouts'] else 0
    # Python digest bytes, ints and Counter table overhead, with allocation headroom.
    counters=n**order*multisets*192
    buffers=config['chunk_size']*width*4*10
    moments=max(config['p_values'])*width*4*4
    graphs=2 if config.get('evaluation_mode')=='direct' else 4*config['relabels']+config['permutation_checks']
    audit_bytes=graphs*3*max(config['p_values'])*channel_count(3,4,'joint')*4*3 if config.get('joint_scale_audit') and order==3 else 0
    raw=160*1024**2 + audit_bytes + counters*(budgets+4) + buffers + moments + (q+1)*n*n*4*5
    key='slices:'+json.dumps([order,variants,config['readouts'],config['bases'],
                             config['q_values'],config['p_values'],config['decimals'],
                             (n+15)//16])
    observed=calibration.get(key,[])
    # Only completed same-shape jobs can reduce an initially conservative estimate.
    factor=max(.15,min(4.,max(observed[-20:])*1.3)) if len(observed)>=3 else 1.
    reservation=max(160*1024**2+buffers+moments,raw*factor,
                    entry.get('retry_memory_bytes',0))
    graphs=(2 if config.get('evaluation_mode')=='direct' else 4*config['relabels']+config['permutation_checks'])
    disk=graphs*(n**order*multisets*budgets*64 + width*max(config['p_values'])*4*budgets)
    return dict(bytes=reservation,raw=raw,key=key,stage='all-Q/P',disk=disk)


def ordered_cpus():
    """Allowed CPUs, physical cores before SMT siblings, interleaved sockets."""
    if not hasattr(os, 'sched_getaffinity'):
        return []
    cores={}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            key=(int((topology/'physical_package_id').read_text()),
                 int((topology/'core_id').read_text()))
        except OSError:
            key=(0,cpu)
        cores.setdefault(key,[]).append(cpu)
    keys=sorted(cores,key=lambda key:(key[1],key[0]))
    return [cores[key][rank] for rank in range(max(map(len,cores.values()),default=0))
            for key in keys if rank<len(cores[key])]


def needs_drain(head_fits, head_wait_seconds, running_count, delay):
    """Drain only for a resource-blocked old queue head, never just a long worker."""
    return bool(running_count and not head_fits and head_wait_seconds>=delay)


def check_job(job, directory, spec, partial=False):
    result=validate_run(directory,allow_partial=partial)
    manifest=result['manifest']
    config=manifest['configuration']
    if manifest['dataset_sha256']!=spec['dataset_sha256']:
        raise ValueError('Job dataset identity mismatch')
    required=('brec_experiments.py','brec_encodings.py','budget_sweep.py',
              'brec_theory.py','brec_budget_cache.py','brec_validation.py')
    if manifest['sources']!={name:spec['sources'][name] for name in required}:
        raise ValueError('Job source identity mismatch')
    wanted=job['configuration']
    if expected_keys(config)!=expected_keys(wanted):
        raise ValueError('Job condition set differs from plan')
    for key in ('relabels','permutation_checks','compare_tolerance','joint_normalise',
                'chunk_size','seed','budget_ladder'):
        if config.get(key)!=wanted[key]:
            raise ValueError(f'Job protocol differs from plan: {key}')
    if config.get('evaluation_mode','audit')!=wanted.get('evaluation_mode','audit'):
        raise ValueError('Job evaluation mode differs from plan')
    if config.get('max_budget_slices',False)!=wanted.get('max_budget_slices',False):
        raise ValueError('Job slice mode differs from plan')
    if config.get('joint_scale_audit',False)!=wanted.get('joint_scale_audit',False):
        raise ValueError('Job scale audit differs from plan')
    return result


def run(args):
    root=args.root.resolve()
    budget=args.memory_gib*GIB
    cpus=ordered_cpus() if args.pin_cpus else []
    if args.pin_cpus and (not cpus or not shutil.which('taskset')):
        raise ValueError('--pin-cpus requires Linux taskset and sched_getaffinity')
    scratch=((args.scratch_root / ('brec-'+hashlib.sha256(str(root).encode()).hexdigest()[:16]))
             if args.scratch_root else root/'scratch').resolve()
    scratch.mkdir(parents=True,exist_ok=True)
    lock=(root/'scheduler.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec=json.loads((root/'plan.json').read_text())
    if digest(spec['dataset']) != spec['dataset_sha256']:
        raise ValueError('Dataset changed')
    for name,sha in spec['sources'].items():
        if digest(root/'frozen'/name) != sha:
            raise ValueError(f'Frozen source changed: {name}')
        if digest(source_path(name)) != sha:
            raise ValueError('Use this plan\'s frozen/brec_schedule.py; live sources changed')
    statepath=root/'state.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else {}
    by_id={job['id']:job for job in spec['jobs']}
    for ident,entry in state.items():
        if ident not in by_id:
            raise ValueError('State contains an unplanned job')
        if entry['status']=='running':
            try:
                os.kill(entry['pid'],0)
            except ProcessLookupError:
                entry['status']='interrupted'
            else:
                try:
                    same=(not entry.get('process_created') or
                          psutil.Process(entry['pid']).create_time()==entry['process_created'])
                except psutil.NoSuchProcess:
                    same=False
                if same:
                    raise RuntimeError('A tracked process still exists; inspect before resuming')
                entry['status']='interrupted'
        if entry['status']=='complete':
            check_job(by_id[ident],entry['output'],spec)
        elif entry.get('output') and (Path(entry['output'])/'checkpoint.json').exists():
            validate_checkpoint(entry['output'])
    calpath=root/'calibration.json'
    calibration=json.loads(calpath.read_text()) if calpath.exists() else {}
    pending=[j for j in spec['jobs'] if state.get(j['id'],{}).get('status')!='complete']
    pending.sort(key=lambda j:j['work'],reverse=True)
    waiting={j['id']:time.monotonic() for j in pending}
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
        env[key]='1'
    running={}
    admitted=set()
    previous_signal=signal.getsignal(signal.SIGTERM)
    def stop(signum, frame):
        raise KeyboardInterrupt('Scheduler interrupted')
    signal.signal(signal.SIGTERM,stop)
    last_watch=0.
    print(f'Worker budget: {args.memory_gib:g} GiB; system reserve: {args.reserve_gib:g} GiB; '
          f'CPU slots: {min(args.slots,len(cpus)) if cpus else args.slots}',flush=True)
    try:
        while pending or running:
            for pid,active in list(running.items()):
                proc=active['proc']
                if proc.poll() is None:
                    continue
                active['handle'].close()
                shutil.rmtree(active['scratch'],ignore_errors=True)
                job=active['job']
                entry=state[job['id']]
                elapsed=time.monotonic()-active['started']
                out=Path(entry['output'])
                try:
                    if proc.returncode==0:
                        result=check_job(job,out,spec)
                        status='complete'
                    elif proc.returncode==75:
                        checkpoint=validate_checkpoint(out)
                        result=check_job(job,out,spec,partial=True)
                        if checkpoint['identity']!=protocol_identity(result['manifest']):
                            raise ValueError('Paused checkpoint identity mismatch')
                        if checkpoint['next_stage'] is None:
                            raise ValueError('Worker paused without a next stage')
                        status='paused'
                    else:
                        raise ValueError(f'Worker exited with code {proc.returncode}')
                    new_rows=result['rows'][active['previous_records']:]
                    rss=max((r.get('peak_rss_bytes') or 0 for r in new_rows),default=0)
                    entry['peak_rss_bytes']=max(entry.get('peak_rss_bytes',0),rss,active.get('rss',0))
                    if rss:
                        calibration.setdefault(active['request']['key'],[]).append(
                            rss/active['request']['raw'])
                        save(calpath,calibration)
                    entry.pop('error',None)
                except (ValueError,OSError,KeyError,TypeError) as error:
                    status='failed'
                    entry['error']=str(error)
                    if active.get('memory_killed'):
                        entry['error']='Stopped by memory watchdog; retry with larger reservation'
                        entry['retry_memory_bytes']=max(active['reservation']*2,active.get('rss',0)*1.5)
                entry.update(status=status,returncode=proc.returncode,
                             seconds=entry.get('seconds',0)+elapsed)
                entry.setdefault('stages',[]).append(dict(
                    stage=active['request']['stage'],seconds=elapsed,status=status,
                    reserved_bytes=active['reservation']))
                del running[pid]
                print(job['id'],status,f'{elapsed:.1f}s',flush=True)
                if status=='paused':
                    pending.append(job)
                    pending.sort(key=lambda j:j['work'],reverse=True)
                    waiting[job['id']]=time.monotonic()
                save(statepath,state)
                save(root/'completed_runs.json',[e['output'] for e in state.values()
                                                  if e['status']=='complete'])
            now=time.monotonic()
            if now-last_watch>=args.poll_seconds:
                last_watch=now
                for active in running.values():
                    try:
                        active['rss']=psutil.Process(active['proc'].pid).memory_info().rss
                    except psutil.NoSuchProcess:
                        active['rss']=0
                    active['reservation']=max(active['reservation'],active['rss']*1.15)
                total_rss=sum(a.get('rss',0) for a in running.values())
                available=psutil.virtual_memory().available
                if running and (total_rss>budget or available<args.reserve_gib*GIB):
                    victim=max(running.values(),key=lambda a:a.get('rss',0))
                    victim['memory_killed']=True
                    victim['proc'].kill()
                    print('Memory watchdog stopped',victim['job']['id'],flush=True)
                    time.sleep(.1)
                    continue
            # max-jobs caps admitted job chains, not the number of stage processes.
            candidates=[j for j in pending if not args.max_jobs or
                        len(admitted)<args.max_jobs or j['id'] in admitted]
            if not candidates:
                if not running:
                    break
                time.sleep(.2)
                continue
            reserved=sum(v['reservation'] for v in running.values())
            available=available_memory()
            room=args.memory_gib*GIB-reserved
            if available is not None:
                outstanding=sum(max(0,a['reservation']-a.get('rss',0)) for a in running.values())
                room=min(room,available-args.reserve_gib*GIB-outstanding)
            disk_room=shutil.disk_usage(scratch).free-args.disk_reserve_gib*GIB-sum(
                a['request'].get('disk',0) for a in running.values())
            requests={j['id']:stage_request(j,state.get(j['id'],{}),calibration) for j in candidates}
            head=candidates[0]
            head_request=requests[head['id']]['bytes']*args.memory_factor
            # An intrinsically oversized request must not starve feasible work.
            can_ever_fit=head_request<=args.memory_gib*GIB
            draining=can_ever_fit and needs_drain(
                head_request<=room,time.monotonic()-waiting[head['id']],
                len(running),args.drain_seconds)
            choice=next((j for j in candidates if requests[j['id']]['bytes']*args.memory_factor<=room
                         and requests[j['id']].get('disk',0)<=disk_room),None)
            if choice and len(running)<min(args.slots,len(cpus) if cpus else args.slots) and not draining:
                pending.remove(choice)
                ident=choice['id']
                previous=state.get(ident,{})
                oldout=Path(previous['output']) if previous.get('output') else None
                resume=oldout is not None and (oldout/'checkpoint.json').exists()
                attempt=previous.get('attempt',0) if resume else previous.get('attempt',0)+1
                base=root/'jobs'/ident
                base.mkdir(parents=True,exist_ok=True)
                out=oldout if resume else base/f'attempt{attempt:03d}'
                launch=previous.get('launch',0)+1
                handle=(base/f'attempt{attempt:03d}-launch{launch:04d}.log').open('x')
                command=[sys.executable,'-B',str(root/'frozen/brec_experiments.py'),
                         *choice['command'],'--out-dir',str(out)]
                prior=0
                if resume:
                    checkpoint=validate_checkpoint(out)
                    command.append('--resume-run')
                    with (out/'records.jsonl').open('rb') as records:
                        prior=len(records.read(checkpoint['records_bytes']).splitlines())
                request=requests[ident]
                reservation=request['bytes']*args.memory_factor
                task_scratch=scratch/ident
                shutil.rmtree(task_scratch,ignore_errors=True)
                task_scratch.mkdir()
                worker_env=dict(env,TMPDIR=str(task_scratch),PYTHONUNBUFFERED='1')
                cpu=None
                if cpus:
                    used={a['cpu'] for a in running.values()}
                    cpu=next(c for c in cpus if c not in used)
                    command=[shutil.which('taskset'),'-c',str(cpu),*command]
                proc=subprocess.Popen(command,env=worker_env,stdout=handle,stderr=subprocess.STDOUT)
                entry=dict(previous,status='running',pid=proc.pid,attempt=attempt,
                           launch=launch,output=str(out),cpu=cpu,
                           process_created=psutil.Process(proc.pid).create_time())
                state[ident]=entry
                running[proc.pid]=dict(proc=proc,job=choice,handle=handle,
                                      started=time.monotonic(),reservation=reservation,
                                      request=request,previous_records=prior,
                                      scratch=task_scratch,cpu=cpu,rss=0)
                admitted.add(ident)
                save(statepath,state)
            elif not running and candidates:
                smallest=min(candidates,key=lambda j:requests[j['id']]['bytes'])
                req=requests[smallest['id']]
                print('Pending jobs exceed available RAM or scratch disk. Completed jobs are retained.',
                      f"Smallest RAM request: {req['bytes']*args.memory_factor/GIB:.2f} GiB; "
                      f"its scratch request: {req.get('disk',0)/GIB:.2f} GiB; "
                      f"current rooms: {room/GIB:.2f} GiB RAM, {disk_room/GIB:.2f} GiB disk.",flush=True)
                break
            else:
                time.sleep(.2)
    finally:
        signal.signal(signal.SIGTERM,previous_signal)
        for active in running.values():
            proc=active['proc']
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            active['handle'].close()
            shutil.rmtree(active['scratch'],ignore_errors=True)
            state[active['job']['id']]['status']='interrupted'
        save(statepath,state)
        completed=[]
        for ident,entry in state.items():
            if entry['status']=='complete':
                check_job(by_id[ident],entry['output'],spec)
                completed.append(entry['output'])
        save(root/'completed_runs.json',completed)
    complete=sum(v['status']=='complete' for v in state.values())
    print(f'{complete}/{len(spec["jobs"])} jobs complete; rerun to resume unfinished stages.')
    if complete<len(spec['jobs']) and not args.max_jobs:
        raise SystemExit(2)


def status(args):
    root=args.root.resolve()
    spec=json.loads((root/'plan.json').read_text())
    state=json.loads((root/'state.json').read_text()) if (root/'state.json').exists() else {}
    counts={}
    for job in spec['jobs']:
        label=state.get(job['id'],{}).get('status','pending')
        counts[label]=counts.get(label,0)+1
    print(json.dumps(dict(total_jobs=len(spec['jobs']),counts=counts,
                          failed={key:value.get('error') for key,value in state.items()
                                  if value['status']=='failed'},
                          completed_runs=str(root/'completed_runs.json')),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('plan')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--max-budget-slices',action='store_true')
    p.add_argument('--joint-scale-audit',action='store_true')
    p.add_argument('--brec-file',type=Path,required=True)
    p.add_argument('--trust-pickle',action='store_true')
    p.add_argument('--q-values',nargs='+',type=int,default=[8,12,16])
    p.add_argument('--p-values',nargs='+',type=int,default=[2,4])
    p.add_argument('--orders',nargs='+',type=int,choices=[2,3],default=[2,3])
    p.add_argument('--bases',nargs='+',choices=['monomial','chebyshev'],default=['monomial','chebyshev'])
    p.add_argument('--variants',nargs='+',choices=['polynomial','endpoint','joint'],default=['polynomial','endpoint','joint'])
    p.add_argument('--readouts',nargs='+',choices=['mean','multiset'],default=['mean','multiset'])
    p.add_argument('--decimals',nargs='+',type=int,default=[8])
    p.add_argument('--compare-tolerance',type=float,default=1e-8)
    p.add_argument('--joint-normalise',choices=['none','n'],default='none')
    p.add_argument('--evaluation-mode',choices=['direct','audit'],default='direct')
    p.add_argument('--relabels',type=int,choices=range(1,33),default=32)
    p.add_argument('--chunk-size',type=int,default=128)
    p=sub.add_parser('run')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--slots',type=int,default=100)
    p.add_argument('--pin-cpus',action='store_true',help='Linux physical-core-first affinity')
    p.add_argument('--scratch-root',type=Path,help='Fast local disk for temporary signatures')
    p.add_argument('--disk-reserve-gib',type=float,default=8)
    p.add_argument('--poll-seconds',type=float,default=1)
    p.add_argument('--memory-gib',type=float,default=56,help='Combined worker RAM budget')
    p.add_argument('--reserve-gib',type=float,default=8)
    p.add_argument('--memory-factor',type=float,default=1.3)
    p.add_argument('--drain-seconds',type=float,default=600)
    p.add_argument('--max-jobs',type=int,default=0,help='Limit new jobs for calibration; completed jobs are reusable')
    p=sub.add_parser('status')
    p.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='run' and (args.slots<1 or args.memory_gib<=0 or args.reserve_gib<0
                              or args.memory_factor<=0 or args.max_jobs<0 or args.drain_seconds<0
                              or args.disk_reserve_gib<0 or args.poll_seconds<=0
                              or any(not math.isfinite(x) for x in
                                     (args.memory_gib,args.reserve_gib,args.memory_factor,args.drain_seconds,args.disk_reserve_gib,args.poll_seconds))):
        parser.error('Invalid resource budget')
    if args.action=='plan':
        if args.joint_scale_audit and (not args.max_budget_slices or 4 not in args.q_values or 'mean' not in args.readouts):
            parser.error('--joint-scale-audit requires slice mode, Q=4 and mean')
        if min(args.q_values+args.p_values)<1 or args.chunk_size<1:
            parser.error('Budgets must be positive')
        if not math.isfinite(args.compare_tolerance) or args.compare_tolerance<0:
            parser.error('Comparison tolerance must be finite and nonnegative')
        for key in ('q_values','p_values','orders','bases','variants','readouts','decimals'):
            if len(getattr(args,key))!=len(set(getattr(args,key))):
                parser.error(f'Duplicate {key}')
    {'plan':plan,'run':run,'status':status}[args.action](args)


if __name__=='__main__':
    main()
