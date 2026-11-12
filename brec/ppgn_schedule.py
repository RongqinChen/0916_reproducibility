"""Dynamic GPU queue for pair/order jobs, plus a separate CPU multiset queue."""
from .determinism import prepare_environment, start_brec, child_environment
prepare_environment()
import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import torch
from supplementary.common import begin_run, finish_run, atomic_json, checksum, load_brec, graph_at
from .ppgn_realization import summarize
from .theory import group_for_pair, impossible

REPO=Path(__file__).resolve().parents[1]


def records_for(job, args):
    records=[]
    for seed in args.seeds:
        path=Path(job['output'])/f"pair-{job['pair_id']:03d}_order-{job['order']}_seed-{seed}.json"
        if path.exists():
            records.append(json.loads(path.read_text()))
        else:
            records.append(dict(pair_id=job['pair_id'],order=job['order'],seed=seed,
                                reference_group=group_for_pair(job['pair_id']),status='pending',
                                input_baseline=dict(outcome='not_evaluated')))
    return records


def training_done(job,args,verify=False):
    rows=records_for(job,args)
    if any(r['status']!='complete' for r in rows):return False
    if verify:
        for row in rows:
            for file,digest in row['artifact_sha256'].items():
                if checksum(Path(job['output'])/file)!=digest:
                    raise ValueError(f"Damaged artifact: {job['id']}/{file}")
    return True


def baseline_done(job):
    directory=Path(job['output'])
    path=directory/'baseline_status.json'
    base=directory/'input_baselines'/f"pair-{job['pair_id']:03d}_order-{job['order']}.json"
    return (path.exists() and base.exists() and json.loads(path.read_text())['status']=='complete'
            and json.loads(base.read_text())['outcome'] in ('separated','not_separated'))


def guard_blocked(job,args):
    """Deterministic state-element guard rejection: retrying cannot help.

    The guard only depends on n, order, q, width and --max-state-elements, so a
    seed rejected at every batch stays rejected for this campaign identity.
    OOM (resource_error) and execution errors stay retryable.
    """
    return any(row['status']=='resource_guard' for row in records_for(job,args))


def job_outcome(job):
    """Manifest status: never report an unfinished job as failed."""
    if job['gpu_status']=='complete' and job['cpu_status']=='complete':return 'complete'
    if job['gpu_status']=='blocked':return 'blocked'
    if 'failed' in (job['gpu_status'],job['cpu_status']):return 'failed'
    return 'pending'


def command(job,args,phase):
    batch=args.pair_batch if job['order']==2 else args.triple_batch
    cmd=[sys.executable,'-m','brec.ppgn_realization','--brec-file',str(args.brec_file),
         '--pair-ids',str(job['pair_id']),'--orders',str(job['order']),
         '--seeds',*map(str,args.seeds),'--q',str(args.q),'--width',str(args.width),
         '--layers',str(args.layers),'--physical-batch',str(batch),
         '--epochs',str(args.epochs),'--lr',str(args.lr),'--weight-decay',str(args.weight_decay),
         '--early-stop-loss',str(args.early_stop_loss),'--threshold',str(args.threshold),
         '--chunk-size',str(args.chunk_size),'--decimals',str(args.decimals),
         '--max-state-elements',str(args.max_state_elements),'--threads',str(args.threads),
         # Same manifest identity in both phases. baseline-only never accesses CUDA.
         '--device','cuda:0','--out-dir',job['output'],'--resume',
         '--train-only' if phase=='gpu' else '--baseline-only']
    if args.trust_pickle:cmd.append('--trust-pickle')
    return cmd


def predict(job,calibration):
    # Dense pair products scale as n^3; shared-pivot triple products as n^4.
    work=job['n']**(job['order']+1)
    samples=calibration[str(job['order'])]
    other=calibration['3' if job['order']==2 else '2']
    factor=(statistics.median(samples[-30:]) if samples else
            statistics.median(other[-30:])*(1./3. if job['order']==2 else 3.) if other else
            1. if job['order']==2 else 3.)
    return work*factor


def publish(jobs,args,active,calibration,skipped=()):
    records=[r for job in jobs for r in records_for(job,args)]
    atomic_json(args.out_dir/'summary.json',summarize(records))
    atomic_json(args.out_dir/'state.json',dict(
        jobs=jobs,calibration=calibration,
        skipped_theory=list(skipped),
        active=[dict(job=a['job']['id'],phase=a['phase'],gpu=a['gpu'],pid=a['proc'].pid)
                for a in active],
        counts={state:sum(j['gpu_status']==state for j in jobs)
                for state in ('pending','running','complete','failed','blocked')},
        cpu_counts={state:sum(j['cpu_status']==state for j in jobs)
                    for state in ('pending','running','complete','failed')},
        dtypes={state:sum(r.get('status')=='complete' and r.get('dtype','float32')==state
                          for r in records)
                for state in ('float32','bfloat16')},
        dtype_fallbacks=sum(r.get('status')=='complete' and bool(r.get('dtype_fallback'))
                            for r in records)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--brec-file',type=Path,required=True)
    p.add_argument('--trust-pickle',action='store_true')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',default=['0','1'],help='Physical GPU indices or GPU UUIDs')
    p.add_argument('--pair-ids',nargs='+',type=int,default=list(range(400)))
    p.add_argument('--orders',nargs='+',type=int,choices=[2,3],default=[2,3])
    p.add_argument('--seeds',nargs='+',type=int,default=[100,200,300])
    p.add_argument('--q',type=int,choices=[2,4,8,16],default=8)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--layers',type=int,default=5)
    p.add_argument('--physical-batch',type=int,choices=[2,4,8,16,32],default=32)
    p.add_argument('--pair-batch',type=int,choices=[2,4,8,16,32])
    p.add_argument('--triple-batch',type=int,choices=[2,4,8,16,32])
    p.add_argument('--cpu-workers',type=int,default=1,help='Concurrent CPU multiset jobs; independent of GPU slots')
    p.add_argument('--threads',type=int,default=1,help='BREC policy requires one CPU thread per worker')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--early-stop-loss',type=float,default=.2)
    p.add_argument('--threshold',type=float,default=72.34)
    p.add_argument('--chunk-size',type=int,default=128)
    p.add_argument('--decimals',type=int,default=8)
    p.add_argument('--max-state-elements',type=int,default=1_200_000_000)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    start_brec(args.seeds[0], use_torch=True)
    if args.threads != 1:
        p.error('BREC deterministic runs require --threads 1')
    args.brec_file=args.brec_file.resolve(); args.out_dir=args.out_dir.resolve()
    args.pair_batch=args.pair_batch or args.physical_batch
    args.triple_batch=args.triple_batch or args.physical_batch
    for key in ('gpus','pair_ids','orders','seeds'):
        values=getattr(args,key)
        if len(values)!=len(set(values)):p.error('Duplicate '+key)
    if any(i<0 or i>=400 for i in args.pair_ids) or min(args.seeds)<0:p.error('Invalid pair IDs/seeds')
    if min(args.width,args.layers,args.cpu_workers,args.threads,args.epochs,args.chunk_size,args.max_state_elements)<1:
        p.error('Dimensions/limits must be positive')
    if (not all(math.isfinite(x) for x in (args.lr,args.weight_decay,args.early_stop_loss,args.threshold))
            or args.lr<=0 or args.weight_decay<0 or args.early_stop_loss<0 or args.threshold<=0):
        p.error('Invalid training parameters')
    args.torch_version=str(torch.__version__)
    args.gpu_hardware={}
    # Each child gets exactly one physical device; prevent inherited remapping.
    for gpu in args.gpus:
        env=dict(child_environment(),CUDA_VISIBLE_DEVICES=gpu)
        check=subprocess.run([sys.executable,'-c',
            'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1, "GPU unavailable"; print(torch.cuda.get_device_name(0))'],
            env=env,capture_output=True,text=True)
        if check.returncode:p.error(f'Cannot use GPU {gpu}: {check.stderr}')
        args.gpu_hardware[str(gpu)]=check.stdout.strip()
        print('GPU',gpu,check.stdout.strip(),flush=True)
    extra=list((REPO/'brec').glob('*.py'))
    identity=begin_run(args,'ppgn_dynamic_gpu_v1_fp32',extra)
    raw=load_brec(args.brec_file,args.trust_pickle)
    jobs=[]
    skipped=[]
    old_path=args.out_dir/'state.json'
    previous=json.loads(old_path.read_text()) if old_path.exists() else {}
    old_jobs={j['id']:j for j in previous.get('jobs',[])}
    calibration=previous.get('calibration',{'2':[],'3':[]})
    for pid in args.pair_ids:
        n=len(graph_at(raw,pid*64))
        for order in args.orders:
            # Theoretically inseparable (pair, order) combinations carry no gain
            # information; the fixed-encoding campaign skips them too.
            if impossible(group_for_pair(pid),order):
                skipped.append(dict(pair_id=pid,order=order,
                                    reference_group=group_for_pair(pid),
                                    reason='theoretically inseparable'))
                continue
            name=f'pair-{pid:03d}_order-{order}'
            job=dict(id=name,pair_id=pid,order=order,n=n,output=str(args.out_dir/'jobs'/name),
                     assigned_gpu=args.gpus[(2 * pid + order - 2) % len(args.gpus)])
            if training_done(job,args,verify=True):
                job['gpu_status']='complete'
            elif guard_blocked(job,args):
                job['gpu_status']='blocked'
            else:
                job['gpu_status']='pending'
            job['cpu_status']='complete' if baseline_done(job) and job['gpu_status']=='complete' else 'pending'
            job['attempts']=old_jobs.get(name,{}).get('attempts',[])
            jobs.append(job)
    del raw
    active=[]
    def stop(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    def launch(job,phase,gpu=None):
        # Refuse source changes in the middle of a campaign.
        for file,digest in identity['sources'].items():
            if checksum(REPO/file)!=digest:raise ValueError('Source changed during campaign: '+file)
        directory=Path(job['output']);directory.parent.mkdir(parents=True,exist_ok=True)
        log_dir=args.out_dir/'logs';log_dir.mkdir(exist_ok=True)
        log=log_dir/f"{job['id']}.{phase}.{time.time_ns()}.log"
        handle=log.open('w')
        env=dict(child_environment(),CUDA_VISIBLE_DEVICES=gpu if gpu is not None else '',
                 OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),
                 OPENBLAS_NUM_THREADS=str(args.threads),PYTHONUNBUFFERED='1',
                 # Reduce allocator fragmentation on the memory-tight fallback jobs.
                 PYTORCH_CUDA_ALLOC_CONF=os.environ.get('PYTORCH_CUDA_ALLOC_CONF',
                                                        'expandable_segments:True'))
        try:
            proc=subprocess.Popen(command(job,args,phase),cwd=REPO,env=env,
                                  stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        finally:handle.close()
        job[phase+'_status']='running'
        active.append(dict(job=job,phase=phase,gpu=gpu,proc=proc,started=time.monotonic(),log=str(log)))
        print('START',phase,gpu,job['id'],flush=True)
    try:
        publish(jobs,args,active,calibration,skipped)
        while True:
            changed=False
            for worker in list(active):
                code=worker['proc'].poll()
                if code is None:continue
                job=worker['job'];phase=worker['phase'];elapsed=time.monotonic()-worker['started']
                ok=training_done(job,args,verify=True) if phase=='gpu' else baseline_done(job)
                if ok:
                    job[phase+'_status']='complete'
                elif phase=='gpu' and guard_blocked(job,args):
                    job[phase+'_status']='blocked'
                    print('GUARD',job['id'],'rejected at every batch; recorded as blocked, not retried',
                          flush=True)
                else:
                    job[phase+'_status']='failed'
                job['attempts'].append(dict(phase=phase,gpu=worker['gpu'],exit_code=code,
                                           seconds=elapsed,log=worker['log']))
                if phase=='gpu' and ok:
                    calibration[str(job['order'])].append(elapsed/job['n']**(job['order']+1))
                active.remove(worker);changed=True
                print('END',phase,worker['gpu'],job['id'],job[phase+'_status'],flush=True)
            busy={a['gpu'] for a in active if a['phase']=='gpu'}
            for gpu in args.gpus:
                if gpu in busy:continue
                pending=[j for j in jobs if j['gpu_status']=='pending' and j['assigned_gpu']==gpu]
                if not pending:continue
                job=min(pending,key=lambda j:(-j['n']**(j['order']+1),j['id']))
                launch(job,'gpu',gpu);changed=True
            while sum(a['phase']=='cpu' for a in active)<args.cpu_workers:
                pending=[j for j in jobs if j['cpu_status']=='pending' and j['gpu_status'] in ('complete','failed')]
                if not pending:break
                launch(max(pending,key=lambda j:j['n']**j['order']),'cpu');changed=True
            if changed:publish(jobs,args,active,calibration,skipped)
            if not active:break
            time.sleep(.5)
    finally:
        # Stop all children before releasing the campaign lock, so resume cannot race.
        for worker in active:
            if worker['proc'].poll() is None:os.killpg(worker['proc'].pid,signal.SIGTERM)
        for worker in active:
            try:worker['proc'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(worker['proc'].pid,signal.SIGKILL);worker['proc'].wait()
            worker['job'][worker['phase']+'_status']='pending'
        active.clear()
        publish(jobs,args,active,calibration,skipped)
    statuses=[job_outcome(j) for j in jobs]
    finish_run(args,identity,statuses)
    counts={state:statuses.count(state) for state in ('complete','failed','pending','blocked')}
    if counts['failed'] or counts['pending']:
        raise SystemExit(f"{counts['failed']} job(s) failed and {counts['pending']} unfinished. "
                         'Inspect state.json/logs and rerun with --resume.')
    if counts['blocked']:
        blocked=[j['id'] for j in jobs if j['gpu_status']=='blocked']
        print(f"{counts['blocked']} job(s) blocked by the state-element guard (not retried):",flush=True)
        for name in blocked[:20]:print('  blocked',name,flush=True)
        if len(blocked)>20:print(f'  ... and {len(blocked)-20} more',flush=True)
        print('Resolve them with different settings in a new campaign directory '
              '(a raised --max-state-elements changes the run identity).',flush=True)
    records=[r for job in jobs for r in records_for(job,args)]
    fallbacks=sorted({r['pair_id'] for r in records
                      if r.get('status')=='complete' and r.get('dtype')=='bfloat16'})
    if fallbacks:
        print(f'{len(fallbacks)} pair(s) trained in the bfloat16 fallback: {fallbacks[:20]}',flush=True)
    if skipped:
        by_order={order:sum(1 for s in skipped if s['order']==order) for order in args.orders}
        print(f'{len(skipped)} (pair, order) combination(s) skipped as theoretically '
              f'inseparable: {by_order}',flush=True)
    print('Complete:',args.out_dir/'summary.json',flush=True)


if __name__=='__main__':main()
