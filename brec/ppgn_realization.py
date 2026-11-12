"""Train models, then directly compare A/B input multisets for every pair."""
from .determinism import (prepare_environment, start_brec, assert_torch_determinism,
                          is_determinism_error)
prepare_environment()
import argparse
from collections import Counter
import gc
import hashlib
import json
import random
from pathlib import Path
import time
import numpy as np
import torch
from .ppgn_models import FixedInput, RealizationPPGN, input_channels
from .theory import group_for_pair, impossible
from supplementary.common import (load_brec,pair_graphs,validate_pair_ids,read_pair_ids,
                                  begin_run,finish_run,atomic_json,checksum,graph_at)


def resolve_baselines(records, raw, args):
    """Compute A/B input multisets for all requested pairs, once across seeds."""
    baselines={}
    atomic_json(args.out_dir/'baseline_provenance.json', dict(
        policy='all requested pairs; direct A/B multiset; no external baseline'))
    cache_dir=args.out_dir/'input_baselines'
    cache_dir.mkdir(exist_ok=True)
    candidates={(pid,order) for pid in args.pair_ids for order in args.orders}
    for pid,order in sorted(candidates):
        key=(pid,order)
        path=cache_dir/f'pair-{pid:03d}_order-{order}.json'
        if path.exists():
            cached=json.loads(path.read_text())
            if cached['outcome'] in ('separated','not_separated'):
                baselines[key]=cached
                continue
        tick=time.perf_counter()
        try:
            # The baseline is the direct A/B multiset of the model's own input
            # rows (symmetric-normalized monomial powers).
            encoder=FixedInput(args.q,args.chunk_size,cache_bytes=args.input_cache_gb*1024**3)
            multisets=[]
            for index in (pid*64,pid*64+1):
                adj=np.asarray(graph_at(raw,index),dtype=np.float32)
                rows=encoder.array(adj,order).reshape(-1,input_channels(order,args.q))
                multisets.append(Counter(
                    hashlib.sha256(np.ascontiguousarray(row,dtype='<f4').tobytes()).digest()
                    for row in rows))
            separated=multisets[0]!=multisets[1]
            base=dict(outcome='separated' if separated else 'not_separated',
                      pair_id=pid,order=order,q=args.q,basis='monomial',variant='polynomial',
                      kernel='symmetric-normalized',readout='raw-float32-multiset',
                      channels=input_channels(order,args.q),
                      decimals=None,requested_decimals_metadata=args.decimals,
                      dtype='float32',evaluation_mode='direct',
                      invariance_checks_performed=False,signature_graph_count=2,
                      source=str(path.resolve()),seconds=time.perf_counter()-tick)
        except (RuntimeError,FloatingPointError,MemoryError) as error:
            base=dict(outcome='missing',error=str(error),source=str(path.resolve()))
        finally:
            multisets=[]
            gc.collect()
        atomic_json(path,base)
        baselines[key]=base
        print(f'input baseline pair-{pid:03d} order-{order}: {base["outcome"]}',flush=True)
    for record in records:
        key=(record['pair_id'],record['order'])
        record['input_baseline']=baselines[key]
        target=args.out_dir/f'pair-{key[0]:03d}_order-{key[1]}_seed-{record["seed"]}.json'
        atomic_json(target,record)
    atomic_json(args.out_dir/'summary.json',summarize(records))


def rpc_score(values):
    """BREC convention: mean(D)^T cov(D)^+ mean(D), no extra sample factor.

    Float32 covariance pinv uses an explicit cutoff. A centered-data SVD
    independently recomputes the same retained-subspace statistic.
    """
    assert_torch_determinism(torch)
    x=torch.as_tensor(values,dtype=torch.float32,device='cpu')
    if x.ndim!=2 or len(x)!=64 or not torch.isfinite(x).all():
        raise ValueError('RPC requires 64 finite embeddings')
    d=x[::2]-x[1::2]; mu=d.mean(0)
    centered=d-mu
    cov=centered.T@centered/(len(d)-1)
    eps=torch.finfo(cov.dtype).eps
    cutoff=eps*cov.shape[0]*torch.linalg.eigvalsh(cov).abs().max()
    pinv=torch.linalg.pinv(cov,atol=cutoff,rtol=0.,hermitian=True)
    score=float(mu@pinv@mu)
    _,singular,vh=torch.linalg.svd(centered,full_matrices=False)
    keep=singular.square()/(len(d)-1)>cutoff
    svd=float((len(d)-1)*((vh[keep]@mu)/singular[keep]).square().sum())
    agreement=bool(np.isclose(score,svd,rtol=1e-3,atol=1e-5))
    degenerate=not bool(keep.any()) and float(mu.norm())>1e-7
    return dict(score=score,svd_score=svd,scorers_agree=agreement,
                covariance_rank=int(keep.sum()),mean_difference_norm=float(mu.norm()),
                covariance_cutoff=float(cutoff),singular_values=singular.tolist(),
                degenerate_nonzero_mean=degenerate,
                numerically_valid=bool(np.isfinite(score) and score>=0 and np.isfinite(svd) and agreement and not degenerate))


@torch.no_grad()
def embed(model,graphs,inputs,args):
    assert_torch_determinism(torch)
    model.eval(); results=[]
    for start in range(0,len(graphs),args.physical_batch):
        y=model(inputs.batch(graphs[start:start+args.physical_batch],model.order,args.device,
                             work_dtype(args)))
        if not torch.isfinite(y).all():
            raise FloatingPointError('Nonfinite model output')
        # NumPy serialization and RPC use float32, including a bf16 fallback.
        # This cast is a no-op for the archived all-float32 campaign.
        results.append(y.float().cpu())
    return torch.cat(results).numpy()


def evaluate(main,reliability,threshold,order,pid):
    a,b=rpc_score(main),rpc_score(reliability)
    reliable=b['score']<threshold and a['numerically_valid'] and b['numerically_valid']
    separated=a['score']>threshold and not np.isclose(a['score'],b['score'],atol=1e-6,rtol=1e-5)
    upper=bool(reliable and separated and impossible(group_for_pair(pid),order))
    return dict(main=a,reliability=b,reliability_passed=bool(reliable),
                raw_separated=bool(separated),reference_upper_bound_violation=upper,
                confirmed_separated=bool(reliable and separated and not upper),
                numerical_outcome=('unreliable' if not reliable or upper else
                                   'separated' if separated else 'not_separated'))


def train(model,graphs,inputs,args):
    assert_torch_determinism(torch)
    model.train()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
    loss_fn=torch.nn.CosineEmbeddingLoss(margin=0.)
    accumulation=32//args.physical_batch
    history=[]
    for epoch in range(args.epochs):
        opt.zero_grad(set_to_none=True); total=0.
        for step,start in enumerate(range(0,64,args.physical_batch),1):
            y=model(inputs.batch(graphs[start:start+args.physical_batch],model.order,args.device,
                                 work_dtype(args)))
            loss=loss_fn(y[::2],y[1::2],-torch.ones(len(y)//2,device=args.device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            (loss/accumulation).backward()
            total+=float(loss.detach())*len(y)/2
            if step%accumulation==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step(); opt.zero_grad(set_to_none=True)
        mean=total/32
        history.append(dict(epoch=epoch,loss=mean))
        scheduler.step(mean)
        if mean<args.early_stop_loss:
            break
    return history


def work_dtype(args):
    """Model/input dtype of one attempt: float32 unless bf16 was selected."""
    return torch.bfloat16 if getattr(args,'dtype','float32')=='bfloat16' else torch.float32


def fit_condition(graphs, reliability, inputs, order, seed, pid, n, args, stem):
    """Retry only OOM/preflight failures; each retry resets this seed's model.

    The ladder walks batches 32/16/8/4/2 in the requested dtype; with
    --allow-bf16-fallback and a float32 ladder that never fit, it repeats the
    ladder in bfloat16 (half the activation and input memory). The dtype that
    produced the artifact is recorded in every attempt and in the record.
    """
    attempts=[]
    batches=[b for b in (32,16,8,4,2) if b<=args.physical_batch] if args.auto_batch else [args.physical_batch]
    ladder=[(args.dtype,batch,False) for batch in batches]
    if getattr(args,'allow_bf16_fallback',False) and args.dtype=='float32':
        checkpoint=bool(getattr(args,'checkpoint_fallback',False))
        ladder+=[('bfloat16',batch,checkpoint) for batch in batches]
    answer=None
    for dtype,batch,checkpoint in ladder:
        local=argparse.Namespace(**vars(args)); local.physical_batch=batch; local.dtype=dtype
        elements=batch*n**order*max(args.width,input_channels(order,args.q))
        # Checkpointed bfloat16 activations are roughly half the float32 size, so the
        # fallback pass may admit one more batch rung (batch 4 instead of 2 at n=198).
        limit=args.max_state_elements*(2 if dtype!=args.dtype else 1)
        attempt=dict(physical_batch=batch,effective_batch=32,dtype=dtype,
                     grad_checkpoint=checkpoint,state_element_limit=limit)
        tick=time.perf_counter(); model=None
        if elements>limit:
            attempt.update(status='resource_guard',state_elements=elements,seconds=0.)
            attempts.append(attempt)
            answer=dict(status='resource_guard',dtype=dtype,grad_checkpoint=checkpoint,
                        error='State-element guard exceeded')
            continue
        try:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if args.device.startswith('cuda'):
                # reset_peak_memory_stats must not be the first CUDA call in a fresh
                # worker: torch 2.7 raises 'Invalid device argument' on an
                # uninitialized device. Initialize and select the device first.
                torch.cuda.init(); torch.cuda.set_device(torch.device(args.device))
                torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats(args.device)
                attempt['gpu_name']=torch.cuda.get_device_name(args.device)
            model=RealizationPPGN(order,args.q,args.width,args.layers).to(args.device,
                                                                         dtype=work_dtype(local))
            model.grad_checkpoint=checkpoint
            history=train(model,graphs,inputs,local)
            main_values=embed(model,graphs,inputs,local)
            rel_values=embed(model,reliability,inputs,local)
            result=evaluate(main_values,rel_values,args.threshold,order,pid)
            output=args.out_dir/(stem+'.trained.npz')
            checkpoint=args.out_dir/(stem+'.trained.pt')
            np.savez_compressed(output,main=main_values,reliability=rel_values)
            torch.save(model.state_dict(),checkpoint)
            answer=dict(status='complete',trained=result,history=history,dtype=dtype,
                        grad_checkpoint=checkpoint,
                        parameters=sum(x.numel() for x in model.parameters()),
                        artifact_sha256={f.name:checksum(f) for f in (output,checkpoint)})
            attempt['status']='complete'
        except (RuntimeError,FloatingPointError,MemoryError) as error:
            if is_determinism_error(error):
                raise
            oom=isinstance(error, (torch.cuda.OutOfMemoryError,MemoryError)) or 'out of memory' in str(error).lower()
            attempt.update(status='resource_error' if oom else 'execution_error',error=str(error))
            answer=dict(status=attempt['status'],error=str(error),dtype=dtype,
                        grad_checkpoint=checkpoint)
        finally:
            attempt['seconds']=time.perf_counter()-tick
            if args.device.startswith('cuda') and torch.cuda.is_available():
                attempt['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(args.device)
            del model
            gc.collect()
            if args.device.startswith('cuda'):torch.cuda.empty_cache()
        attempts.append(attempt)
        print(stem, 'batch', batch, dtype, attempt['status'], flush=True)
        if attempt['status']!='resource_error':
            break
    if answer is None:
        answer=dict(status=attempts[-1]['status'],dtype=attempts[-1].get('dtype',args.dtype),
                    error=attempts[-1].get('error','State-element guard exceeded'))
    answer.update(physical_batch=attempts[-1]['physical_batch'],effective_batch=32,
                  dtype=attempts[-1].get('dtype',answer.get('dtype',args.dtype)),
                  dtype_fallback=attempts[-1].get('dtype',args.dtype)!=args.dtype,
                  grad_checkpoint=attempts[-1].get('grad_checkpoint',False),
                  batch_attempts=attempts,seconds=sum(a['seconds'] for a in attempts),
                  peak_cuda_allocated_bytes=max((a.get('peak_cuda_allocated_bytes',0) for a in attempts),default=0))
    return answer


def summarize(records):
    summary=[]
    for order in sorted({r['order'] for r in records}):
        for seed in sorted({r['seed'] for r in records}):
            for group in ('2-FWL','3-FWL-only','3-FWL-indistinguishable'):
                subset=[r for r in records if (r['order'],r['seed'],r['reference_group'])==(order,seed,group)]
                entry=dict(order=order,seed=seed,reference_group=group,
                           requested_ids=[r['pair_id'] for r in subset],input_separated_ids=[],
                           trained_separated_ids=[],gained_ids=[],lost_ids=[],both_ids=[],neither_ids=[],
                           comparable_ids=[],matched_input_separated_ids=[],matched_trained_separated_ids=[],unresolved_ids=[],
                           baseline_not_evaluated_ids=[],gain_unresolved_ids=[])
                for r in subset:
                    pid=r['pair_id']; base=r['input_baseline']['outcome']
                    if base=='separated': entry['input_separated_ids'].append(pid)
                    if r['status']=='complete' and r['trained']['confirmed_separated']:
                        entry['trained_separated_ids'].append(pid)
                    positive=r['status']=='complete' and r['trained']['confirmed_separated']
                    if base=='not_evaluated':entry['baseline_not_evaluated_ids'].append(pid)
                    if positive and base in ('missing','unreliable','not_evaluated'):
                        entry['gain_unresolved_ids'].append(pid)
                    if (r['status']!='complete' or base in ('missing','unreliable','not_evaluated') or
                            r['trained']['numerical_outcome']=='unreliable'):
                        entry['unresolved_ids'].append(pid); continue
                    entry['comparable_ids'].append(pid)
                    a=base=='separated'; b=r['trained']['confirmed_separated']
                    if a:entry['matched_input_separated_ids'].append(pid)
                    if b:entry['matched_trained_separated_ids'].append(pid)
                    entry['both_ids' if a and b else 'lost_ids' if a else 'gained_ids' if b else 'neither_ids'].append(pid)
                entry['gain_comparison_complete']=not entry['gain_unresolved_ids']
                entry['training_complete']=all(r['status']=='complete' for r in subset)
                entry['full_input_coverage']=all(r['input_baseline']['outcome'] in
                    ('separated','not_separated','theoretical_nonseparation') for r in subset)
                entry['input_and_lost_counts_are_partial']=not entry['full_input_coverage']
                entry["counts"]={key[:-4]:len(value) for key,value in entry.items() if key.endswith("_ids")}
                summary.append(entry)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--brec-file',type=Path,required=True)
    p.add_argument('--trust-pickle',action='store_true')
    phase=p.add_mutually_exclusive_group()
    phase.add_argument('--train-only',action='store_true',help='Save neural results and defer the baseline phase')
    phase.add_argument('--baseline-only',action='store_true',help='Compute baselines and join saved neural records; never train')
    p.add_argument('--auto-batch',action='store_true',help='On OOM/guard, retry 32/16/8/4/2 up to requested batch; effective batch stays 32')
    pairs=p.add_mutually_exclusive_group(required=True)
    pairs.add_argument('--all-pairs',action='store_true')
    pairs.add_argument('--pair-ids',nargs='+',type=int)
    pairs.add_argument('--pair-list',type=Path)
    p.add_argument('--orders',nargs='+',type=int,choices=[2,3],default=[2,3])
    p.add_argument('--q',type=int,choices=[2,4,8,16],default=8)
    p.add_argument('--seeds',nargs='+',type=int,default=[100,200,300])
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--layers',type=int,default=5)
    p.add_argument('--physical-batch',type=int,choices=[2,4,8,16,32],default=32,
                   help='Graphs per forward pass; accumulate to an effective batch of 32')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--early-stop-loss',type=float,default=.2)
    p.add_argument('--threshold',type=float,default=72.34)
    p.add_argument('--device',default='cpu')
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32',
                   help='Model/input compute dtype (scorer always scores in float32)')
    p.add_argument('--allow-bf16-fallback',action='store_true',
                   help='If float32 fits no batch, repeat the batch ladder in bfloat16')
    p.add_argument('--checkpoint-fallback',action='store_true',
                   help='Recompute block activations inside the bfloat16 fallback pass')
    p.add_argument('--input-cache-gb',type=float,default=4.,
                   help='Per-process budget for cached fixed-encoding feature arrays')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--max-state-elements',type=int,default=1_200_000_000)
    p.add_argument('--chunk-size',type=int,default=128)
    p.add_argument('--decimals',type=int,default=8)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    start_brec(args.seeds[0], use_torch=True)
    if args.threads != 1:
        p.error('BREC deterministic runs require --threads 1')
    if args.auto_batch or args.allow_bf16_fallback or args.checkpoint_fallback:
        p.error('Freeze physical batch, dtype and checkpoint mode before running; adaptive fallbacks are disabled for BREC reproducibility')
    args.pair_ids=list(range(400)) if args.all_pairs else read_pair_ids(args.pair_list) if args.pair_list else args.pair_ids
    validate_pair_ids(args.pair_ids)
    if min(args.width,args.layers,args.epochs,args.threads,args.chunk_size,args.max_state_elements)<1:
        p.error('Dimensions and limits must be positive')
    if args.input_cache_gb<0:
        p.error('Input cache budget must be nonnegative')
    for field in ('orders','seeds'):
        if len(getattr(args,field))!=len(set(getattr(args,field))):p.error('Duplicate '+field)
    if min(args.seeds)<0 or args.lr<=0 or args.weight_decay<0 or args.early_stop_loss<0 or args.threshold<=0:
        p.error('Invalid seed/optimizer/threshold')
    if not all(np.isfinite(v) for v in (args.lr,args.weight_decay,args.early_stop_loss,args.threshold)):
        p.error('Nonfinite option')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    args.torch_version=str(torch.__version__)
    args.input_protocol='symmetric-float32-monomial-three-plane-lifts'
    args.scorer_protocol='float32-covariance-pinv-with-matched-SVD-cutoff-v1'
    extra=[Path(__file__),*(Path(__file__).parent/name for name in
           ('ppgn_models.py','budget_sweep.py','encodings.py','theory.py','validation.py'))]
    # The execution phase does not change trained artifacts.
    training_args=argparse.Namespace(**{k:v for k,v in vars(args).items()
                                      if k not in ('train_only','baseline_only')})
    identity=begin_run(training_args,'ppgn_realization_v5_deterministic_fixed_batch',extra)
    raw=load_brec(args.brec_file,args.trust_pickle)
    records=[]
    for pid in args.pair_ids:
        if args.baseline_only:
            for order in args.orders:
                for seed in args.seeds:
                    target=args.out_dir/f'pair-{pid:03d}_order-{order}_seed-{seed}.json'
                    if target.exists():
                        record=json.loads(target.read_text())
                        if record['status']=='complete':
                            for file,digest in record['artifact_sha256'].items():
                                if checksum(args.out_dir/file)!=digest:raise ValueError('Damaged neural artifact')
                    else:
                        record=dict(pair_id=pid,order=order,seed=seed,reference_group=group_for_pair(pid),
                                    status='not_trained',input_baseline=dict(outcome='not_evaluated'))
                    records.append(record)
            continue
        graphs,reliability=pair_graphs(raw,pid)
        if len({len(g) for g in graphs+reliability})!=1:
            raise ValueError('This experiment compares equal-size graph pairs')
        n=len(graphs[0]); inputs=FixedInput(args.q,args.chunk_size,cache_bytes=args.input_cache_gb*1024**3)
        for order in args.orders:
            for seed in args.seeds:
                stem=f'pair-{pid:03d}_order-{order}_seed-{seed}'
                target=args.out_dir/(stem+'.json')
                if target.exists():
                    old=json.loads(target.read_text())
                    if old['status']=='complete':
                        for file,digest in old['artifact_sha256'].items():
                            if checksum(args.out_dir/file)!=digest:raise ValueError('Damaged neural artifact')
                        records.append(old); continue
                record=dict(pair_id=pid,order=order,seed=seed,n=n,q=args.q,
                            reference_group=group_for_pair(pid),input_baseline=dict(outcome='not_evaluated',reason='deferred until training completes'))
                record.update(fit_condition(graphs,reliability,inputs,order,seed,pid,n,args,stem))
                atomic_json(target,record); records.append(record)
                atomic_json(args.out_dir/'summary.json',summarize(records))
                print(stem,record['status'],flush=True)
    atomic_json(args.out_dir/'summary.json',summarize(records))
    gc.collect()
    if not args.train_only:
        resolve_baselines(records,raw,args)
    finish_run(args,identity,[r['status'] for r in records])
    gain_pending=sum(len(entry['gain_unresolved_ids']) for entry in summarize(records))
    baseline_pending=len({(r['pair_id'],r['order']) for r in records
                          if r['input_baseline']['outcome'] not in ('separated','not_separated')})
    atomic_json(args.out_dir/'baseline_status.json',dict(
        status='deferred' if args.train_only else 'incomplete' if baseline_pending else 'complete',
        unresolved_pair_order_count=baseline_pending,
        unresolved_model_positive_count=gain_pending))
    if not args.train_only and baseline_pending:
        raise SystemExit(f'{baseline_pending} input comparisons unresolved; training artifacts retained')
    failed=sum(r['status']!='complete' for r in records)
    if failed:
        raise SystemExit(f'{failed} jobs incomplete; inspect per-job JSON and resume after resolving the errors')


if __name__=='__main__':main()
