"""BREC encoding comparison, numerical audits, costs, and optional exact grouping."""
try:
    from .determinism import prepare_environment, start_brec
except ImportError:
    from brec_determinism import prepare_environment, start_brec
prepare_environment()
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import re
import resource
import time
import tracemalloc
import tempfile
import os
import shutil
try:  # package entry point; fallback is used by the scheduler's frozen flat copy
    from .validation import (atomic_json, protocol_identity, validate_checkpoint, validate_run,
                             numerical_outcome)
    from .budget_cache import BudgetCache, scale_audit_decisions, channel_keys
    from .budget_sweep import category, polynomial_bank
    from .theory import POLICY, group_for_pair, impossible
    from .encodings import (channel_count, readout_signatures,
                            readout_signatures_shared, signatures_equal,
                            exact_fwl_pair)
except ImportError:  # frozen scheduler snapshot
    from brec_validation import (atomic_json, protocol_identity, validate_checkpoint, validate_run,
                                 numerical_outcome)
    from brec_budget_cache import BudgetCache, scale_audit_decisions, channel_keys
    from budget_sweep import category, polynomial_bank
    from brec_theory import POLICY, group_for_pair, impossible
    from brec_encodings import (channel_count, readout_signatures,
                                readout_signatures_shared, signatures_equal,
                                exact_fwl_pair)
import numpy as np
import networkx as nx


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--brec-file",type=Path,required=True)
    p.add_argument("--trust-pickle",action="store_true")
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--resume-run",action="store_true",help="Resume the committed stage in this output directory")
    p.add_argument("--stage-limit",type=int,default=0,help="Pause after this many new stages (0: no limit)")
    p.add_argument("--joint-scale-audit",action="store_true",
                   help="Persist Q=4 joint raw/scaled/restored mean moments in the same feature pass")
    p.add_argument("--max-budget-slices", action="store_true",
                   help="Compute Qmax/Pmax once and project all fixed-budget signatures")
    p.add_argument("--budget-ladder",action="store_true",
                   help="P-major ascending Q ladder, reuse moments, stop audited separated routes")
    p.add_argument("--smoke",action="store_true",
                   help="Explicit non-reportable subset/order test; formal order-specific shards are supported")
    p.add_argument("--pair-ids",nargs="+",type=int,default=list(range(400)))
    p.add_argument("--pair-shard",nargs=2,type=int,metavar=("START","END"),
                   help="Formal subset run over START<=pair_id<END")
    p.add_argument("--pair-list",type=Path,
                   help="Formal subset run over explicit pair IDs (file: whitespace/comma separated)")
    p.add_argument("--p-values",nargs="+",type=int,default=[2,4])
    p.add_argument("--q-values",nargs="+",type=int,default=[8,12,16])
    p.add_argument("--bases",nargs="+",choices=["monomial","chebyshev"],
                   default=["monomial","chebyshev"])
    p.add_argument("--orders",nargs="+",type=int,choices=[2,3],default=[2,3])
    p.add_argument("--variants",nargs="+",choices=["polynomial","endpoint","joint"],
                   default=["polynomial","endpoint","joint"])
    p.add_argument("--decimals",nargs="+",type=int,default=[8])
    p.add_argument("--readouts",nargs="+",choices=["multiset","mean"],
                   default=["mean","multiset"])
    p.add_argument("--compare-tolerance",type=float,default=0.0,
                   help="Absolute tolerance when comparing mean vectors "
                        "(0 keeps exact byte equality; multiset stays exact)")
    p.add_argument("--joint-normalise",choices=["none","n"],default="none",
                   help="Optional joint scaling by n; changes finite-precision resolution")
    p.add_argument("--evaluation-mode",choices=["direct","audit"],default="direct",
                   help="direct: one A/B pair only; audit: optional relabeling diagnostics")
    p.add_argument("--relabels",type=int,default=32,help="Relabelings per side in audit mode only")
    p.add_argument("--permutation-checks",type=int,default=1)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--chunk-size",type=int,default=128)
    p.add_argument("--max-tuples",type=int,default=0,
                   help="Optional guard; 0 disables pre-emptive skipping")
    p.add_argument("--max-joint-work",type=int,default=0,
                   help="Optional n^4*(Q+1)^3 guard; 0 disables pre-emptive skipping")
    p.add_argument("--no-trace-allocations",action="store_true",
                   help="Skip tracemalloc instrumentation of every condition "
                        "(records process peak RSS instead of traced allocations)")
    p.add_argument("--groups",type=Path,help="Verified JSON: dataset_sha256 and groups mapping pair IDs")
    p.add_argument("--compute-groups",action="store_true")
    p.add_argument("--group-only",action="store_true")
    p.add_argument("--fwl-max-work",type=int,default=2000000)
    p.add_argument("--fwl-max-rounds",type=int,default=100)
    args = p.parse_args()
    determinism = start_brec(args.seed)
    if args.joint_scale_audit and (not args.max_budget_slices or 4 not in args.q_values or 'mean' not in args.readouts):
        p.error("--joint-scale-audit requires --max-budget-slices, Q=4 and mean readout")
    if args.max_budget_slices and (args.budget_ladder or args.resume_run or args.stage_limit):
        p.error("--max-budget-slices is a full-grid mode, incompatible with ladder checkpoints")
    if args.max_budget_slices and (args.max_tuples or args.max_joint_work):
        p.error("Slice mode requires unrestricted budgets; use pair shards to bound jobs")
    if args.out_dir.exists() and not args.resume_run:
        p.error("Use a new output directory (existing results are never overwritten)")
    shard = None
    if args.pair_shard:
        start, end = args.pair_shard
        if not 0 <= start < end <= 400:
            p.error("--pair-shard must satisfy 0 <= START < END <= 400")
        args.pair_ids = list(range(start, end))
        shard = dict(kind="range", start=start, end=end, count=end - start)
    elif args.pair_list:
        tokens = [x for x in re.split(r"[,\s]+", args.pair_list.read_text().strip()) if x]
        ids = [int(x) for x in tokens]
        if not ids or any(not 0 <= i < 400 for i in ids) or len(set(ids)) != len(ids):
            p.error("--pair-list must list distinct pair IDs in 0..399")
        args.pair_ids = sorted(ids)
        shard = dict(kind="list", count=len(ids),
                     sha256=hashlib.sha256(args.pair_list.read_bytes()).hexdigest())
    if args.evaluation_mode == "direct":
        args.relabels = 1
        args.permutation_checks = 0
    if shard and args.smoke:
        p.error("Shard runs are formal subset runs; they cannot be combined with --smoke")
    if (min(args.q_values+args.p_values)<1 or not 1<=args.relabels<=32 or args.chunk_size<1
            or args.permutation_checks<0 or any(not 0<=x<400 for x in args.pair_ids)):
        p.error("Invalid budgets or pair IDs")
    if args.group_only and not args.compute_groups:
        p.error("--group-only requires --compute-groups")
    if args.max_tuples < 0 or args.max_joint_work < 0:
        p.error("Resource limits must be nonnegative")
    if len(set(args.pair_ids)) != len(args.pair_ids):
        p.error("Duplicate pair IDs")
    if not args.smoke and not args.group_only:
        if shard is None and set(args.pair_ids) != set(range(400)):
            p.error("Formal experiments require all 400 graph pairs, a --pair-shard, or a "
                    "--pair-list; use --smoke for partial checks")
    args.pair_ids = sorted(args.pair_ids)
    if args.stage_limit<0:
        p.error("stage-limit must be nonnegative")
    if args.stage_limit or args.resume_run:
        if not args.budget_ladder or len(args.pair_ids)!=1 or len(args.bases)!=1 or len(args.orders)!=1:
            p.error("Stage checkpoints require one pair, basis, order and --budget-ladder")
    for name in ("p_values","q_values","bases","orders","variants","decimals","readouts"):
        if len(getattr(args,name))!=len(set(getattr(args,name))):
            p.error("Duplicate budget/condition value")
    raw = np.load(args.brec_file,allow_pickle=args.trust_pickle)
    if raw.ndim != 1 or len(raw)!=51200:
        p.error("Expected official 51200 graph6 entries")
    data_hash = checksum(args.brec_file)
    groups = {}
    if args.groups:
        supplied = json.loads(args.groups.read_text())
        if supplied["dataset_sha256"] != data_hash:
            p.error("Grouping dataset checksum mismatch")
        groups = supplied["groups"]
        if any(g not in ("2-FWL","3-FWL-only","3-FWL-indistinguishable","unknown")
               for g in groups.values()):
            p.error("Invalid group label")
    # Synthetic smoke/group-computation runs must not inherit official pair labels.
    use_theory = not args.smoke and not args.compute_groups
    if use_theory:
        for pid in range(400):
            supplied_group = groups.get(str(pid), 'unknown')
            expected_group = group_for_pair(pid)
            if supplied_group not in ('unknown', expected_group):
                p.error(f'Grouping conflicts with user-supplied table at pair {pid}')
            groups[str(pid)] = expected_group
    if not np.isfinite(args.compare_tolerance) or args.compare_tolerance < 0:
        p.error('Comparison tolerance must be finite and nonnegative')
    def adjacency(index):
        val=raw[index]
        val=val.encode() if isinstance(val,str) else bytes(val)
        return nx.to_numpy_array(nx.from_graph6_bytes(val.strip()),dtype=np.float32)
    args.out_dir.mkdir(parents=True,exist_ok=args.resume_run)
    source_dir = Path(__file__).resolve().parent
    source_map = {
        'brec_experiments.py': 'experiments.py',
        'brec_encodings.py': 'encodings.py',
        'budget_sweep.py': 'budget_sweep.py',
        'brec_theory.py': 'theory.py',
        'brec_budget_cache.py': 'budget_cache.py',
        'brec_validation.py': 'validation.py',
        'brec_determinism.py': 'determinism.py',
    }
    manifest = dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    dataset_sha256=data_hash,
                    sources={name:checksum(source_dir/(name if (source_dir/name).exists()
                                                          else source_map[name]))
                             for name in source_map},
                    python=platform.python_version(),numpy=np.__version__,networkx=nx.__version__,
                    determinism=determinism,
                    groups_sha256=checksum(args.groups) if args.groups else None,
                    status="running",
                    evaluation_scope=("smoke" if args.smoke else
                                      ("full400-shard" if shard else "full400")),
                    pair_shard=shard,
                    target_graph_pairs=400,
                    theory_policy=POLICY if use_theory else None,
                    theory_source='User-supplied stable FWL results; official pair ordering assumed',
                    protocol="Fixed deterministic encoding evaluation on BREC; not GNN RPC",
                    numerical_protocol=dict(version=3, rpc=False, dtype="float32",
                        evaluation_mode=args.evaluation_mode,
                        invariance_checks_performed=args.evaluation_mode == "audit",
                        sample_policy=("First A and B only" if args.evaluation_mode == "direct" else "Supplied main and reliability blocks"),
                        comparison="Rounded signatures; mean absolute tolerance, exact multiset digests",
                        audit_scope=("Not performed; direct finite-precision comparison" if args.evaluation_mode == "direct" else "Reference-based consistency on tested relabelings only; not an exact certificate"),
                        outcome="separated / not_separated / unreliable; theoretical and adaptive skips unmeasured"),
                    merge_note=("Shard runs must tile pair IDs 0..400 without overlap and be merged "
                                "with brec_report.py --runs before any full400 claim."
                                if shard else None))
    checkpoint=None
    if args.resume_run:
        checkpoint=validate_checkpoint(args.out_dir,protocol_identity(manifest))
        record_file=args.out_dir/"records.jsonl"
        data=record_file.read_bytes()
        offset=checkpoint["records_bytes"]
        if len(data)>offset:
            archive=args.out_dir/f"uncommitted-{time.time_ns()}.jsonl"
            archive.write_bytes(data[offset:])
            with record_file.open("r+b") as records:
                records.truncate(offset)
        groups=checkpoint["groups"]
    atomic_json(args.out_dir/"manifest.json",manifest)
    summary=defaultdict(lambda:dict(completed=0,separated=0,reliability_failed=0,
                                  relabel_failed=0,permutation_failed=0,upper_bound_flags=0))
    if checkpoint:
        summary.update(checkpoint["summary"])
    completed_stages=set(tuple(x) for x in checkpoint["completed_stages"]) if checkpoint else set()
    stage_count=0
    with (args.out_dir/"records.jsonl").open("a" if args.resume_run else "x") as output:
        def commit_stage(stage):
            nonlocal stage_count
            if not args.stage_limit and not args.resume_run:
                return
            output.flush()
            os.fsync(output.fileno())
            validate_run(args.out_dir,allow_partial=True)
            completed_stages.add(stage)
            stage_count+=1
            name=f"moments-{len(completed_stages):04d}.json"
            cache.dump(args.out_dir/name)
            remaining=[(power,q,args.orders[0]) for power,q in stages
                       if (power,q,args.orders[0]) not in completed_stages
                       and (power==min(args.p_values) or "mean" in args.readouts)]
            next_stage=None
            if remaining:
                power,next_q,order=remaining[0]
                reads=[r for r in args.readouts if r!="multiset" or power==min(args.p_values)]
                unresolved={v:[r for r in reads if (order,v,r) not in solved]
                            for v in args.variants if not(order==2 and v=="joint")}
                next_stage=dict(p=power,q=next_q,order=order,n=max_nodes,
                                requested={v:rs for v,rs in unresolved.items() if rs})
            committed=dict(identity=protocol_identity(manifest),
                           completed_stages=sorted(completed_stages),
                           solved=[(list(key),value) for key,value in solved.items()],
                           summary=dict(summary),groups=groups,next_stage=next_stage,
                           moments_file=name,moments_sha256=checksum(args.out_dir/name),
                           records_bytes=(args.out_dir/"records.jsonl").stat().st_size,
                           records_sha256=checksum(args.out_dir/"records.jsonl"))
            atomic_json(args.out_dir/"checkpoint.json",committed)
            # Metadata-only adaptive skips need no new worker/reservation.
            if remaining and next_stage['requested'] and args.stage_limit and stage_count>=args.stage_limit:
                manifest["status"]="paused"
                atomic_json(args.out_dir/"manifest.json",manifest)
                raise SystemExit(75)
        for pid in args.pair_ids:
            group = groups.get(str(pid), 'unknown')
            blocked_orders = {order for order in args.orders
                              if use_theory and impossible(group, order)}
            # Emit explicit per-condition coverage records before loading graphs
            # or constructing polynomial banks. Never label these as measured failures.
            for basis in args.bases:
                for q in args.q_values:
                    for order in sorted(blocked_orders):
                        for variant in args.variants:
                            if order == 2 and variant == 'joint':
                                continue
                            for readout in args.readouts:
                                for d in args.decimals:
                                    for power in (args.p_values if readout == 'mean' else [1]):
                                        output.write(json.dumps(dict(
                                            pair_id=pid, category=category(pid), fwl_group=group,
                                            basis=basis, q=q, order=order, variant=variant,
                                            readout=readout, p=power, decimals=d,
                                            status='theoretical_skip', separated=None,
                                            theory_policy=POLICY,
                                            reason=f'Stable {order}-FWL cannot separate this pair'))+'\n')
            output.flush()
            if blocked_orders == set(args.orders):
                print(f'theoretically excluded pair {pid} ({group})', flush=True)
                continue
            main=[adjacency(pid*64+i) for i in range(2)]
            indices = ([pid*64, pid*64+1] if args.evaluation_mode == "direct" else
                       [pid*64+i for i in range(args.relabels*2)] +
                       [(pid+400)*64+i for i in range(args.relabels*2)])
            max_nodes = max(len(adjacency(i)) for i in indices)
            if args.compute_groups and str(pid) not in groups:
                s2=exact_fwl_pair(*main[:2],2,args.fwl_max_rounds,args.fwl_max_work)
                if s2 is True:
                    group="2-FWL"
                elif s2 is False:
                    s3=exact_fwl_pair(*main[:2],3,args.fwl_max_rounds,args.fwl_max_work)
                    group="3-FWL-only" if s3 is True else (
                        "3-FWL-indistinguishable" if s3 is False else "unknown")
                else:
                    group="unknown"
                groups[str(pid)]=group
            group=groups.get(str(pid),"unknown")
            if args.group_only:
                continue
            for basis in args.bases:
                persistent=bool(args.stage_limit or args.resume_run)
                temporary=None if persistent else tempfile.TemporaryDirectory(prefix="brec_banks_")
                cache=BudgetCache(args.out_dir/"bank-cache" if persistent else temporary.name)
                solved={}
                if checkpoint:
                    cache.restore(args.out_dir/checkpoint["moments_file"])
                    solved={tuple(key):value for key,value in checkpoint["solved"]}
                stages=([(power,q) for power in sorted(set(args.p_values))
                         for q in sorted(set(args.q_values))] if args.budget_ladder
                        else [(None,q) for q in args.q_values])
                for stage_power,q in stages:
                    stage_p=[stage_power] if stage_power is not None else args.p_values
                    stage_readouts=[r for r in args.readouts if r!="multiset" or
                                    stage_power is None or stage_power==min(args.p_values)]
                    if not stage_readouts:
                        continue
                    for order in args.orders:
                        stage_key=(stage_power,q,order)
                        if stage_key in completed_stages:
                            continue
                        if order in blocked_orders:
                            continue
                        # One streamed feature pass serves every requested
                        # variant of this (q, order): polynomial is a prefix of
                        # endpoint, which is a prefix of joint.
                        active=[v for v in args.variants if not(order==2 and v=="joint")]
                        if not active:
                            continue
                        n=max_nodes
                        metas={v:dict(pair_id=pid,category=category(pid),fwl_group=group,
                                      basis=basis,q=q,order=order,variant=v,
                                      channels=channel_count(order,q,v)) for v in active}
                        pending=[]
                        for v in active:
                            for r in stage_readouts:
                                source=solved.get((order,v,r))
                                if source is not None:
                                    for d in args.decimals:
                                        for power in (stage_p if r=="mean" else [1]):
                                            output.write(json.dumps(dict(
                                                metas[v],readout=r,p=power,decimals=d,
                                                status="adaptive_skip",separated=None,
                                                stopped_after=source,
                                                reason="Earlier budget separated this route with all audits passed"))+"\n")
                            if any((order,v,r) not in solved for r in stage_readouts):
                                pending.append(v)
                        compute=[v for v in pending if not (
                            (args.max_tuples and n**order>args.max_tuples) or (
                                v=="joint" and args.max_joint_work
                                and n**4*(q+1)**3>args.max_joint_work))]
                        for v in pending:
                            if v not in compute:
                                output.write(json.dumps(dict(metas[v],status="resource_limit"))+"\n")
                        if not compute:
                            output.flush()
                            commit_stage(stage_key)
                            continue
                        if True:
                            traced = not args.no_trace_allocations
                            if traced:
                                tracemalloc.start()
                            tick=time.perf_counter()
                            try:
                                signature_keys=[(r,d,power) for r in stage_readouts for d in args.decimals
                                                for power in (stage_p if r == "mean" else [1])]
                                failures={v:{key:[False,False,False] for key in signature_keys}
                                          for v in compute}
                                refs={}
                                scale_samples=[]
                                scale_labels=[]
                                scale_indices=[]
                                scale_sizes=[]
                                bank_time=0.0
                                def make_signature(a):
                                    nonlocal bank_time
                                    started=time.perf_counter()
                                    if args.max_budget_slices:
                                        return cache.sliced_signatures(
                                            a,q,args.q_values,basis,order,args.decimals,args.chunk_size,
                                            stage_readouts,stage_p,compute,args.joint_normalise,
                                            scale_audit=args.joint_scale_audit)
                                    if args.budget_ladder:
                                        return cache.signatures(
                                            a,q,basis,order,args.decimals,args.chunk_size,
                                            stage_readouts,stage_p,compute,args.joint_normalise,
                                            requested={v:{r for r in stage_readouts
                                                          if (order,v,r) not in solved} for v in compute})
                                    bank=polynomial_bank(a,q,basis)
                                    bank_time+=time.perf_counter()-started
                                    return readout_signatures_shared(
                                        a,bank,q,order,args.decimals,args.chunk_size,
                                        stage_readouts,stage_p,compute,args.joint_normalise)
                                def compare(other,reference,flag):
                                    for v in compute:
                                        for key in signature_keys:
                                            if key not in other[v]:
                                                continue
                                            failures[v][key][flag] |= not signatures_equal(
                                                other[v][key],reference[v][key],key[0],
                                                args.compare_tolerance)
                                for i,index in enumerate(indices):
                                    graph=adjacency(index)
                                    current=make_signature(graph)
                                    if '_scale_moments' in current:
                                        scale_samples.append(current.pop('_scale_moments'))
                                        scale_labels.append(('left' if i%2==0 else 'right') if i<args.relabels*2 else 'reliability')
                                        scale_indices.append(index)
                                        scale_sizes.append(len(graph))
                                    if i<2:
                                        refs[i]=current
                                    elif i<args.relabels*2:
                                        compare(current,refs[i%2],1)
                                    elif i==args.relabels*2:
                                        refs[2]=current
                                    else:
                                        compare(current,refs[2],0)
                                    del current
                                rng=np.random.Generator(np.random.PCG64(args.seed+pid))
                                for _ in range(args.permutation_checks):
                                    perm=rng.permutation(len(main[0]))
                                    current=make_signature(main[0][np.ix_(perm,perm)])
                                    if '_scale_moments' in current:
                                        scale_samples.append(current.pop('_scale_moments'))
                                        scale_labels.append('left')
                                        scale_indices.append(-1)
                                        scale_sizes.append(len(main[0]))
                                    compare(current,refs[0],2)
                                    del current
                                perm_failed={v:{key:failures[v][key][2] for key in signature_keys}
                                             for v in compute}
                                elapsed=time.perf_counter()-tick
                                _,peak=tracemalloc.get_traced_memory() if traced else (0,None)
                                rss=(None if traced else resource.getrusage(
                                        resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=="Darwin" else 1024))
                                scale_artifact=None
                                if scale_samples:
                                    scale_samples=np.stack(scale_samples)
                                    audit_dir=args.out_dir/'scale_moments'
                                    audit_dir.mkdir(exist_ok=True)
                                    audit_path=audit_dir/f'pair{pid:03d}_{basis}_q{q}.npz'
                                    np.savez_compressed(audit_path,moments=scale_samples,
                                                        labels=np.array(scale_labels),indices=np.array(scale_indices),
                                                        sizes=np.array(scale_sizes),
                                                        modes=np.array(['K','K_over_n','restored']),
                                                        channels=np.array([json.dumps(k) for k in channel_keys(q,order,'joint')]),
                                                        powers=np.arange(1,max(stage_p)+1),
                                                        joint_offset=channel_count(3,q,'endpoint'))
                                    scale_artifact=dict(path=str(audit_path.relative_to(args.out_dir)),
                                                        sha256=checksum(audit_path))
                                confirmed={}
                                for variant in compute:
                                    meta=metas[variant]
                                    for readout,d,power in signature_keys:
                                        if (order,variant,readout) in solved:
                                            continue
                                        signature_key=(readout,d,power)
                                        separated=not signatures_equal(
                                            refs[0][variant][signature_key],refs[1][variant][signature_key],
                                            readout,args.compare_tolerance)
                                        rel_failed,label_failed,_=failures[variant][signature_key]
                                        upper=separated and (group=="3-FWL-indistinguishable" or
                                              (order==2 and group=="3-FWL-only"))
                                        row=dict(meta,status="ok",readout=readout,p=power,
                                                 readout_channels=meta["channels"]*power,decimals=d,
                                                 separated=separated,
                                                 evaluation_mode=args.evaluation_mode,
                                                 invariance_checks_performed=args.evaluation_mode == "audit",
                                                 signature_graph_count=len(indices)+args.permutation_checks,
                                                 reliability_failed=rel_failed,relabel_failed=label_failed,
                                                 permutation_failed=perm_failed[variant][signature_key],
                                                 upper_bound_flag=upper,
                                                 measured_seconds=elapsed,
                                                 measured_seconds_scope="variant-family-including-banks",
                                                 python_traced_peak_bytes=peak,peak_rss_bytes=rss,
                                                 polynomial_bank_seconds=bank_time,
                                                 dense_feature_bytes=len(main[0])**order*meta["channels"]*4)
                                        if args.max_budget_slices:
                                            row["budget_execution"]="maximum-budget-slices"
                                            row["shared_q_values"]=args.q_values
                                            row["shared_p_max"]=max(args.p_values)
                                            row["measured_seconds_scope"]="shared-budget-pass-or-cache-read"
                                            row["polynomial_bank_seconds"]=None
                                        if scale_artifact and variant=='joint' and readout=='mean':
                                            row['joint_scale_audit']=dict(
                                                artifact=scale_artifact,
                                                comparisons=scale_audit_decisions(scale_samples,scale_labels,
                                                                                 power,d,args.compare_tolerance))
                                        if args.evaluation_mode == 'direct':
                                            for field in ('reliability_failed','relabel_failed','permutation_failed'):
                                                row[field]=None
                                        row['numerical_outcome']=numerical_outcome(row)
                                        if args.budget_ladder:
                                            row["budget_search"]="P-major ascending Q"
                                            row["polynomial_bank_seconds"]=None
                                            good=(separated and not rel_failed and not label_failed and
                                                  not perm_failed[variant][signature_key] and not upper)
                                            confirmed.setdefault((order,variant,readout),[]).append(good)
                                        output.write(json.dumps(row)+"\n")
                                        key=json.dumps([basis,q,order,variant,d,readout,power,group])
                                        counts=summary[key]
                                        counts["completed"]+=1
                                        for name,value in (("separated",separated),
                                                           ("reliability_failed",rel_failed),
                                                           ("relabel_failed",label_failed),
                                                           ("permutation_failed",
                                                            perm_failed[variant][signature_key]),
                                                           ("upper_bound_flags",upper)):
                                            counts[name]+=int(value)
                                if args.budget_ladder:
                                    for route,checks in confirmed.items():
                                        if checks and all(checks):
                                            solved[route]=dict(q=q,p=stage_power,decimals=args.decimals)
                            except (MemoryError,FloatingPointError) as exc:
                                for v in compute:
                                    output.write(json.dumps(dict(metas[v],status="error",
                                                                 error=str(exc)))+"\n")
                            finally:
                                refs={}
                                if traced:
                                    tracemalloc.stop()
                            output.flush()
                            commit_stage(stage_key)
                if temporary:
                    temporary.cleanup()
            print(f"completed pair {pid} ({group})",flush=True)
    (args.out_dir/"groups.json").write_text(json.dumps(
        dict(dataset_sha256=data_hash,groups=groups),indent=2))
    if args.evaluation_mode == "direct":
        for counts in summary.values():
            counts["invariance_checks_performed"] = False
            for field in ("reliability_failed", "relabel_failed", "permutation_failed"):
                counts[field] = None
    (args.out_dir/"summary.json").write_text(json.dumps(dict(summary),indent=2))
    manifest["status"]="complete"
    manifest["records_sha256"]=checksum(args.out_dir/"records.jsonl")
    atomic_json(args.out_dir/"manifest.json",manifest)
    if args.stage_limit or args.resume_run:
        validate_run(args.out_dir)
        if (args.out_dir/"bank-cache").is_dir():
            shutil.rmtree(args.out_dir/"bank-cache")


if __name__=="__main__":
    main()
