"""Aggregate completed BREC encoding runs, preserving skipped/error records."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
try:
    from .theory import POLICY, group_for_pair, impossible
    from .validation import validate_run
except ImportError:  # standalone module inside a frozen BREC snapshot
    from brec_theory import POLICY, group_for_pair, impossible
    from brec_validation import validate_run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--runs",type=Path,nargs="+")
    inputs.add_argument("--runs-file",type=Path,help="JSON list of run directories")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.runs_file:
        args.runs=[Path(x) for x in json.loads(args.runs_file.read_text())]
        if not args.runs:
            p.error("Run list is empty")
    if args.out.exists():
        p.error("Output already exists")
    rows={}
    theoretical={}
    adaptive={}
    excluded=[]
    identity=None
    expected=set()
    smoke=False
    for run in args.runs:
        # Shared schema, condition-key, witness, and checksum checks. Reports
        # may expose incomplete/error runs, but cannot turn them into successes.
        validate_run(run,allow_partial=True,allow_errors=True)
        manifest=json.loads((run/"manifest.json").read_text())
        config=manifest["configuration"]
        if 'sum' in config.get('readouts', []):
            p.error('Legacy sum runs must not be merged into the mean protocol')
        smoke |= config.get("smoke",False)
        if config.get("group_only"):
            p.error("Group-only runs have no encoding measurements")
        for basis in config["bases"]:
            for q in config["q_values"]:
                for order in config["orders"]:
                    for variant in config["variants"]:
                        if order==2 and variant=="joint":
                            continue
                        for d in config["decimals"]:
                            for readout in config.get("readouts", ["multiset"]):
                                for power in (config.get("p_values",[1])
                                              if readout in ("sum","mean") else [1]):
                                    expected.add((basis,q,order,variant,d,readout,power))
        current=(manifest["dataset_sha256"],manifest["sources"],
                 manifest["configuration"]["relabels"],
                 manifest["configuration"]["permutation_checks"],
                 manifest["configuration"]["seed"],
                 config.get('compare_tolerance', 0.), config.get('joint_normalise', 'none'),
                 config.get('chunk_size'), manifest.get('numpy'),
                 manifest.get('theory_policy'), manifest.get('groups_sha256'),
                 config.get('budget_ladder',False),manifest.get('python'),manifest.get('networkx'),
                 config.get('max_budget_slices',False), config.get('joint_scale_audit',False), manifest.get('numerical_protocol'),
                 (sorted(config['q_values']),sorted(config['p_values']))
                 if config.get('budget_ladder') or config.get('max_budget_slices') else None)
        if identity is not None and current!=identity:
            p.error("Dataset, source, or reliability/permutation protocol mismatch")
        identity=current
        for line in (run/"records.jsonl").read_text().splitlines():
            row=json.loads(line)
            if row['status']=='adaptive_skip':
                key=tuple(row[k] for k in ('pair_id','basis','q','order','variant','decimals','readout','p'))
                if not config.get('budget_ladder') or key in adaptive or key in rows or key in theoretical:
                    p.error('Invalid or duplicate adaptive skip')
                adaptive[key]=row
                continue
            if row['status'] == 'theoretical_skip':
                key=tuple(row[k] for k in ('pair_id','basis','q','order','variant','decimals','readout','p'))
                if (row.get('theory_policy') != POLICY or manifest.get('theory_policy') != POLICY
                        or row['fwl_group'] != group_for_pair(row['pair_id'])
                        or not impossible(row['fwl_group'], row['order'])):
                    p.error('Invalid theoretical exclusion record')
                if key in rows or key in theoretical or key in adaptive:
                    p.error(f'Duplicate condition {key}')
                theoretical[key]=row
                continue
            if row["status"]!="ok":
                excluded.append(row)
                continue
            row.setdefault("readout", "multiset")
            row.setdefault("p",1)
            key=tuple(row[k] for k in ("pair_id","basis","q","order","variant","decimals","readout","p"))
            if key in rows or key in theoretical or key in adaptive:
                p.error(f"Duplicate condition {key}; do not merge overlapping runs")
            if row.get('joint_scale_audit'):
                row['joint_scale_audit']['artifact']['path']=str((run/row['joint_scale_audit']['artifact']['path']).resolve())
            rows[key]=row
    for key,row in adaptive.items():
        pid,basis,q,order,variant,d,readout,power=key
        source=row['stopped_after']
        for precision in source['decimals']:
            ref=rows.get((pid,basis,source['q'],order,variant,precision,readout,
                          source['p'] if readout=='mean' else 1))
            if ref is None or not ref['separated'] or any(ref[f] for f in
                    ('reliability_failed','relabel_failed','permutation_failed','upper_bound_flag')):
                p.error('Adaptive skip lacks an audited separation witness')
    tables=defaultdict(lambda:dict(completed_ids=[],raw_separated_ids=[],
                                  unflagged_separated_ids=[],flagged_ids=[]))
    comparisons=defaultdict(lambda:dict(compared_ids=[],gained_ids=[],lost_ids=[],pending_ids=[]))
    basis_disagreements=[]
    readout_comparisons=defaultdict(lambda:dict(compared_ids=[],multiset_only_ids=[],mean_only_ids=[],both_ids=[],pending_ids=[]))
    mean_coverage=defaultdict(lambda:dict(reference_ids=[],covered_ids=[],pending_ids=[]))
    rounding=defaultdict(dict)
    for key,row in rows.items():
        pid,basis,q,order,variant,d,readout,power=key
        table=tables[json.dumps([basis,q,order,variant,d,readout,power,row["fwl_group"]])]
        table["completed_ids"].append(pid)
        bad=any(row[k] for k in ("reliability_failed","relabel_failed",
                                "permutation_failed","upper_bound_flag"))
        if row["separated"]:
            table["raw_separated_ids"].append(pid)
            if not bad:
                table["unflagged_separated_ids"].append(pid)
        if bad:
            table["flagged_ids"].append(pid)
        rounding[(pid,basis,q,order,variant,readout,power)][d]=row["separated"]
        if basis=="chebyshev":
            other=rows.get((pid,"monomial",q,order,variant,d,readout,power))
            if other and other["separated"]!=row["separated"]:
                basis_disagreements.append(list(key))
        if readout == "mean":
            other=rows.get((pid,basis,q,order,variant,d,"multiset",1))
            if other:
                coverage=mean_coverage[json.dumps([basis,q,order,variant,d,power,row["fwl_group"]])]
                audit_flags=("reliability_failed","relabel_failed","permutation_failed","upper_bound_flag")
                if any(r[f] for r in (row,other) for f in audit_flags):
                    coverage["pending_ids"].append(pid)
                elif other["separated"]:
                    coverage["reference_ids"].append(pid)
                    if row["separated"]:
                        coverage["covered_ids"].append(pid)
                comparison=readout_comparisons[json.dumps([basis,q,order,variant,d,power,row["fwl_group"]])]
                if any(r[f] for r in (row,other) for f in audit_flags):
                    comparison["pending_ids"].append(pid)
                else:
                    comparison["compared_ids"].append(pid)
                    if row["separated"] and other["separated"]:
                        comparison["both_ids"].append(pid)
                    elif row["separated"]:
                        comparison["mean_only_ids"].append(pid)
                    elif other["separated"]:
                        comparison["multiset_only_ids"].append(pid)
        parent={"endpoint":"polynomial","joint":"endpoint"}.get(variant)
        if parent:
            other=rows.get((pid,basis,q,order,parent,d,readout,power))
            if other:
                entry=comparisons[json.dumps([basis,q,order,parent,variant,d,readout,power,row["fwl_group"]])]
                if any(r[f] for r in (row,other) for f in
                       ('reliability_failed','relabel_failed','permutation_failed','upper_bound_flag')):
                    entry['pending_ids'].append(pid)
                    continue
                entry["compared_ids"].append(pid)
                if row["separated"] and not other["separated"]:
                    entry["gained_ids"].append(pid)
                if other["separated"] and not row["separated"]:
                    entry["lost_ids"].append(pid)
    # Fixed population, including conditions with no successful records.
    totals={}
    for condition in sorted(expected):
        measured={key[0]:row for key,row in rows.items() if key[1:]==condition}
        completed=set(measured)
        skipped={key[0] for key in theoretical if key[1:]==condition}
        stopped={key[0] for key in adaptive if key[1:]==condition}
        missing=sorted(set(range(400))-completed-skipped)
        separated=sorted(pid for pid,row in measured.items() if row["separated"])
        flagged=sorted(pid for pid,row in measured.items() if any(
            row[k] for k in ("reliability_failed","relabel_failed",
                             "permutation_failed","upper_bound_flag")))
        eligible = (sum(not impossible(group_for_pair(pid), condition[2]) for pid in range(400))
                    if manifest.get('theory_policy') == POLICY else 400)
        totals[json.dumps(condition)]=dict(
            denominator=400,completed_count=len(completed),completed_ids=sorted(completed),
            theoretical_skip_count=len(skipped), theoretical_skip_ids=sorted(skipped),
            adaptive_skip_count=len(stopped), adaptive_skip_ids=sorted(stopped),
            accounted_count=len(completed | skipped),
            theoretically_eligible_count=eligible,
            confirmed_over_eligible=(len(set(separated)-set(flagged))/eligible
                                     if not missing and not flagged and not smoke else None),
            incomplete_ids=missing,separated_count=len(separated),separated_ids=separated,
            flagged_ids=flagged,unreliable_ids=flagged,
            not_separated_ids=sorted(completed-set(separated)-set(flagged)),
            coverage_complete=not missing,
            reportable_full400=(not missing and not flagged and not smoke),
            unflagged_separated_ids=sorted(set(separated)-set(flagged)),
            confirmed_separation_rate=(len(set(separated)-set(flagged))/400
                                     if not missing and not flagged and not smoke else None),
            # A missing measurement is not an indistinguishable graph pair.
            observed_separated_over_400=len(separated)/400,
            final_raw_separation_rate=(len(separated)/400 if not missing and not smoke else None))
    moment_recovery={}
    for key,row in rows.items():
        pid,basis,q,order,variant,d,readout,power=key
        if readout!="mean":
            continue
        base=rows.get((pid,basis,q,order,variant,d,"mean",1))
        ref=rows.get((pid,basis,q,order,variant,d,"multiset",1))
        if base is None or ref is None:
            continue
        flags=("reliability_failed","relabel_failed","permutation_failed","upper_bound_flag")
        entry=moment_recovery.setdefault(json.dumps([basis,q,order,variant,d,power]),
            dict(eligible_ids=[],recovered_ids=[],pending_ids=[]))
        if any(r[f] for r in (base,ref) for f in flags):
            entry["pending_ids"].append(pid)
        elif ref["separated"] and not base["separated"]:
            if any(row[f] for f in flags):
                entry["pending_ids"].append(pid)
            else:
                entry['eligible_ids'].append(pid)
                if row["separated"]:
                    entry["recovered_ids"].append(pid)
    for entry in moment_recovery.values():
        entry["recovery_rate"]=(len(entry["recovered_ids"])/len(entry["eligible_ids"])
                                if entry["eligible_ids"] else None)
    for entry in mean_coverage.values():
        entry["coverage_rate"]=(len(entry["covered_ids"])/len(entry["reference_ids"])
                                if entry["reference_ids"] else None)
    search={}
    routes={(basis,order,variant,d,readout) for basis,q,order,variant,d,readout,power in expected}
    for route in sorted(routes):
        basis,order,variant,d,readout=route
        conditions=[c for c in expected if (c[0],c[2],c[3],c[4],c[5])==route]
        entry=dict(separated_ids=[],exhausted_ids=[],pending_ids=[],unreliable_ids=[],theoretical_skip_ids=[],first_success={})
        for pid in range(400):
            observed=[rows[(pid,*c)] for c in conditions if (pid,*c) in rows]
            clean=[r for r in observed if not any(r[f] for f in
                    ('reliability_failed','relabel_failed','permutation_failed','upper_bound_flag'))]
            successes=[r for r in clean if r['separated']]
            if successes:
                first=min(successes,key=lambda r:(r['p'],r['q']))
                entry['separated_ids'].append(pid)
                entry['first_success'][str(pid)]={
                    'q':first['q'],'p':first['p'],
                    'meaning':'first successful point along the predefined budget ladder'}
            elif all((pid,*c) in theoretical for c in conditions):
                entry['theoretical_skip_ids'].append(pid)
            elif len(clean)==len(conditions):
                entry['exhausted_ids'].append(pid)
            else:
                entry['pending_ids'].append(pid)
                if len(clean)<len(observed):
                    entry['unreliable_ids'].append(pid)
        entry['coverage_complete']=not entry['pending_ids']
        entry['separated_count']=len(entry['separated_ids'])
        points=sorted({(c[6] if readout=='mean' else None,c[1]) for c in conditions},
                      key=lambda point: ((-1 if point[0] is None else point[0]),point[1]))
        first_counts=defaultdict(int)
        for witness in entry['first_success'].values():
            point=(witness['p'] if readout=='mean' else None,witness['q'])
            first_counts[point]+=1
        cumulative=0
        entry['ladder_marginal_gains']=[]
        for power,q in points:
            gain=first_counts[(power,q)]
            cumulative+=gain
            entry['ladder_marginal_gains'].append(dict(
                p=power,q=q,newly_separated=gain,
                cumulatively_separated=cumulative))
        search[json.dumps(route)]=entry
    scale_tables=defaultdict(lambda:dict(completed_ids=[],confirmed_separated_ids=[],unreliable_ids=[]))
    scale_artifacts=[]
    for row in rows.values():
        audit=row.get('joint_scale_audit')
        if not audit:
            continue
        scale_artifacts.append(dict(pair_id=row['pair_id'],basis=row['basis'],q=row['q'],
                                    p=row['p'],**audit['artifact']))
        for route,result in audit['comparisons'].items():
            key=json.dumps([row['basis'],row['q'],row['p'],row['decimals'],row['fwl_group'],route])
            entry=scale_tables[key]
            entry['completed_ids'].append(row['pair_id'])
            if result['confirmed_separated']:
                entry['confirmed_separated_ids'].append(row['pair_id'])
            if result['numerical_outcome']=='unreliable':
                entry['unreliable_ids'].append(row['pair_id'])
    payload=dict(evaluation_mode=config.get('evaluation_mode','audit'),
                 invariance_checks_performed=config.get('evaluation_mode','audit')=='audit',
                 joint_scale_audit=dict(tables=dict(scale_tables),artifacts=scale_artifacts),budget_search=search,adaptive_skips=list(adaptive.values()),
                 mean_coverage=dict(mean_coverage),moment_recovery=moment_recovery,full400_totals=totals,smoke=smoke,
                 readout_comparisons=dict(readout_comparisons),
                 tables=dict(tables),incremental_comparisons=dict(comparisons),
                 basis_disagreements=basis_disagreements,
                 rounding_disagreements=[dict(condition=list(k),decisions=v)
                                        for k,v in rounding.items() if len(set(v.values()))>1],
                 skipped_or_errors=excluded,
                 protocol="Fixed deterministic encoding evaluation; not GNN RPC",
                 budget_result_semantics=("first successful point along the predefined "
                                          "budget ladder; not a globally minimal budget"),
                 note="Direct mode compares only A/B and performs no invariance checks; null flags are unperformed, not passed. "
                      "Audit mode concerns tested relabelings. Neither is an exact-arithmetic certificate. "
                      "Flagged observations are unreliable, not negative results. "
                      "Overall denominator is always 400 per encoding. Missing/error/resource-limit "
                      "pairs are incomplete, never indistinguishable. No final rate for partial "
                      "or smoke runs. Explicit theoretical exclusions account for coverage but "
                      "are not measured failures. Mean-only separations require numerical review: "
                      "tuple rounding and post-mean rounding are different operations. "
                      "Adaptive skips are not fixed-budget measurements. Budget search reports "
                      "separation at any tested budget, not an accuracy at an unmeasured budget.")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x") as f:
        json.dump(payload,f,indent=2)


if __name__=="__main__":
    main()
