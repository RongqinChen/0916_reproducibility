"""Shared structural validation for execution, resumption, and reporting."""
import hashlib
import json
import os
from pathlib import Path
try:
    from .theory import POLICY, group_for_pair, impossible
except ImportError:  # standalone module inside a frozen BREC snapshot
    from brec_theory import POLICY, group_for_pair, impossible

FIELDS=('pair_id','basis','q','order','variant','decimals','readout','p')
FLAGS=('reliability_failed','relabel_failed','permutation_failed','upper_bound_flag')


def numerical_outcome(row):
    """Finite-sample numerical assessment, not RPC or an exact certificate."""
    if any(row[f] for f in FLAGS):
        return 'unreliable'
    return 'separated' if row['separated'] else 'not_separated'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path=Path(path)
    temp=path.with_suffix('.tmp')
    with temp.open('w') as output:
        json.dump(payload,output,indent=2,allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    temp.replace(path)


def expected_keys(config):
    for field in ('pair_ids','bases','q_values','orders','variants','decimals','readouts','p_values'):
        values=config[field]
        if not isinstance(values,list) or not values or len(values)!=len(set(values)):
            raise ValueError(f'Empty or duplicate configured dimension: {field}')
    return {(pid,b,q,o,v,d,r,p) for pid in config['pair_ids']
            for b in config['bases'] for q in config['q_values'] for o in config['orders']
            for v in config['variants'] if not(o==2 and v=='joint')
            for d in config['decimals'] for r in config['readouts']
            for p in (config['p_values'] if r=='mean' else [1])}


def protocol_identity(manifest):
    config={k:v for k,v in manifest['configuration'].items()
            if k not in ('out_dir','resume_run','stage_limit')}
    return dict(configuration=config,**{k:manifest.get(k) for k in
                ('dataset_sha256','sources','python','numpy','networkx','groups_sha256','theory_policy')})


def validate_run(directory, allow_partial=False, allow_errors=False):
    root=Path(directory)
    manifest=json.loads((root/'manifest.json').read_text())
    if not allow_partial and manifest.get('status')!='complete':
        raise ValueError('Run is not complete')
    data=(root/'records.jsonl').read_bytes()
    if (manifest.get('status')=='complete' and 'brec_validation.py' in manifest.get('sources',{})
            and not manifest.get('records_sha256')):
        raise ValueError('Completed run lacks a record checksum')
    if manifest.get('records_sha256') and hashlib.sha256(data).hexdigest()!=manifest['records_sha256']:
        raise ValueError('Record file checksum mismatch')
    expected=expected_keys(manifest['configuration'])
    rows={}
    diagnostics=[]
    audit_checksums={}
    for line in data.splitlines():
        row=json.loads(line)
        if row.get('status') in ('error','resource_limit'):
            if not allow_errors:
                raise ValueError('Run contains execution errors/resource limits')
            diagnostics.append(row)
            continue
        key=tuple(row[k] for k in FIELDS)
        if key not in expected or key in rows:
            raise ValueError(f'Unexpected or duplicate condition: {key}')
        status=row.get('status')
        if status=='ok':
            direct=manifest['configuration'].get('evaluation_mode','audit')=='direct'
            if type(row.get('separated')) is not bool or type(row.get('upper_bound_flag')) is not bool:
                raise ValueError('Invalid measurement flags')
            if direct:
                if (row.get('evaluation_mode')!='direct' or row.get('invariance_checks_performed') is not False
                        or row.get('signature_graph_count')!=2 or any(row.get(f) is not None for f in FLAGS[:3])):
                    raise ValueError('Direct comparison must declare unperformed invariance checks')
            elif any(type(row.get(f)) is not bool for f in FLAGS):
                raise ValueError('Invalid audit flags')
            if 'numerical_outcome' in row and row['numerical_outcome'] != numerical_outcome(row):
                raise ValueError('Numerical outcome contradicts raw decision/audit flags')
            audit=row.get('joint_scale_audit')
            needs_audit=(manifest['configuration'].get('joint_scale_audit',False) and
                         row['order']==3 and row['variant']=='joint' and row['q']==4 and row['readout']=='mean')
            if needs_audit and not audit:
                raise ValueError('Missing integrated scale audit')
            if audit:
                artifact=audit['artifact']
                path=(root/artifact['path']).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError('Invalid scale audit artifact path')
                if path not in audit_checksums:
                    audit_checksums[path]=sha256(path)
                if audit_checksums[path]!=artifact['sha256']:
                    raise ValueError('Scale audit artifact checksum mismatch')
        elif status=='theoretical_skip':
            if (manifest.get('theory_policy')!=POLICY or row.get('theory_policy')!=POLICY or
                row.get('fwl_group')!=group_for_pair(row['pair_id']) or
                not impossible(group_for_pair(row['pair_id']),row['order']) or row.get('separated') is not None):
                raise ValueError('Invalid theoretical exclusion')
        elif status=='adaptive_skip':
            if not manifest['configuration'].get('budget_ladder') or row.get('separated') is not None:
                raise ValueError('Invalid adaptive exclusion')
        else:
            raise ValueError(f'Unknown record status: {status}')
        rows[key]=row
    for key,row in rows.items():
        if row['status']!='adaptive_skip':
            continue
        pid,b,q,o,v,d,r,p=key
        src=row['stopped_after']
        config=manifest['configuration']
        if set(src['decimals'])!=set(config['decimals']):
            raise ValueError('Adaptive witness must cover every configured precision')
        current_p=p if r=='mean' else min(config['p_values'])
        if (src['p'],src['q']) >= (current_p,q):
            raise ValueError('Adaptive witness is not earlier in the budget ladder')
        for precision in config['decimals']:
            ref=rows.get((pid,b,src['q'],o,v,precision,r,src['p'] if r=='mean' else 1))
            if not ref or ref['status']!='ok' or not ref['separated'] or any(ref[f] for f in FLAGS):
                raise ValueError('Adaptive skip lacks an audited witness')
    missing=expected-set(rows)
    if missing and not allow_partial:
        raise ValueError(f'{len(missing)} expected conditions are missing')
    return dict(manifest=manifest,rows=list(rows.values()),missing=missing,diagnostics=diagnostics)


def validate_checkpoint(directory, identity=None):
    root=Path(directory)
    checkpoint=json.loads((root/'checkpoint.json').read_text())
    if identity is None:
        identity=protocol_identity(json.loads((root/'manifest.json').read_text()))
    if identity is not None and checkpoint['identity']!=identity:
        raise ValueError('Checkpoint protocol/source/dataset/runtime mismatch')
    data=(root/'records.jsonl').read_bytes()
    offset=checkpoint['records_bytes']
    if offset>len(data) or hashlib.sha256(data[:offset]).hexdigest()!=checkpoint['records_sha256']:
        raise ValueError('Committed records are missing or corrupted')
    cache=root/checkpoint['moments_file']
    if cache.parent!=root or sha256(cache)!=checkpoint['moments_sha256']:
        raise ValueError('Checkpoint moment cache mismatch')
    return checkpoint
