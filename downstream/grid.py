#!/usr/bin/env python3
"""Plan or execute matched Powers/Endpoints downstream encoding comparisons.

For GNNPlus each total width creates Powers-d, Endpoints-d and Mixed-d
(two d/2 branches). Optional Wider Powers uses 2d. GRIT retains its legacy
branch-width convention. Conditions are selected by ``--encodings``. The runner writes a safe command
manifest and a collector manifest accepted by ``downstream.report``.
Training is only started when ``--execute`` is supplied.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def config_value(config_text: str, key: str, default=None):
    match = re.search(rf'(?m)^\s*{re.escape(key)}:\s*([^#\n]+)', config_text)
    return match.group(1).strip().strip('"\'') if match else default


def encoding_overrides(engine, encoding, width, degree_budget, domain,
                       config_text):
    powers = encoding in ('powers', 'mixed', 'wider_powers')
    endpoints = encoding in ('endpoints', 'mixed')
    power_width = 2 * width if encoding == 'wider_powers' else width
    if engine == 'grit':
        return [
            'posenc_RRWP.enable', str(powers),
            'posenc_RRWP.ksteps', str(power_width),
            'posenc_RRWP.add_identity', 'False',
            'posenc_RRWP.channel_layout', 'symmetric_monomial_1_to_k_v1',
            'posenc_Contraction.enable', str(endpoints),
            'posenc_Contraction.qsteps', str(degree_budget),
            'posenc_Contraction.channel_budget', str(width),
            'posenc_Contraction.exclude_constant', 'True',
            'posenc_Contraction.symmetric_channels', 'True',
        ]
    if engine == 'gnnplus':
        node_encoder = config_value(config_text, 'node_encoder_name', '')
        native_node = node_encoder.removesuffix('+RWSE')
        edge_encoder = config_value(config_text, 'edge_encoder_name', '')
        supported = {
            ('TypeDictNode', 'TypeDictEdge'), ('Atom', 'Bond'),
            ('LinearNode', 'LinearEdge'), ('LinearNode', 'DummyEdge'),
            ('VOCNode', 'VOCEdge'), ('COCONode', 'COCOEdge'),
            ('PPANode', 'PPAEdge'),
        }
        if config_value(config_text, 'layer_type') != 'gine':
            raise ValueError('Matched GNNPlus experiments require GINE')
        if (native_node, edge_encoder) not in supported:
            raise ValueError(f'Unsupported native encoder combination: {native_node}/{edge_encoder}')
        if domain not in ('node', 'edge'):
            raise ValueError('Matched GNNPlus supports node or edge injection')
        if encoding == 'mixed' and width % 2:
            raise ValueError('Mixed requires an even total width')
        return [
            'posenc_RWSE.enable', 'False', 'posenc_Endpoint.enable', 'False',
            'dataset.node_encoder_name', native_node,
            'dataset.edge_encoder_name', edge_encoder,
            'posenc_StructuralEdge.enable', 'True',
            'posenc_StructuralEdge.injection', domain,
            'posenc_StructuralEdge.powers_normalization', 'rw',
            'posenc_StructuralEdge.endpoints_normalization', 'sym',
            'posenc_StructuralEdge.route', encoding,
            'posenc_StructuralEdge.width', str(width),
            'posenc_StructuralEdge.qsteps', str(degree_budget),
            'posenc_StructuralEdge.raw_norm_type', 'BatchNorm',
        ]
    raise ValueError(f'Unsupported engine: {engine}')


def safe_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def find_gnnplus_run(job_root: Path, seed: int, split: int) -> Path:
    matches = [p for p in job_root.rglob(f'seed-{seed}_split-{split}') if p.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one GNNPlus run directory, found {len(matches)} under {job_root}')
    return matches[0]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('grit', 'gnnplus'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--metric', required=True)
    parser.add_argument('--direction', choices=('min', 'max'), required=True)
    parser.add_argument('--selection-metric', required=True)
    parser.add_argument('--selection-direction', choices=('min', 'max'), required=True)
    parser.add_argument('--split-id', required=True,
                        help='human-readable identifier for the fixed official split')
    parser.add_argument('--split-index', type=int, default=0,
                        help='GNNPlus dataset split index; GRIT ignores this field')
    parser.add_argument('--domain', choices=('node', 'edge', 'pair'), required=True,
                        help='actual injection domain; GRIT uses full ordered pairs')
    parser.add_argument('--degree-budget', type=int, default=14,
                        help='endpoint strict total-degree budget Q')
    parser.add_argument('--widths', type=int, nargs='+', default=[20],
                        help='GNNPlus total widths d; Mixed uses d/2 channels from each branch')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--encodings', nargs='+',
                        choices=('powers', 'endpoints', 'mixed', 'wider_powers'),
                        default=['powers', 'endpoints', 'mixed'])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--device', default='cuda:0',
                        help='GRIT device; GNNPlus uses CUDA_VISIBLE_DEVICES and auto-selects')
    parser.add_argument('--device-pool', nargs='+', default=None,
                        help='GNNPlus device pool for parallel execution, e.g. cuda:0 cuda:1')
    parser.add_argument('--execute', action='store_true',
                        help='launch training; otherwise write plans only')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = args.config.resolve()
    if (not config.is_file() or args.degree_budget < 2 or not args.widths
            or min(args.widths) < 1 or not args.seeds
            or (args.engine == 'gnnplus' and args.split_index < 0)):
        parser.error('Config, degree budget, widths, seeds, and split index must be valid')
    if (len(set(args.widths)) != len(args.widths)
            or len(set(args.seeds)) != len(args.seeds)
            or len(set(args.encodings)) != len(args.encodings)):
        parser.error('Widths, seeds, and encodings must be distinct')
    if args.engine == 'gnnplus' and args.device != 'cpu' and not re.fullmatch(r'cuda:\d+', args.device):
        parser.error('GNNPlus --device must be cpu or cuda:N')
    if args.engine == 'grit' and args.device != 'cpu' and not re.fullmatch(r'cuda:\d+', args.device):
        parser.error('GRIT --device must be cpu or cuda:N')
    devices = list(args.device_pool) if args.device_pool else [args.device]
    for device in devices:
        if device != 'cpu' and not re.fullmatch(r'cuda:\d+', device):
            parser.error('Each device must be cpu or cuda:N')
    if args.engine == 'grit' and len(devices) != 1:
        parser.error('GRIT supports exactly one device; --device-pool is for GNNPlus')

    cfg_text = config.read_text()
    export_per_graph = config_value(cfg_text, 'task', 'graph') == 'graph'
    if args.engine == 'gnnplus' and args.domain == 'edge':
        layer = config_value(cfg_text, 'layer_type', '').lower()
        if layer == 'gcn':
            parser.error('Endpoint edge injection requires an edge-aware GNNPlus layer')
    if args.engine == 'grit' and args.domain != 'pair':
        parser.error('GRIT exposes these structural channels on the full pair state; use --domain pair')
    if args.engine == 'gnnplus' and args.domain == 'pair':
        parser.error('GNNPlus injection supports node and existing-edge domains, not all pairs')

    repo = ROOT / ('grit_official' if args.engine == 'grit' else 'gnnplus_official')
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    groups = []
    for width in args.widths:
        encodings = list(args.encodings)
        protocol = f'{args.engine}-{config.stem}-{args.domain}-Q{args.degree_budget}-d{width}'
        if args.engine == 'gnnplus':
            protocol += '-rw-sym-equal-total-v3'
        group = {
            'id': f'{args.engine}-{args.dataset}-{args.domain}-{args.split_id}-d{width}',
            'dataset': args.dataset,
            'domain': args.domain,
            'degree_budget': args.degree_budget,
            'backbone': args.engine.upper(),
            'format': 'graphgym_jsonl' if args.engine == 'grit' else 'gnnplus_final_log',
            'split_id': args.split_id,
            'protocol_id': protocol,
            'metric': args.metric,
            'direction': args.direction,
            'selection': {'metric': args.selection_metric,
                          'direction': args.selection_direction},
            'baseline': 'powers',
            'encodings': encodings,
            'expected_seeds': args.seeds,
            'split_index': args.split_index if args.engine == 'gnnplus' else None,
            'runs': [],
        }
        groups.append(group)
        for encoding in encodings:
            for seed in args.seeds:
                job_root = out / encoding / f'd{width}' / f'seed-{seed}'
                overrides = encoding_overrides(
                    args.engine, encoding, width, args.degree_budget,
                    args.domain, cfg_text)
                if args.engine == 'grit':
                    tag = f'{encoding}-d{width}'
                    command = [args.python, str(repo / 'main.py'), '--cfg', str(config),
                               '--repeat', '1', 'wandb.use', 'False',
                               'accelerator', args.device,
                               'dataset.dir', str(args.data_root.resolve()),
                               'out_dir', str(job_root), 'seed', str(seed),
                               'name_tag', tag, 'train.export_per_graph', str(export_per_graph),
                               *overrides]
                    run_dir = job_root / f'{config.stem}-{tag}' / str(seed)
                    log_file = None
                    result_format = 'graphgym_jsonl'
                    env_overrides = {'OMP_NUM_THREADS': '1'}
                else:
                    command = [args.python, str(repo / 'main.py'), '--cfg', str(config),
                               '--repeat', '1', 'seed', str(seed),
                               'dataset.dir', str(args.data_root.resolve()),
                               'out_dir', str(job_root),
                               'dataset.split_index', str(args.split_index),
                               'train.export_per_graph', str(export_per_graph), *overrides]
                    run_dir = None
                    log_file = None
                    result_format = 'gnnplus_final_log'
                    job_device = devices[len(jobs) % len(devices)]
                    visible = '' if job_device == 'cpu' else job_device.split(':', 1)[1]
                    env_overrides = {'CUDA_VISIBLE_DEVICES': visible,
                                     'OMP_NUM_THREADS': '1'}
                jobs.append({
                    'encoding': encoding, 'width': width, 'seed': seed,
                    'collector_group_id': group['id'],
                    'command': command, 'cwd': str(repo),
                    'env_overrides': env_overrides,
                    'run_root': safe_relative(job_root, out),
                    'expected_format': result_format,
                    'expected_run_dir': safe_relative(run_dir, out) if run_dir else None,
                    'status': 'queued',
                })
                group['runs'].append({
                    'encoding': encoding, 'seed': seed, 'status': 'queued',
                    'run_dir': None, 'log_file': None,
                })

    collector_manifest = {'schema_version': 1, 'groups': groups}
    manifest = {
        'schema_version': 2,
        'engine': args.engine,
        'dataset': args.dataset,
        'domain': args.domain,
        'split_id': args.split_id,
        'degree_budget': args.degree_budget,
        'widths': args.widths,
        'config': str(config),
        'config_sha256': sha256(config),
        'data_root': str(args.data_root.resolve()),
        'device': args.device,
        'runner_sha256': sha256(SCRIPT),
        'source_sha256': {
            str(path.relative_to(repo)): sha256(path)
            for path in (
                [repo / 'main.py']
                + ([repo / 'grit' / 'network' / 'grit_model.py',
                    repo / 'grit' / 'loader' / 'master_loader.py',
                    repo / 'grit' / 'transform' / 'posenc_stats.py',
                    repo / 'grit' / 'transform' / 'single_pivot_contraction.py',
                    repo / 'grit' / 'transform' / 'rrwp.py',
                    repo / 'grit' / 'config' / 'posenc_config.py',
                    repo / 'grit' / 'utility_export.py',
                    repo / 'grit' / 'train' / 'custom_train.py']
                   if args.engine == 'grit' else
                   [repo / 'GNNPlus' / 'network' / 'custom_gnn.py',
                    repo / 'GNNPlus' / 'encoder' / 'structural_edge.py',
                    repo / 'GNNPlus' / 'encoder' / 'endpoint_addition.py',
                    repo / 'GNNPlus' / 'encoder' / 'type_dict_encoder.py',
                    repo / 'GNNPlus' / 'encoder' / 'linear_node_encoder.py',
                    repo / 'GNNPlus' / 'encoder' / 'linear_edge_encoder.py',
                    repo / 'GNNPlus' / 'encoder' / 'dummy_edge_encoder.py',
                    repo / 'GNNPlus' / 'encoder' / 'ppa_encoder.py',
                    repo / 'GNNPlus' / 'encoder' / 'voc_superpixels_encoder.py',
                    repo / 'GNNPlus' / 'loader' / 'master_loader.py',
                    repo / 'GNNPlus' / 'transform' / 'posenc_stats.py',
                    repo / 'GNNPlus' / 'transform' / 'single_pivot_contraction.py',
                    repo / 'GNNPlus' / 'config' / 'posenc_config.py',
                    repo / 'GNNPlus' / 'setting_output.py',
                    repo / 'GNNPlus' / 'utility_export.py',
                    repo / 'GNNPlus' / 'train' / 'custom_train.py'])
            ) if path.is_file()
        },
        'jobs': jobs,
    }
    jobs_path = out / 'jobs.json'
    collector_path = out / 'collector_manifest.json'

    def normalized_jobs(value):
        runtime_fields = {'status', 'returncode', 'stdout_log', 'elapsed_seconds'}
        return {
            'base': {key: item for key, item in value.items() if key != 'jobs'},
            'jobs': [{key: item for key, item in job.items()
                      if key not in runtime_fields} for job in value.get('jobs', [])],
        }

    def normalized_collector(value):
        return [{key: item for key, item in group.items() if key != 'runs'}
                for group in value.get('groups', [])]

    if any(out.iterdir()):
        if not args.execute or not jobs_path.is_file() or not collector_path.is_file():
            parser.error('Output root is non-empty; execution may reuse only a matching saved plan')
        previous_manifest = json.loads(jobs_path.read_text())
        previous_collector = json.loads(collector_path.read_text())
        if (normalized_jobs(previous_manifest) != normalized_jobs(manifest)
                or normalized_collector(previous_collector)
                != normalized_collector(collector_manifest)):
            parser.error('Saved plan differs in configuration, data, source, or job list')
        old_conditions = {(group['id'], run['encoding'], run['seed']): run
                          for group in previous_collector.get('groups', [])
                          for run in group.get('runs', [])}
        for group in collector_manifest['groups']:
            for run in group['runs']:
                key = (group['id'], run['encoding'], run['seed'])
                previous = old_conditions.get(key)
                if previous is None or previous.get('status') not in ('queued', 'complete'):
                    parser.error(f'Cannot resume a failed or invalid condition: {key}')
                run.update(previous)
        manifest = previous_manifest
        collector_manifest = previous_collector
        jobs = manifest['jobs']
        groups = collector_manifest['groups']

    def save_manifests():
        jobs_path.write_text(json.dumps(manifest, indent=2) + '\n')
        collector_path.write_text(json.dumps(collector_manifest, indent=2) + '\n')

    save_manifests()
    if not args.execute:
        print(f'Wrote {len(jobs)} commands to {jobs_path}; no training launched.')
        return

    for job in jobs:
        if job.get('status') not in ('queued', 'complete'):
            raise SystemExit(f'Cannot execute job in state {job.get("status")}: {job}')

    manifest_lock = threading.Lock()

    def execute_job(job, index):
        job_root = out / job['run_root']
        if job_root.exists() and any(job_root.iterdir()):
            raise RuntimeError(f'Refusing non-empty run root: {job_root}')
        job_root.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(job['env_overrides'])
        device_label = job['env_overrides'].get('CUDA_VISIBLE_DEVICES', 'cpu')
        print(f'[{index}/{len(jobs)}] {job["encoding"]} d={job["width"]} '
              f'seed={job["seed"]} device=cuda:{device_label}', flush=True)
        started = time.perf_counter()
        result = subprocess.run(job['command'], cwd=job['cwd'], env=env,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        elapsed = time.perf_counter() - started
        actual_run = None
        log_file = None
        config_file = None
        if result.returncode == 0:
            if args.engine == 'grit':
                actual_run = out / job['expected_run_dir']
                if not actual_run.is_dir():
                    raise RuntimeError(f'Expected GRIT run directory is missing: {actual_run}')
            else:
                actual_run = find_gnnplus_run(job_root, job['seed'], args.split_index)
            if args.engine == 'gnnplus':
                log_file = safe_relative(actual_run / 'logging.log', out)
                config_file = safe_relative(actual_run / 'config.yaml', out)
        with manifest_lock:
            job['elapsed_seconds'] = elapsed
            job['status'] = 'complete' if result.returncode == 0 else 'failed'
            job['returncode'] = result.returncode
            group = next(g for g in groups if g['id'] == job['collector_group_id'])
            record = next(r for r in group['runs']
                          if r['encoding'] == job['encoding'] and r['seed'] == job['seed'])
            record['status'] = job['status']
            record['elapsed_seconds'] = job['elapsed_seconds']
            if result.returncode == 0:
                record['run_dir'] = safe_relative(actual_run, out)
                if args.engine == 'gnnplus':
                    record['log_file'] = log_file
                    record['config_file'] = config_file
            save_manifests()
        if result.returncode:
            raise RuntimeError(f'Job failed ({result.returncode}); plans and logs are retained.')

    pending = [job for job in jobs if job.get('status') == 'queued']
    failures = {}
    if len(devices) > 1 and len(pending) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
            future_to_job = {pool.submit(execute_job, job, index): (job, index)
                             for index, job in enumerate(pending, 1)}
            for future in concurrent.futures.as_completed(future_to_job):
                job, index = future_to_job[future]
                try:
                    future.result()
                except Exception as exc:
                    failures[(job['encoding'], job['width'], job['seed'])] = str(exc)
    else:
        for index, job in enumerate(pending, 1):
            try:
                execute_job(job, index)
            except Exception as exc:
                failures[(job['encoding'], job['width'], job['seed'])] = str(exc)

    if failures:
        details = '\n'.join(f'{key}: {msg}' for key, msg in failures.items())
        raise SystemExit(f'{len(failures)} job(s) failed:\n{details}')

    print(f'All {len(jobs)} jobs completed. Collector manifest: {collector_path}')


if __name__ == '__main__':
    main()
