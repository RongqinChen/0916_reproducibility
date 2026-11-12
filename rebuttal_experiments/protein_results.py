"""Predict held-out protein labels and summarize paired out-of-fold results."""
from __future__ import annotations

import json
import time

import numpy as np

from .common import atomic_json, atomic_npz, sha256
from .protein_data import ProteinInputs, VARIANTS, load_prepared
from .protein_training import configure_torch, evaluate_graphs, build_model
from .tu_data import load_graph


def predict(args):
    """Apply each selected checkpoint to its held-out outer fold."""
    import torch
    prepared_summary, _ = load_prepared(args.prepared)
    run_manifest = json.loads((args.run / 'manifest.json').read_text())
    config = run_manifest['identity']['configuration']
    if config['prepared_identity'] != prepared_summary['identity_sha256']:
        raise ValueError('Prepared input identity mismatch')
    splits = json.loads((args.prepared / 'splits.json').read_text())
    for job in sorted((args.run / 'jobs').glob('fold*_seed*_*')):
        done = json.loads((job / 'complete.json').read_text())
        if sha256(job / 'best.pt') != done['checkpoint_sha256']:
            raise ValueError('Checkpoint checksum mismatch')
        target = job / 'predictions.npz'
        if target.exists():
            evidence = json.loads((job / 'prediction.json').read_text())
            if sha256(target) != evidence['predictions_sha256'] or evidence['checkpoint_sha256'] != done['checkpoint_sha256']:
                raise ValueError('Existing predictions do not match checkpoint')
            continue
        torch = configure_torch(done['initialization_seed'], config, args.device)
        inputs = ProteinInputs(args.prepared, done['variant'], config)
        model = build_model(inputs, config).to(args.device)
        checkpoint_data = torch.load(job / 'best.pt', map_location=args.device, weights_only=True)
        if checkpoint_data['identity_sha256'] != done['identity_sha256']:
            raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(checkpoint_data['model'])
        ids = splits[done['fold']]['test']
        started = time.monotonic()
        nll, accuracy, logits, targets = evaluate_graphs(model, inputs, ids, checkpoint_data['normalization'], args.device)
        atomic_npz(target, graph_ids=np.asarray(ids), targets=targets, logits=logits)
        atomic_json(job / 'prediction.json', {'status': 'complete', 'split': 'outer_test',
                                             'checkpoint_sha256': done['checkpoint_sha256'],
                                             'predictions_sha256': sha256(target), 'nll': nll, 'accuracy': accuracy,
                                             'seconds': time.monotonic() - started, 'device': args.device,
                                             'prepared_identity': prepared_summary['identity_sha256']})


def report(args):
    """Pool held-out predictions and compare variants using paired seeds."""
    prepared = json.loads((args.prepared / 'summary.json').read_text())
    splits = json.loads((args.prepared / 'splits.json').read_text())
    groups, seen, costs, pairing, common_config = {}, set(), [], {}, None
    for run in args.runs:
        manifest = json.loads((run / 'manifest.json').read_text())
        config = manifest['identity']['configuration']
        scientific = {key: value for key, value in config.items()
                      if key not in ('device', 'selected_seeds', 'selected_folds', 'selected_variants')}
        if common_config is None:
            common_config = scientific
        elif scientific != common_config:
            raise ValueError('Reports cannot merge different scientific configurations')
        if config['prepared_identity'] != prepared['identity_sha256']:
            raise ValueError('Runs used different prepared data')
        for job in sorted((run / 'jobs').glob('fold*_seed*_*')):
            done = json.loads((job / 'complete.json').read_text())
            prediction = json.loads((job / 'prediction.json').read_text())
            if (sha256(job / 'predictions.npz') != prediction['predictions_sha256']
                    or sha256(job / 'best.pt') != done['checkpoint_sha256']
                    or prediction['checkpoint_sha256'] != done['checkpoint_sha256']
                    or sha256(job / 'initial.pt') != done['initial_file_sha256']
                    or sha256(job / 'history.json') != done['history_sha256']
                    or sha256(job / 'normalization.json') != done['normalization_sha256']):
                raise ValueError('Training/prediction provenance mismatch')
            key = done['fold'], done['seed'], done['variant']
            if key in seen:
                raise ValueError('Duplicate job across report inputs')
            seen.add(key)
            pair = pairing.setdefault((done['fold'], done['seed']), [])
            pair.append((done['initial_state_sha256'], done['parameter_count'], done['optimizer_updates'],
                         done['device_name'], done['torch_version'], job))
            with np.load(job / 'predictions.npz', allow_pickle=False) as data:
                if data['graph_ids'].tolist() != splits[done['fold']]['test']:
                    raise ValueError('Prediction graph IDs differ from the frozen outer test')
                groups.setdefault((done['seed'], done['variant']), []).append(
                    (data['graph_ids'].copy(), data['targets'].copy(), data['logits'].copy()))
            costs.append({key: done[key] for key in ('fold', 'seed', 'variant', 'training_seconds',
                                                     'sampled_process_rss_peak_bytes', 'gpu_peak_allocated_bytes')})
    required = {(fold, seed, variant) for fold in range(len(splits)) for seed in config['seeds'] for variant in VARIANTS}
    if seen != required:
        raise ValueError(f'Full report requires exactly the frozen job matrix; missing={sorted(required-seen)}, extra={sorted(seen-required)}')
    for pair in pairing.values():
        if len({item[:-1] for item in pair}) != 1:
            raise ValueError('Variants did not share initial state, parameter count, update budget and device type')
        histories = [json.loads((item[-1] / 'history.json').read_text()) for item in pair]
        if any([x['train_order'] for x in h] != [x['train_order'] for x in histories[0]] for h in histories[1:]):
            raise ValueError('Variants used different actual graph orders')
    results = []
    target_reference = np.asarray([int(load_graph(args.prepared / 'graphs' / f'graph{i:04d}.npz')['y'])
                                   for i in range(prepared['graph_count'])])
    for (seed, variant), chunks in sorted(groups.items()):
        ids = np.concatenate([x[0] for x in chunks])
        targets = np.concatenate([x[1] for x in chunks])
        logits = np.concatenate([x[2] for x in chunks])
        order = np.argsort(ids)
        if ids[order].tolist() != list(range(prepared['graph_count'])):
            raise ValueError('OOF predictions must cover every record exactly once')
        logits, targets = logits[order], targets[order]
        if not np.array_equal(targets, target_reference):
            raise ValueError('Saved prediction targets differ from prepared data')
        shifted = logits - logits.max(1, keepdims=True)
        log_probability = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
        results.append({'seed': seed, 'variant': variant,
                        'accuracy': float(np.mean(logits.argmax(1) == targets)),
                        'nll': float(-log_probability[np.arange(len(targets)), targets].mean())})
    lookup = {(r['seed'], r['variant']): r for r in results}
    comparisons = [{'seed': seed, 'comparison': f'joint_minus_{other}',
                    'accuracy_difference': lookup[seed, 'joint']['accuracy'] - lookup[seed, other]['accuracy'],
                    'nll_difference': lookup[seed, 'joint']['nll'] - lookup[seed, other]['nll']}
                   for seed in config['seeds'] for other in ('base', 'nonjoint')]
    construction = [json.loads((args.prepared / 'records' / f'graph{i:04d}.json').read_text())['construction']
                    for i in range(prepared['graph_count'])]
    seed_summary = {}
    for variant in VARIANTS:
        seed_summary[variant] = {}
        for metric in ('accuracy', 'nll'):
            values = [row[metric] for row in results if row['variant'] == variant]
            seed_summary[variant][metric] = {'mean': float(np.mean(values)),
                                             'sample_std': float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                             'seed_count': len(values)}
    atomic_json(args.out, {'status': 'complete', 'prepared_identity': prepared['identity_sha256'],
                           'per_seed_oof': results, 'per_variant_seed_summary': seed_summary,
                           'paired_comparisons': comparisons,
                           'training_costs': costs,
                           'feature_construction_costs': {
                               'base_seconds': sum(x['base_seconds'] for x in construction),
                               'joint_extra_seconds': sum(x['joint_extra_seconds'] for x in construction),
                               'nonjoint_extra_seconds': sum(x['nonjoint_extra_seconds'] for x in construction)},
                           'scope': 'One fixed protein task. Folds×seeds are not independent datasets. Equal width/updates do not imply equal construction compute or a 3-FWL necessity result.'})
