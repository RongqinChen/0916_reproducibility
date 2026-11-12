"""Train paired protein runs and select checkpoints using validation data."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time

from .common import (Budget, atomic_json, configuration, hard_wall_limit,
                     run_directory, sha256)
from .protein_data import ProteinInputs, VARIANTS, load_prepared


def configure_torch(seed, config, device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.set_num_threads(config['torch_threads'])
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    return torch


def build_model(inputs, config):
    from .protein_model import ControlledClassifier
    return ControlledClassifier(inputs.metadata['node_feature_dim'], inputs.metadata['edge_feature_dim'],
                                len(inputs.metadata['classes']), config)


def model_state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate_graphs(model, inputs, ids, normalization, device):
    import torch
    model.eval()
    logits = []
    with torch.no_grad():
        for index in ids:
            logits.append(inputs.forward(model, index, normalization, device).cpu().double())
    logits = torch.stack(logits)
    targets = torch.tensor([int(inputs.graph(i)['y']) for i in ids], dtype=torch.long)
    nll = torch.nn.functional.cross_entropy(logits, targets).item()
    accuracy = (logits.argmax(1) == targets).double().mean().item()
    return nll, accuracy, logits.numpy(), targets.numpy()


def train_one_run(prepared, job, variant, split, seed, config, device, identity):
    """Train one fold/seed/variant; select by validation NLL without using test."""
    inputs = ProteinInputs(prepared, variant, config)
    actual_seed = config['initialization_seed'] + 1009 * split['fold'] + seed
    torch = configure_torch(actual_seed, config, device)
    model = build_model(inputs, config).to(device)
    initial_hash = model_state_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    normalization = inputs.normalization(split['train'])
    atomic_json(job / 'normalization.json', normalization)
    torch.save({'model': model.state_dict(), 'initialization_seed': actual_seed}, job / 'initial.pt')
    best, selected_epoch = float('inf'), None
    history = []
    started = time.monotonic()
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    import psutil
    peak_rss = psutil.Process().memory_info().rss
    updates = 0
    budget = Budget(config['seconds_per_job'], config['training_memory_gib'])
    for epoch in range(config['epochs']):
        model.train()
        order = list(split['train'])
        random.Random(actual_seed + 1_000_003 * (epoch + 1)).shuffle(order)
        train_nll = 0.0
        for start in range(0, len(order), config['batch_size']):
            batch = order[start:start + config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                budget.check()
                logits = inputs.forward(model, index, normalization, device)
                target = torch.tensor([int(inputs.graph(index)['y'])], dtype=torch.long, device=device)
                loss = torch.nn.functional.cross_entropy(logits[None], target)
                (loss / len(batch)).backward()
                train_nll += float(loss.detach().cpu())
                peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            optimizer.step()
            updates += 1
        validation_nll, validation_accuracy, _, _ = evaluate_graphs(
            model, inputs, split['validation'], normalization, device)
        if validation_nll < best:  # Unrounded metric; exact ties keep earlier epoch.
            best, selected_epoch = validation_nll, epoch
            temporary = job / 'best.pt.partial'
            torch.save({'model': model.state_dict(), 'selected_epoch': epoch,
                        'validation_nll': validation_nll, 'normalization': normalization,
                        'identity_sha256': identity, 'variant': variant,
                        'fold': split['fold'], 'seed': seed}, temporary)
            temporary.replace(job / 'best.pt')
        history.append({'epoch': epoch, 'train_order': order,
                        'train_nll': train_nll / len(order),
                        'validation_nll': validation_nll, 'validation_accuracy': validation_accuracy,
                        'updates_completed': updates})
        atomic_json(job / 'history.json', history)
        print(f'{job.name} epoch={epoch} val_nll={validation_nll:.12g}', flush=True)
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    atomic_json(job / 'complete.json', {
        'status': 'training_complete', 'identity_sha256': identity, 'variant': variant,
        'fold': split['fold'], 'seed': seed, 'initialization_seed': actual_seed,
        'initial_state_sha256': initial_hash, 'initial_file_sha256': sha256(job / 'initial.pt'),
        'checkpoint_sha256': sha256(job / 'best.pt'), 'history_sha256': sha256(job / 'history.json'),
        'normalization_sha256': sha256(job / 'normalization.json'),
        'selected_epoch': selected_epoch, 'selection_validation_nll': best,
        'epochs_completed': config['epochs'], 'optimizer_updates': updates,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'training_seconds': time.monotonic() - started,
        'sampled_process_rss_peak_bytes': peak_rss,
        'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
        'device': device, 'device_name': torch.cuda.get_device_name(device) if device.startswith('cuda') else 'cpu',
        'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
        'test_was_not_evaluated': True})


def train(args):
    config = configuration(args.config)
    prepared_summary, prepared_manifest = load_prepared(args.prepared)
    config = {**config, 'dataset': prepared_summary['dataset']}
    prepared_config = prepared_manifest['identity']['configuration']
    for key in ('folds', 'validation_fraction', 'split_seed'):
        if config[key] != prepared_config[key]:
            raise ValueError(f'Training split setting {key} differs from prepared data')
    split_list = json.loads((args.prepared / 'splits.json').read_text())
    if len(split_list) != config['folds']:
        raise ValueError('Prepared folds do not match training configuration')
    seeds = args.seeds if args.seeds is not None else config['seeds']
    folds = args.folds if args.folds is not None else list(range(config['folds']))
    variants = args.variants if args.variants is not None else list(VARIANTS)
    if (len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds)
            or len(set(folds)) != len(folds) or len(set(variants)) != len(variants)
            or any(fold not in range(config['folds']) for fold in folds)):
        raise ValueError('Invalid job selection')
    frozen = {**config, 'prepared_identity': prepared_summary['identity_sha256'],
              'prepared_manifest_sha256': sha256(args.prepared / 'manifest.json'),
              'device': args.device, 'selected_seeds': seeds, 'selected_folds': folds,
              'selected_variants': variants}
    with run_directory(args.out, frozen, [args.config, args.prepared / 'splits.json'], args.resume) as (out, identity):
        atomic_json(out / 'prepared.json', {'path': str(args.prepared.resolve()),
                                          'identity': prepared_summary['identity_sha256'],
                                          'summary': prepared_summary})
        for fold in folds:
            for seed in seeds:
                for variant in variants:
                    job = out / 'jobs' / f'fold{fold}_seed{seed}_{variant}'
                    done = job / 'complete.json'
                    if done.exists():
                        record = json.loads(done.read_text())
                        if record['identity_sha256'] != identity or sha256(job / 'best.pt') != record['checkpoint_sha256']:
                            raise ValueError('Completed training record identity/checkpoint mismatch')
                        continue
                    if job.exists():
                        # Only an explicit retry discards this incomplete job and starts over.
                        if not args.retry_incomplete:
                            raise RuntimeError(f'Incomplete job {job}; inspect it, then use explicit --retry-incomplete to discard it and restart')
                        shutil.rmtree(job)
                    job.mkdir(parents=True)
                    atomic_json(job / 'started.json', {'identity_sha256': identity, 'fold': fold,
                                                       'seed': seed, 'variant': variant})
                    with hard_wall_limit(config['seconds_per_job']):
                        train_one_run(args.prepared, job, variant, split_list[fold], seed, config, args.device, identity)
        atomic_json(out / 'summary.json', {'status': 'training_finished', 'test_predictions': 'not_run',
                                          'jobs': len(folds) * len(seeds) * len(variants),
                                          'identity_sha256': identity})
