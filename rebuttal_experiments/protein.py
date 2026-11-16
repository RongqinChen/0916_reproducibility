"""Run the Co-Sparsify TU pipeline with structural encodings and GNNPlus GINE."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
import types

from .common import atomic_json, atomic_npz, configuration, run_directory, sha256
from .protein_encodings import ENCODINGS, make_transform
from .protein_report import read_record, record_stem, report


def reference_state(root, config):
    root = root.resolve()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != config['reference_commit']:
        raise ValueError('Co-Sparsify checkout differs from the pinned reference commit')
    dirty = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all',
         '--', '*.py', '*.yaml'],
        text=True,
    )
    if dirty.strip():
        raise ValueError('Use an unmodified Co-Sparsify checkout; changes belong only in the encoding adapter')
    sources = [*root.rglob('*.py'), *root.rglob('*.yaml')]
    return {'commit': commit, 'sources': {str(path.relative_to(root)): sha256(path) for path in sorted(sources)}}


def load_transform_registry(reference, include_2fwl=True):
    """Expose the two upstream transforms used by the TU configuration.

    The pinned upstream package eagerly imports unrelated Sage/connectivity
    transforms, some of whose source files are absent from the pinned commit.
    Load the original Polynomials and K2FWLConnSpTransform directly.
    """
    def load(name):
        source = reference / f'transforms/{name}.py'
        spec = importlib.util.spec_from_file_location(f'_cosparsify_{name}', source)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load upstream transform: {source}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    polynomial = load('compute_polynomial')
    registry = types.ModuleType('transforms')
    registry.__file__ = str(reference / 'transforms/__init__.py')
    registry.__path__ = [str(reference / 'transforms')]
    registry.__package__ = 'transforms'
    registry.transform_dict = {
        'poly': polynomial.Polynomials,
    }
    if include_2fwl:
        registry.transform_dict['2fwl_connsp'] = load(
            'compute_2fwl_connsp').K2FWLConnSpTransform
    sys.modules['transforms'] = registry
    return registry


def load_model_registry(reference):
    """Register only the native SPPGN1 network selected by the TU config."""
    importlib.invalidate_caches()
    import models

    package = types.ModuleType('models.network')
    package.__file__ = str(reference / 'models/network/__init__.py')
    package.__path__ = [str(reference / 'models/network')]
    package.__package__ = 'models.network'
    sys.modules['models.network'] = package
    models.network = package

    name = 'models.network.sppgn1'
    source = reference / 'models/network/sppgn1.py'
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load upstream SPPGN1 model: {source}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    package.sppgn1 = module
    package.network_dict = {'sppgn1': module.SPPGN1}
    return package


def run(args):
    config = configuration(args.config)
    reference = args.reference_root or Path(config['reference_root'])
    reference = reference.resolve()
    if args.dataset not in config['datasets']:
        raise ValueError('Use the reference TU dataset versions')
    encodings = args.encodings or config['encodings']
    widths = args.poly_dims or config['poly_dims']
    folds = args.folds if args.folds is not None else config['folds']
    batch_size_override = config.get('batch_size_override')
    if batch_size_override is not None and (
            not isinstance(batch_size_override, int) or batch_size_override <= 0):
        raise ValueError('batch_size_override must be a positive integer')
    model_backend = config.get('model_backend', 'cosparsify_sppgn1')
    if model_backend not in ('cosparsify_sppgn1', 'gnnplus_gine'):
        raise ValueError('Unsupported model_backend')
    for selected, allowed in ((encodings, config['encodings']), (widths, config['poly_dims']), (folds, config['folds'])):
        if len(selected) != len(set(selected)) or not set(selected) <= set(allowed):
            raise ValueError('Invalid TU job selection')
    upstream = reference_state(reference, config)
    frozen = {**config, 'dataset': args.dataset, 'data_root': str(args.data_root.resolve()),
              'reference_root': str(reference), 'upstream': upstream}
    inputs = [args.config, reference / config['reference_config']]
    if model_backend == 'gnnplus_gine':
        gnnplus_root = Path(__file__).resolve().parents[1] / 'gnnplus_official/GNNPlus'
        inputs.extend([
            gnnplus_root / '__init__.py',
            gnnplus_root / 'layer/gine_conv_layer.py',
            gnnplus_root / 'head/san_graph.py',
        ])
    with run_directory(args.out, frozen, inputs, args.resume) as (out, identity):
        for encoding in encodings:
            for width in widths:
                pending = [fold for fold in folds
                           if read_record(out, record_stem(encoding, width, fold), identity, missing_ok=True) is None]
                if not pending:
                    continue
                command = [sys.executable, '-m', 'rebuttal_experiments', 'protein', '_native',
                           '--config', str(Path(args.config).resolve()), '--dataset', args.dataset,
                           '--data-root', str(args.data_root.resolve()), '--reference-root', str(reference),
                           '--out', str(out.resolve()), '--encoding', encoding, '--poly-dim', str(width),
                           '--identity', identity, '--folds', *map(str, pending)]
                subprocess.run(command, check=True)
                for fold in pending:
                    read_record(out, record_stem(encoding, width, fold), identity)
        completed = len(list((out / 'records').glob('*.json')))
        expected = len(config['encodings']) * len(config['poly_dims']) * len(config['folds'])
        atomic_json(out / 'summary.json', {'status': 'execution_finished', 'completed_runs': completed,
                                          'expected_runs': expected, 'full_grid_complete': completed == expected,
                                          'identity_sha256': identity, 'evaluation': config['evaluation']})


def native(args):
    """Use upstream main(), datasets, model, Lightning training and evaluators."""
    config = configuration(args.config)
    manifest = json.loads((args.out / 'manifest.json').read_text())
    if manifest['identity_sha256'] != args.identity:
        raise ValueError('Native worker identity mismatch')
    reference = args.reference_root.resolve()
    sys.path.insert(0, str(reference))
    import comet_ml  # noqa: F401 -- upstream requires this before torch/sklearn.
    model_backend = config.get('model_backend', 'cosparsify_sppgn1')
    transforms = load_transform_registry(
        reference, include_2fwl=model_backend == 'cosparsify_sppgn1')
    if model_backend == 'gnnplus_gine':
        from .protein_gnnplus import IdentityTransform, register_model
        transforms.transform_dict['2fwl_connsp'] = (
            lambda **kwargs: IdentityTransform(**kwargs))
        register_model(config)
    else:
        load_model_registry(reference)
    upstream_poly = transforms.transform_dict['poly']
    transforms.transform_dict['poly'] = lambda **kwargs: make_transform(upstream_poly, **kwargs)
    import utils
    from lighting_interface import LightningModel
    import torch

    original_load = utils.load_cfg
    def load_cfg(native_args):
        cfg = original_load(native_args)
        # Filesystem locations change; the configured physical batch override is explicit.
        cfg.dataset.path = str(args.data_root.resolve())
        cfg.train.save_dir = str(args.out.resolve() / 'native')
        if config.get('batch_size_override') is not None:
            cfg.train.batch_size = config['batch_size_override']
        return cfg
    utils.load_cfg = load_cfg

    original_reset = LightningModel._reset_parameters
    def reset_with_identity(model):
        original_reset(model)
        digest = hashlib.sha256()
        for name, value in sorted(model.net.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        model.encoding_initial_sha256 = digest.hexdigest()
    LightningModel._reset_parameters = reset_with_identity

    original_log = utils.log_final_results
    remaining = iter(args.folds)
    def capture(model, results, total_runs):
        original_log(model, results, total_runs)
        fold = next(remaining)
        stem = record_stem(args.encoding, args.poly_dim, fold)
        trainer, cfg = model.trainer, utils.cfg
        module = trainer.datamodule
        train_ids = list(map(int, module.train_dataset.indices()))
        heldout_ids = list(map(int, module.val_dataset.indices()))
        if heldout_ids != list(map(int, module.test_dataset.indices())):
            raise ValueError('Native TU scoring fold differs from the checkpoint-selection fold')
        raw = module.train_dataset._data
        digest = hashlib.sha256()
        for name in ('x', 'edge_index', 'edge_attr', 'y'):
            digest.update(name.encode())
            value = getattr(raw, name, None)
            if value is not None:
                digest.update(str((tuple(value.shape), value.dtype)).encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        prediction = args.out / 'predictions' / (stem + '.npz')
        atomic_npz(prediction, graph_ids=heldout_ids,
                   logits=torch.cat(model.step_results['test_pred']).numpy(),
                   targets=torch.cat(model.step_results['test_target']).numpy())
        checkpoint = Path(trainer.checkpoint_callback.best_model_path).resolve()
        metrics = {key.removeprefix('final/'): float(values[-1]) for key, values in results.items()}
        if model_backend == 'gnnplus_gine':
            model_settings = {'backend': model_backend, **config['gnnplus']}
        else:
            model_settings = {name: getattr(cfg.model, name) for name in (
                'name', 'hidden_dim', 'num_layers', 'mlp_depth', 'pooling', 'drop_prob',
                'jk_mode', 'norm_type', 'act_type', 'num_tasks', 'node_encoder', 'edge_encoder')}
        training = {name: getattr(cfg.train, name) for name in (
            'batch_size', 'lr', 'min_lr', 'weight_decay', 'num_warmup_epochs',
            'num_epochs', 'optimizer', 'scheduler', 'num_workers')}
        packages = {}
        for package in ('torch', 'torch-geometric', 'lightning', 'pytorch-lightning', 'numpy', 'scikit-learn', 'torchmetrics'):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        atomic_json(args.out / 'records' / (stem + '.json'), {
            'status': 'complete', 'identity_sha256': args.identity, 'dataset': args.dataset,
            'encoding': args.encoding, 'poly_dim': args.poly_dim, 'fold': fold,
            'initialization_seed': (fold + 1) * 1000,
            'initial_state_sha256': model.encoding_initial_sha256,
            'parameter_count': sum(p.numel() for p in model.net.parameters()),
            'train_graph_ids': train_ids, 'selection_and_scoring_graph_ids': heldout_ids,
            'raw_dataset_sha256': digest.hexdigest(),
            'native_model_settings': model_settings, 'native_training_settings': training,
            'monitor': cfg.log.monitor, 'monitor_mode': cfg.log.monitor_mode,
            'checkpoint_score': float(trainer.checkpoint_callback.best_model_score),
            'checkpoint': str(checkpoint.relative_to(args.out.resolve())),
            'checkpoint_sha256': sha256(checkpoint),
            'predictions': str(prediction.relative_to(args.out)), 'predictions_sha256': sha256(prediction),
            'metrics': metrics, 'environment': packages,
            'evaluation': config['evaluation']})
    utils.log_final_results = capture
    sys.argv = [str(reference / 'run_tud.py'), '--cfg', str(reference / config['reference_config']),
                '--dataname', args.dataset, '--poly_method', args.encoding, '--poly_dim', str(args.poly_dim),
                '--specified_run', *map(str, args.folds), '--local_log']
    runpy.run_path(str(reference / 'run_tud.py'), run_name='__main__')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    p = commands.add_parser('run')
    p.add_argument('--config', default='configs/rebuttal/protein.json')
    p.add_argument('--reference-root', type=Path)
    p.add_argument('--dataset', required=True, choices=['PROTEINS_full', 'ENZYMES'])
    p.add_argument('--data-root', type=Path, default=Path('data/TUD'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--encodings', nargs='+', choices=ENCODINGS)
    p.add_argument('--poly-dims', nargs='+', type=int, choices=[8, 12, 14, 16])
    p.add_argument('--folds', nargs='+', type=int, choices=range(10))
    p.add_argument('--resume', action='store_true')
    p.set_defaults(function=run)
    p = commands.add_parser('report')
    p.add_argument('--out', type=Path, required=True,
                   help='Report output directory; also the input root for an unsharded run')
    p.add_argument('--shards', type=Path, nargs='+',
                   help='Completed encoding/width shard roots to validate and merge')
    p.set_defaults(function=report)
    p = commands.add_parser('_native', help=argparse.SUPPRESS)
    for flag in ('config', 'dataset', 'encoding', 'identity'):
        p.add_argument('--' + flag, required=True)
    for flag in ('reference-root', 'data-root', 'out'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--poly-dim', type=int, required=True)
    p.add_argument('--folds', nargs='+', type=int, required=True)
    p.set_defaults(function=native)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == '__main__':
    main()
