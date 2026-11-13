"""Aggregate native Co-Sparsify scores; do not reinterpret them as independent tests."""
from __future__ import annotations

import json
from pathlib import Path

from .common import atomic_json, sha256
from .protein_encodings import COMPARISONS


def record_stem(encoding, width, fold):
    return f'{encoding}_k{width}_fold{fold:02d}'


def read_record(out, stem, identity, missing_ok=False):
    out = Path(out)
    path = out / 'records' / (stem + '.json')
    if not path.exists() and missing_ok:
        return None
    record = json.loads(path.read_text())
    if record['status'] != 'complete' or record['identity_sha256'] != identity:
        raise ValueError('Completed TU record identity mismatch')
    for field in ('checkpoint', 'predictions'):
        if sha256(out / record[field]) != record[field + '_sha256']:
            raise ValueError('Completed TU artifact checksum mismatch')
    return record


def report(args):
    import numpy as np
    manifest = json.loads((args.out / 'manifest.json').read_text())
    config = manifest['identity']['configuration']
    identity = manifest['identity_sha256']
    rows, comparisons, pairing = [], [], {}
    for width in config['poly_dims']:
        scores = {}
        for encoding in config['encodings']:
            records = [read_record(args.out, record_stem(encoding, width, fold), identity)
                       for fold in config['folds']]
            accuracy = [r['metrics']['test_acc'] for r in records]
            scores[encoding] = np.asarray(accuracy)
            for record in records:
                pair = {key: record[key] for key in (
                    'initial_state_sha256', 'parameter_count', 'train_graph_ids',
                    'selection_and_scoring_graph_ids', 'initialization_seed',
                    'raw_dataset_sha256', 'native_model_settings', 'native_training_settings', 'environment')}
                key = width, record['fold']
                if key in pairing and pairing[key] != pair:
                    raise ValueError('Encoding conditions differ beyond the structural features')
                pairing[key] = pair
            rows.append({'encoding': encoding, 'poly_dim': width, 'fold_accuracy': accuracy,
                         'accuracy_mean_percent': float(np.mean(accuracy) * 100),
                         'accuracy_std_percent': float(np.std(accuracy, ddof=0) * 100),
                         'fold_metrics': [r['metrics'] for r in records]})
        for left, right in COMPARISONS:
            if left in scores and right in scores:
                delta = (scores[left] - scores[right]) * 100
                comparisons.append({'poly_dim': width, 'comparison': f'{left}_minus_{right}',
                                    'primary': [left, right] == config['primary_comparison'],
                                    'fold_difference_pp': delta.tolist(),
                                    'mean_difference_pp': float(delta.mean()),
                                    'std_difference_pp': float(delta.std(ddof=0))})
    atomic_json(args.out / 'protein_report.json', {
        'status': 'complete', 'dataset': config['dataset'], 'identity_sha256': identity,
        'reference_commit': config['reference_commit'], 'settings': rows,
        'paired_comparisons': comparisons, 'evaluation': config['evaluation'],
        'scope': 'Native best-validation-checkpoint scores on the same held-out folds. Mean and population standard deviation over ten folds, as upstream; these are not independent-test estimates. No foldwise encoding selection or best-setting-only reporting.'})
